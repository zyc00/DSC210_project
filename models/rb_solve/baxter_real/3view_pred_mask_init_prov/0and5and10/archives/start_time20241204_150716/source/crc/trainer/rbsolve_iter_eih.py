import PIL.Image as Image
import glob
import pdb

import mplib
import torch
import os.path as osp

import cv2
import imageio
import loguru
import sapien.core as sapien
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import transforms3d
import trimesh
from PIL import Image
from dl_ext.timer import EvalTime
from pymp import Planner
from wis3d import Wis3D

from crc.modeling.models.rb_solve.collision_checker import CollisionChecker
from crc.modeling.models.rb_solve.space_explorer import SpaceExplorer

from crc.data import make_data_loader
from crc.modeling.models.rb_solve.space_explorer_eih import SpaceExplorerEih
from crc.solver.build import make_optimizer, make_lr_scheduler
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from loguru import logger
from sapien.utils import Viewer
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import tqdm

from crc.modeling.build import build_model
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.trainer.base import BaseTrainer
from crc.trainer.utils import *
from crc.utils import plt_utils, utils_3d, render_api, sam_api, dkm_api
from crc.utils.os_utils import number_of_monitors, archive_runs
from crc.utils.utils_3d import roty_np, rotx_np
from crc.utils.vis3d_ext import Vis3D


class RBSolverIterEiHTrainer(BaseTrainer):
    def __init__(self, cfg):
        self.output_dir = cfg.output_dir
        self.num_epochs = cfg.solver.num_epochs
        self.begin_epoch = 0
        self.max_lr = cfg.solver.max_lr
        self.save_every = cfg.solver.save_every
        self.save_mode = cfg.solver.save_mode
        self.save_freq = cfg.solver.save_freq

        self.epoch_time_am = AverageMeter()
        self.cfg = cfg
        self._tb_writer = None
        self.state = TrainerState.BASE
        self.global_steps = 0
        self.best_val_loss = 100000
        self.val_loss = 100000
        self.qposes = np.array(self.cfg.model.rbsolver_iter.start_qpos)[None]
        self.sampled_indices = [self.cfg.model.rbsolver_iter.start_index]  # for data pool
        if self.cfg.model.rbsolver_iter.use_realarm.enable is True:
            from crc.utils.realsense_api import setup_realsense
            self.pipeline, self.profile, self.align = setup_realsense()
            ip = self.cfg.model.rbsolver_iter.use_realarm.ip
            from xarm import XArmAPI
            arm = XArmAPI(ip)
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(state=0)
            self.arm = arm
        # self.cc = CollisionChecker(self.cfg.model.space_explorer)

    def train(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        metric_ams = {}
        bar = tqdm.tqdm(self.train_dl, leave=False) if is_main_process() and len(self.train_dl) > 1 else self.train_dl
        begin = time.time()
        for batchid, batch in enumerate(bar):
            self.optimizer.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = batchid
            batch['epoch'] = epoch
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            # print(epoch, loss.item())
            loss.backward()
            if self.cfg.solver.grad_noise.enable is True:
                print("it", self.global_steps, "gradient", list(self.model.parameters())[0].grad)
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            # record and plot loss and metrics
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if 'metrics' in output:
                for k, v in output['metrics'].items():
                    reduced_s = reduce_loss(v)
                    metrics[k] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                if self.global_steps % 200 == 0 and self.cfg.solver.image_grid_on_tb_writer:
                    self.image_grid_on_tb_writer(output['rendered_masks'], self.tb_writer,
                                                 'train/rendered_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['ref_masks'], self.tb_writer,
                                                 'train/ref_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['error_maps'], self.tb_writer,
                                                 "train/error_maps", self.global_steps)
                    print()
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
                for k, v in metrics.items():
                    if k not in metric_ams.keys():
                        metric_ams[k] = AverageMeter()
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                if isinstance(bar, tqdm.tqdm):
                    bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process() and epoch % self.cfg.solver.log_interval == 0:
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()
        return metric_ams

    def image_grid_on_tb_writer(self, images, tb_writer, tag, global_step):
        plt_utils.image_grid(images, show=False)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        tb_writer.add_image(tag, np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), global_step)
        plt.close("all")

    def do_fit(self, explore_it):
        os.makedirs(self.output_dir, exist_ok=True)
        num_epochs = self.num_epochs
        begin = time.time()

        for epoch in tqdm.trange(num_epochs):
            metric_ams = self.train(epoch)
            synchronize()
            if not self.save_every and epoch % self.cfg.solver.val_freq == 0:
                self.val_loss = self.val(epoch)
                synchronize()
            self.try_to_save(explore_it * num_epochs + epoch, 'epoch')

            synchronize()
        if is_main_process():
            logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))
        return metric_ams

    def fit(self):
        self.clean_data_dir()
        evaltime = EvalTime(disable=not self.cfg.evaltime)
        for explore_it in range(self.cfg.solver.explore_iters):
            evaltime("")
            self.capture_data()
            evaltime("capture data")
            print("capture data")
            self.rebuild(explore_it)
            evaltime("rebuild")
            print("rebuild")
            metric_ams = self.do_fit(explore_it)
            evaltime("do fit")
            for k, am in metric_ams.items():
                self.tb_writer.add_scalar("val/" + k, am.avg, explore_it)
            to_zero = explore_it == self.cfg.solver.explore_iters - 1
            if to_zero: break
            self.explore_next_state(explore_it, to_zero)
            evaltime("explore next state")

    def capture_data(self):
        data_pool = self.cfg.model.rbsolver_iter.data_pool
        if self.cfg.model.rbsolver_iter.use_realarm.enable is True:
            loguru.logger.info("Use realsense")
            self.realsense_capture_data()
            print()
        elif len(data_pool) == 0:
            loguru.logger.info("No data pool, use sapien to synthesize data")
            self.sapien_sim_data()
        else:
            loguru.logger.info("Use data pool")
            use_index = self.sampled_indices[-1]
            data_point = data_pool[use_index]
            img_path = osp.join(data_point, "color/000000.png")
            anno_mask_path = osp.join(data_point, "anno_mask/000000.png")
            pred_mask_path = osp.join(data_point, "pred_mask/000000.png")
            qpos_path = osp.join(data_point, "qpos/000000.txt")
            K_path = glob.glob(osp.join(data_point, "*K.txt"))[0]
            Tc_c2b_path = osp.join(data_point, "Tc_c2b.txt")
            out_index = len(self.sampled_indices) - 1

            outdir = self.cfg.model.rbsolver_iter.data_dir
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(osp.join(outdir, "color"), exist_ok=True)
            os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
            os.makedirs(osp.join(outdir, "anno_mask"), exist_ok=True)
            os.makedirs(osp.join(outdir, "pred_mask"), exist_ok=True)
            os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)
            os.system(f"cp {K_path} {outdir}/")
            os.system(f"cp {Tc_c2b_path} {osp.join(outdir, 'Tc_c2b.txt')}")
            os.system(f"cp {img_path} {osp.join(outdir, 'color', f'{out_index:06d}.png')}")
            os.system(f"cp {anno_mask_path} {osp.join(outdir, 'anno_mask', f'{out_index:06d}.png')}")
            if osp.exists(pred_mask_path):
                os.system(f"cp {pred_mask_path} {osp.join(outdir, 'pred_mask', f'{out_index:06d}.png')}")
            os.system(f"cp {qpos_path} {osp.join(outdir, 'qpos', f'{out_index:06d}.txt')}")
            print()

    def sapien_sim_data(self):
        engine = sapien.Engine()
        cfg = self.cfg.sim_hec
        if cfg.rt.enable:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(16)
            sapien.render.set_ray_tracing_denoiser("oidn")

        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = cfg.urdf_path
        builder = loader.load_file_as_articulation_builder(urdf_path)
        robot = builder.build(fix_root_link=True)
        assert robot, 'URDF not loaded.'
        cube_map = "SaintPetersSquare2"
        cube_map = "SaintLazarusChurch"
        scene.set_environment_map_from_files(
            f"data/cubemap/{cube_map}/posx.jpg",
            f"data/cubemap/{cube_map}/negx.jpg",
            f"data/cubemap/{cube_map}/posy.jpg",
            f"data/cubemap/{cube_map}/negy.jpg",
            f"data/cubemap/{cube_map}/posz.jpg",
            f"data/cubemap/{cube_map}/negz.jpg", )

        if cfg.add_ground:
            rm = renderer.create_material()
            rm.set_base_color(np.array([0.0, 0.0, 0.0, 1.0]))
            rm.set_specular(0.3)
            rm.set_roughness(0.8)
            scene.add_ground(0.0, render=True, render_material=rm, render_half_size=[1, 1])
        if cfg.add_desk_cube.enable:
            actor_builder = scene.create_actor_builder()
            half_sizes = cfg.add_desk_cube.half_sizes
            colors = cfg.add_desk_cube.colors
            poses = cfg.add_desk_cube.poses
            assert len(half_sizes) == len(colors) == len(poses)
            for half_size, color, pose in zip(half_sizes, colors, poses):
                actor_builder.add_box_collision(half_size=half_size)
                mtl = renderer.create_material()
                mtl.set_base_color(color + [1])
                actor_builder.add_box_visual(half_size=half_size, material=mtl)
                box = actor_builder.build()
                box.set_pose(sapien.Pose(p=pose))
        scene.set_ambient_light([1] * 3)
        scene.add_directional_light([0, 1, -1], [1.0, 1.0, 1.0], position=[0, 0, 2], shadow=True)
        # pts = trimesh.primitives.Sphere(radius=2.0).sample(cfg.n_point_light)
        sphere = trimesh.primitives.Sphere(radius=2.0)
        pts = trimesh.sample.sample_surface(sphere, cfg.n_point_light, seed=777)[0]
        for pt in pts:
            if pt[2] > 0.5:
                scene.add_point_light(pt, [1, 1, 1], shadow=True)

        outdir = self.cfg.model.rbsolver_iter.data_dir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, "color"), exist_ok=True)
        os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
        os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
        os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)

        near, far = 0.1, 100
        width, height = 1280, 720
        if cfg.Tc_c2e_npy_path != "":
            Tc_c2e = np.load(cfg.Tc_c2e_npy_path)[cfg.Tc_c2e_index]
        else:
            Tc_c2e = np.array(cfg.Tc_c2e)
        np.savetxt(osp.join(outdir, "Tc_c2e.txt"), Tc_c2e)

        coord_convert = utils_3d.Rt_to_pose(roty_np(-np.pi / 2) @ rotx_np(np.pi / 2))
        Te_e2c = np.linalg.inv(Tc_c2e) @ coord_convert
        rot = transforms3d.quaternions.mat2quat(Te_e2c[:3, :3])
        trans = Te_e2c[:3, 3]
        # mount = scene.create_actor_builder().build_kinematic()
        # mount.add_component(robot.get_links()[8])
        # mount = scene.create_actor_builder().build_physx_component(robot.get_links()[8])
        # robot.get_links()[8].entity.add_component(mount)
        camera = scene.add_mounted_camera(
            name="camera",
            mount=robot.get_links()[8].entity,
            pose=sapien.Pose(trans, rot),
            width=width,
            height=height,
            fovy=np.deg2rad(135),
            near=near,
            far=far,
        )

        fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
        camera.set_perspective_parameters(0.01, 100, fx, fy, cx, cy, 0.0)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        np.savetxt(osp.join(outdir, "K.txt"), K)

        active_joints = robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_properties(stiffness=1e6, damping=1e4)

        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="rbsolver_iter_gen_data_for_hec",
                      auto_increase=True, enable=self.cfg.dbg)
        index = len(self.qposes) - 1
        active_joints = robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_properties(stiffness=1e6, damping=1e4)

        nmonitors = number_of_monitors()
        if self.cfg.dbg and nmonitors > 0 and not 'PYCHARM_HOSTED' in os.environ:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(x=-2, y=0, z=1)
            viewer.set_camera_rpy(r=0, p=-0.3, y=0)
        qpose: np.ndarray = self.qposes[-1]
        qpose = np.append(qpose, [0] * (robot.dof - qpose.shape[0]))
        robot.set_qpos(np.array(qpose, dtype=np.float32))
        scene.update_render()
        vis3d.add_xarm(qpose)
        if self.cfg.dbg and nmonitors > 0 and not 'PYCHARM_HOSTED' in os.environ:
            viewer.paused = True
            while not viewer.closed:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
                scene.update_render()
                viewer.render()
        camera.take_picture()

        Tb_b2c = robot.get_links()[8].get_pose().to_transformation_matrix() @ np.linalg.inv(Tc_c2e)
        vis3d.add_xarm(qpose)
        vis3d.add_camera_pose(Tb_b2c, name='camera')
        rgba = camera.get_picture("Color")  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        img = rgba_img[:, :, :3]
        vis3d.add_image(img, name='img')

        seg_labels = camera.get_picture('Segmentation')  # [H, W, 4]
        rendered_mask = np.logical_and(seg_labels[..., 1] > 1, seg_labels[..., 1] < 10)
        vis3d.add_image(rendered_mask, name='mask')
        imageio.imsave(osp.join(outdir, f"gt_mask/{index:06d}.png"),
                       np.repeat(rendered_mask[:, :, None], 3, axis=-1).astype(np.uint8) * 255)
        tmp = plt_utils.vis_mask(img, rendered_mask.astype(np.uint8), color=[255, 0, 0])
        vis3d.add_image(tmp, name="hover")
        imageio.imsave(osp.join(outdir, f"color/{index:06d}.png"), img)
        np.savetxt(osp.join(outdir, f"qpos/{index:06d}.txt"), qpose[:7])

    def segment(self, index):
        outdir = self.cfg.model.rbsolver_iter.data_dir
        image_path = osp.join(outdir, f"color/{index:06d}.png")

        segmentor = self.cfg.model.rbsolver_iter.segmentor
        if segmentor == "pointrend":
            POINTREND_DIR = "./detectron2/projects/PointRend"
            pointrend_cfg_file = self.cfg.model.rbsolver_iter.pointrend_cfg_file
            pointrend_model_weight = self.cfg.model.rbsolver_iter.pointrend_model_weight
            config_file = osp.join(POINTREND_DIR, pointrend_cfg_file)
            model_weight = osp.join(POINTREND_DIR, pointrend_model_weight)
            from crc.utils.pointrend_api import pointrend_api
            pred_binary_mask = pointrend_api(config_file, model_weight, image_path)
        elif segmentor == "promptsam":
            # if index == 0:
            #     pred_binary_mask = cv2.imread(
            #         "data/rb_solver_iter/real_xarm/example_eih_autosam/archives/start_time20231019_135131/pred_mask/000000.png",
            #         2) > 0
            #     print("load ")
            #     print("load ")
            #     print("load ")
            #     print("load ")
            #     print("load ")
            #     print("load ")
            # else:
            from crc.utils.prompt_drawer import PromptDrawer
            prompt_drawer = PromptDrawer(screen_scale=1.5,
                                         sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth")
            img = imageio.imread_v2(image_path)
            _, _, pred_binary_mask = prompt_drawer.run(img)
        elif segmentor == "autosam":
            img = imageio.imread_v2(image_path)
            H, W, _ = img.shape
            K = np.loadtxt(osp.join(outdir, "K.txt"))
            # todo: deduce box
            if index == 0:
                # deduce from init pose
                init_Tc_c2e = np.array(self.cfg.model.rbsolver.init_Tc_c2e)
                Tc_c2e = init_Tc_c2e
            else:
                prev_Tc_c2e = utils_3d.se3_exp_map(self.model.dof[None]).permute(0, 2, 1)[0].detach().cpu().numpy()
                Tc_c2e = prev_Tc_c2e

            qpos = np.loadtxt(osp.join(outdir, f"qpos/{index:06d}.txt"))
            sk = SAPIENKinematicsModelStandalone("data/xarm7_with_gripper_reduced_dof.urdf")
            Tb_b2e = sk.compute_forward_kinematics(qpos, 8).to_transformation_matrix()
            Tc_c2b = Tc_c2e @ np.linalg.inv(Tb_b2e)
            eep = None
            if hasattr(self.model, 'eep'):
                if self.cfg.model.rbsolver.optim_eep_xy is True:
                    eep = utils_3d.se3_exp_map(self.model.eep[None]).permute(0, 2, 1)[0].detach().cpu().numpy()
                else:
                    eep = utils_3d.se3_exp_map(torch.cat([torch.zeros(2, dtype=torch.float, device=self.model.eep.device),
                                                          self.model.eep], dim=0)[None]).permute(0, 2, 1)[0].detach().cpu().numpy()

            rendered_masks = render_api.nvdiffrast_render_xarm_api("data/sapien_packages/xarm7/xarm_urdf/xarm7_gripper.urdf",
                                                                   Tc_c2b, qpos, H, W, K,
                                                                   return_sum=False,
                                                                   link_indices=[0, 1, 2, 3, 4, 5, 6, 7,
                                                                                 9, 10, 11, 12, 13, 14, 15],
                                                                   Te_e2p=eep)
            sam_masks = []
            autosam_links = [1, 2]
            for al in autosam_links:
                for i in range(len(rendered_masks) - al + 1):
                    rms = rendered_masks[i:i + al]
                    if any([m.sum() > 0 for m in rms]):
                        joint_mask = np.logical_or.reduce(rms)
                        sam_mask, dbg_img = sam_api.SAMAPI.segment_api(img, joint_mask,
                                                                       box_scale=self.cfg.model.rbsolver_iter.autosam_box_scale,
                                                                       point_center=self.cfg.model.rbsolver_iter.autosam_point_center,
                                                                       dbg=True)
                        dbg_img_out_dir = osp.join(self.cfg.model.rbsolver_iter.data_dir, 'autosam_dbg')
                        os.makedirs(dbg_img_out_dir, exist_ok=True)
                        out_index = len(glob.glob(osp.join(dbg_img_out_dir, "*.png")))
                        imageio.imsave(osp.join(dbg_img_out_dir, f"{out_index:06d}.png"), dbg_img)
                        sam_masks.append(sam_mask)
            sam_mask = np.stack(sam_masks).clip(0, 1)
            pred_binary_mask = sam_mask.sum(0).clip(0, 1).astype(np.uint8)
            # gripper mask
            if self.cfg.model.rbsolver.ignore_gripper_mask is True:
                sam_masks = []
                autosam_links = [1, 2]
                for al in autosam_links:
                    for i in range(8, len(rendered_masks) - al + 1):
                        rms = rendered_masks[i:i + al]
                        if any([m.sum() > 0 for m in rms]):
                            joint_mask = np.logical_or.reduce(rms)
                            sam_mask, dbg_img = sam_api.SAMAPI.segment_api(img, joint_mask,
                                                                           box_scale=self.cfg.model.rbsolver_iter.autosam_box_scale,
                                                                           point_center=self.cfg.model.rbsolver_iter.autosam_point_center,
                                                                           dbg=True)
                            dbg_img_out_dir = osp.join(self.cfg.model.rbsolver_iter.data_dir, 'autosam_dbg')
                            os.makedirs(dbg_img_out_dir, exist_ok=True)
                            out_index = len(glob.glob(osp.join(dbg_img_out_dir, "*.png")))
                            imageio.imsave(osp.join(dbg_img_out_dir, f"{out_index:06d}.png"), dbg_img)
                            sam_masks.append(sam_mask)
                sam_mask = np.stack(sam_masks).clip(0, 1)
                pred_gripper_binary_mask = sam_mask.sum(0).clip(0, 1).astype(np.uint8)
        else:
            raise NotImplementedError()
        sam_api.SAMAPI.destory()
        outpath = osp.join(outdir, "pred_mask", osp.basename(image_path))
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        imageio.imsave(outpath, (pred_binary_mask * 255).astype(np.uint8)[:, :, None].repeat(3, axis=-1))
        if self.cfg.model.rbsolver.ignore_gripper_mask is True:
            outpath = osp.join(outdir, "pred_gripper_mask", osp.basename(image_path))
            os.makedirs(osp.dirname(outpath), exist_ok=True)
            imageio.imsave(outpath, (pred_gripper_binary_mask * 255).astype(np.uint8)[:, :, None].repeat(3, axis=-1))
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="capture_data",
                      auto_increase=False, enable=True)
        vis3d.set_scene_id(index)
        rgb = imageio.imread_v2(image_path)
        tmp = plt_utils.vis_mask(rgb, pred_binary_mask.astype(np.uint8), [255, 0, 0])
        vis3d.add_image(tmp, name='hover_pred_mask')
        if self.cfg.model.rbsolver.ignore_gripper_mask is True:
            tmp = plt_utils.vis_mask(rgb, pred_gripper_binary_mask.astype(np.uint8), [0, 255, 0])
            vis3d.add_image(tmp, name='gripper_pred_mask')

    def realsense_capture_data(self):
        import pyrealsense2 as rs
        outdir = self.cfg.model.rbsolver_iter.data_dir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, "color"), exist_ok=True)
        os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
        os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
        os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)

        profile, align, pipeline = self.profile, self.align, self.pipeline
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]])
        np.savetxt(osp.join(outdir, "K.txt"), K)

        np.savetxt(osp.join(outdir, "Tc_c2e.txt"), np.eye(4))  # dummy

        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="capture_data",
                      auto_increase=False, enable=True)
        index = len(self.qposes) - 1
        vis3d.set_scene_id(index)

        qpose = self.qposes[-1]
        vis3d.add_xarm(qpose.tolist() + [0, 0])
        arm = self.arm
        speed = self.cfg.model.rbsolver_iter.use_realarm.speed
        if len(self.qposes) == 1:
            arm.set_servo_angle(angle=qpose, is_radian=True, speed=speed, wait=True)
        else:
            plan_qposes = self.plan_result['position']
            if not self.cfg.model.rbsolver_iter.use_realarm.speed_control:
                for plan_qpose in tqdm.tqdm(plan_qposes):
                    arm.set_servo_angle(angle=plan_qpose, is_radian=True, speed=speed, wait=True)
            else:
                safety_factor = self.cfg.model.rbsolver_iter.use_realarm.safety_factor
                timestep = self.cfg.model.rbsolver_iter.use_realarm.timestep
                arm.set_mode(4)
                arm.set_state(state=0)
                time.sleep(1)
                recorded_frames = []
                for ti, target_qpos in enumerate(tqdm.tqdm(plan_qposes)):
                    code, joint_state = arm.get_joint_states(is_radian=True)
                    joint_pos = joint_state[0][:7]
                    diff = target_qpos[:7] - joint_pos
                    qvel = diff / timestep / safety_factor
                    qvel_cliped = np.clip(qvel, -0.3, 0.3)
                    arm.vc_set_joint_velocity(qvel_cliped, is_radian=True, is_sync=True, duration=timestep)
                    if self.cfg.model.rbsolver_iter.use_realarm.record_video.enable is True and ti % 50 == 0:
                        frames = pipeline.wait_for_frames()
                        video_outdir = self.cfg.model.rbsolver_iter.use_realarm.record_video.outdir
                        os.makedirs(video_outdir, exist_ok=True)

                        frames = align.process(frames)
                        color_frame = frames.get_color_frame()

                        if color_frame:
                            color_image = np.asanyarray(color_frame.get_data())
                            recorded_frames.append(color_image)
                            # rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                            # imageio.imwrite(osp.join(video_outdir, f"{out_frame_idx:06d}.png"), rgb)
                    time.sleep(timestep)
                arm.set_mode(0)
                time.sleep(1)
                if len(recorded_frames) > 0:
                    for rf in tqdm.tqdm(recorded_frames, desc="saving recorded frames"):
                        out_frame_idx = len(os.listdir(video_outdir))
                        cv2.imwrite(osp.join(video_outdir, f"{out_frame_idx:06d}.png"), rf)
                print()
            print()
        time.sleep(self.cfg.model.rbsolver_iter.use_realarm.wait_time)
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            imageio.imwrite(osp.join(outdir, f"color/{index:06d}.png"), rgb)
            vis3d.add_image(rgb, name='img')
            imageio.imsave(osp.join(outdir, f"depth/{index:06d}.png"), depth_image)
            # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            # pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth_image / 1000.0)
            # vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=3, name='depth')

            break
        retcode, curr_radian = arm.get_servo_angle(is_radian=True)
        assert retcode == 0
        np.savetxt(osp.join(outdir, f"qpos/{index:06d}.txt"), curr_radian)

        # POINTREND_DIR = "./detectron2/projects/PointRend"
        # pointrend_cfg_file = self.cfg.model.rbsolver_iter.pointrend_cfg_file
        # pointrend_model_weight = self.cfg.model.rbsolver_iter.pointrend_model_weight
        # config_file = osp.join(POINTREND_DIR, pointrend_cfg_file)
        # model_weight = osp.join(POINTREND_DIR, pointrend_model_weight)
        # image_path = osp.join(outdir, f"color/{index:06d}.png")
        #
        # from crc.utils.pointrend_api import pointrend_api
        # pred_binary_mask = pointrend_api(config_file, model_weight, image_path)
        # outpath = osp.join(outdir, "pred_mask", osp.basename(image_path))
        # os.makedirs(osp.dirname(outpath), exist_ok=True)
        # imageio.imsave(outpath, (pred_binary_mask * 255)[:, :, None].repeat(3, axis=-1))
        # tmp = plt_utils.vis_mask(rgb, pred_binary_mask.astype(np.uint8), [255, 0, 0])
        # vis3d.add_image(tmp, name='hover_pred_mask')
        # print()

    def explore_next_state(self, explore_it, to_zero=False):
        space_explorer = SpaceExplorerEih(self.cfg)
        dps = next(iter(self.train_dl))
        if len(self.cfg.model.rbsolver_iter.data_pool) > 0:
            dps['has_selected'] = self.sampled_indices
        dps['to_zero'] = to_zero
        outputs, _ = space_explorer(dps)
        # self.tb_writer.add_scalar("explore/var_max", outputs['var_max'].item(), explore_it)
        # self.tb_writer.add_scalar("explore/var_min", outputs['var_min'].item(), explore_it)
        # self.tb_writer.add_scalar("explore/var_mean", outputs['var_mean'].item(), explore_it)
        self.tb_writer.add_scalar("explore/variance", outputs['variance'].item(), explore_it)
        if len(self.cfg.model.rbsolver_iter.data_pool) > 0:
            self.sampled_indices.append(outputs['qpos_idx'].item())
        else:
            next_qpos = outputs['qpos']
            self.qposes = np.concatenate([self.qposes, next_qpos[:7][None]], axis=0)
            plan_result = outputs['plan_result']
            self.plan_result = plan_result
            print()

    def rebuild(self, index):
        self.initialize_Tc_c2e()
        self.model: nn.Module = build_model(self.cfg).to(torch.device(self.cfg.model.device))
        self.segment(index)
        self.train_dl = make_data_loader(self.cfg, is_train=True)
        self.valid_dl = make_data_loader(self.cfg, is_train=False)
        self.optimizer = make_optimizer(self.cfg, self.model)
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer,
                                           self.cfg.solver.num_epochs * len(self.train_dl))

    def clean_data_dir(self):
        data_dir = self.cfg.model.rbsolver_iter.data_dir
        # os.system(f"rm -rf {data_dir}")
        archive_runs(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    @torch.no_grad()
    def get_preds(self):
        return torch.empty([])

    def reset_to_zero_qpos(self):
        if self.cfg.model.rbsolver_iter.use_realarm.enable is True:
            arm = self.arm
            speed = self.cfg.model.rbsolver_iter.use_realarm.speed
            plan_qposes = self.plan_result['position']
            if not self.cfg.model.rbsolver_iter.use_realarm.speed_control:
                for plan_qpose in tqdm.tqdm(plan_qposes):
                    arm.set_servo_angle(angle=plan_qpose, is_radian=True, speed=speed, wait=True)
            else:
                safety_factor = self.cfg.model.rbsolver_iter.use_realarm.safety_factor
                timestep = self.cfg.model.rbsolver_iter.use_realarm.timestep
                # curr_qpos = arm.get_servo_angle(is_radian=True)[1]
                arm.set_mode(4)
                arm.set_state(state=0)
                time.sleep(1)
                for target_qpos in tqdm.tqdm(plan_qposes):
                    code, joint_state = arm.get_joint_states(is_radian=True)
                    joint_pos = joint_state[0][:7]
                    diff = target_qpos[:7] - joint_pos
                    qvel = diff / timestep / safety_factor
                    qvel_cliped = np.clip(qvel, -0.3, 0.3)
                    arm.vc_set_joint_velocity(qvel_cliped, is_radian=True, is_sync=True, duration=timestep)
                    time.sleep(timestep)
                arm.set_mode(0)
                time.sleep(1)
                print()

    def initialize_Tc_c2e(self):
        if self.cfg.model.rbsolver.init_Tc_c2e != []:
            return
        init_method = self.cfg.model.rbsolver_iter.init_method
        if init_method == 'manual':
            assert self.cfg.model.rbsolver.init_Tc_c2e != [], "init_Tc_c2e is empty"
        elif init_method == 'pvnet':
            raise NotImplementedError()
        elif init_method == 'meshloc':
            db_dir = self.cfg.model.rbsolver_iter.meshloc_database_dir
            # todo netvlad, currently using only one image.
            os.makedirs(db_dir, exist_ok=True)
            if len(os.listdir(db_dir)) == 0:
                tgt_Tc_c2e = self.generate_meshloc_database()
            else:
                ref_img_path = osp.join(db_dir, "rgb.png")
                ref_Tc_c2e_path = osp.join(db_dir, "Tc_c2e.txt")
                ref_img = imageio.imread_v2(ref_img_path)
                H, W = ref_img.shape[:2]
                qpos = np.loadtxt(osp.join(db_dir, "qpos.txt"))
                ref_Tc_c2e = np.loadtxt(ref_Tc_c2e_path)
                outdir = self.cfg.model.rbsolver_iter.data_dir
                K = np.loadtxt(osp.join(outdir, "K.txt"))
                urdf_path = "data/xarm7_with_gripper_reduced_dof.urdf"
                sk = SAPIENKinematicsModelStandalone(urdf_path)
                Tb_b2e = sk.compute_forward_kinematics(qpos.tolist() + [0, 0], 8).to_transformation_matrix()
                Tc_c2b = ref_Tc_c2e @ np.linalg.inv(Tb_b2e)
                ref_mask = render_api.nvdiffrast_render_xarm_api(urdf_path, Tc_c2b, qpos.tolist() + [0, 0], H, W, K,
                                                                 return_sum=True)
                tgt_img_path = osp.join(outdir, "color/000000.png")
                ref_img = imageio.imread_v2(ref_img_path)
                H, W, _ = ref_img.shape
                # tgt_img = imageio.imread_v2(tgt_img_path)
                # plt_utils.image_grid([ref_img, tgt_img], show=True)
                # mkpts0, mkpts1, mconf = get_feature_matching(ref_img_path, tgt_img_path, dbg=True)
                kpts0, kpts1, conf = dkm_api.get_feature_matching(ref_img_path, tgt_img_path, mask0=ref_mask, dbg=False)
                dkm_api.DKMHelper.reset()
                wis3d = Wis3D(
                    out_folder="dbg",
                    sequence_name="initialize_Tc_c2e",
                    xyz_pattern=("x", "-y", "-z"),
                    enable=True
                )
                wis3d.add_keypoint_correspondences(Image.open(ref_img_path), Image.open(tgt_img_path), kpts0, kpts1,
                                                   metrics={'mconf': conf})
                ref_depth_path = osp.join(db_dir, "depth.png")
                if not osp.exists(ref_depth_path):
                    depth = render_api.pyrender_render_xarm_api(urdf_path, Tc_c2b, qpos.tolist() + [0, 0], H, W, K,
                                                                return_depth=True)
                    depth[depth > 10] = 0
                    cv2.imwrite(ref_depth_path, (depth * 1000).astype(np.uint16))
                else:
                    depth = cv2.imread(ref_depth_path, 2).astype(np.float32) / 1000.0
                fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                pc = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
                matched_pts0 = pc.reshape(H, W, 3)[kpts0.astype(int)[:, 1], kpts0.astype(int)[:, 0]]
                matched_pts0_ca = utils_3d.transform_points(matched_pts0, np.linalg.inv(Tc_c2b))
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(matched_pts0_ca, kpts1, K, None,
                                                              iterationsCount=1000, reprojectionError=8.0)
                R, _ = cv2.Rodrigues(rvecs)
                tgt_Tc_c2b = utils_3d.Rt_to_pose(R, tvecs.reshape(3))
                tgt_Tc_c2e = tgt_Tc_c2b @ Tb_b2e
            init_Tc_c2e = tgt_Tc_c2e
        else:
            raise NotImplementedError()
        self.cfg.defrost()
        self.cfg.model.rbsolver.init_Tc_c2e = init_Tc_c2e.tolist()
        self.cfg.freeze()

    def generate_meshloc_database(self):
        raise NotImplementedError("please run autoinit")
