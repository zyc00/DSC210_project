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
import trimesh
from PIL import Image
from crc.modeling.models.hand_eye_solver_eih import HandEyeSolverEIH
from dl_ext.timer import EvalTime
from pymp import Planner
import transforms3d.quaternions

from crc.modeling.models.rb_solve.collision_checker import CollisionChecker
from crc.modeling.models.rb_solve.space_explorer_dvqf import SpaceExplorerDVQF
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone

from crc.data import make_data_loader
from crc.solver.build import make_optimizer, make_lr_scheduler
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from loguru import logger
from sapien.utils import Viewer
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import tqdm
from scipy.spatial.transform import Rotation as R

from crc.modeling.build import build_model
from crc.trainer.base import BaseTrainer
from crc.trainer.utils import *
from crc.utils import plt_utils, utils_3d, render_api
from crc.utils.os_utils import number_of_monitors, archive_runs
from crc.utils.utils_3d import roty_np, rotx_np
from crc.utils.vis3d_ext import Vis3D


class SolverIterTrainerNBV(BaseTrainer):
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
        self.qposes = np.array(self.cfg.model.rbsolver_iter.start_qpos)
        self.history_p = []
        self.history_q = []

        self.init = None

        self.sampled_indices = [self.cfg.model.rbsolver_iter.start_index]  # for data pool
        self.Tc_c2e = None
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
            # self.optimizer.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = batchid
            output, loss_dict = self.model(batch)
            self.Tc_c2e = output['Te_e2c']

        return {}

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
            self.rebuild()
            evaltime("rebuild")
            if explore_it == 0:
                init = self.init_model(next(iter(self.train_dl)))
                print()
            metric_ams = self.do_fit(explore_it)
            evaltime("do fit")
            for k, am in metric_ams.items():
                self.tb_writer.add_scalar("val/" + k, am.avg, explore_it)
            to_zero = explore_it == self.cfg.solver.explore_iters - 1
            self.explore_next_state(explore_it, to_zero)
            evaltime("explore next state")
        self.reset_to_zero_qpos()

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

        cfg = self.cfg.sim_hec_eye_in_hand
        dbg = self.cfg.dbg is True

        if cfg.rt.enable:
            sapien.render_config.camera_shader_dir = "rt"
            sapien.render_config.viewer_shader_dir = "rt"
            sapien.render_config.rt_samples_per_pixel = 16
            sapien.render_config.rt_use_denoiser = True

        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = cfg.urdf_path  # 'data/xarm7.urdf'
        builder = loader.load_file_as_articulation_builder(urdf_path)
        robot = builder.build(fix_root_link=True)

        sk = SAPIENKinematicsModelStandalone(urdf_path)

        assert robot, 'URDF not loaded.'
        # scene.set_environment_map_from_files(
        #     "data/cubemap/SaintPetersSquare2/posx.jpg",
        #     "data/cubemap/SaintPetersSquare2/negx.jpg",
        #     "data/cubemap/SaintPetersSquare2/posy.jpg",
        #     "data/cubemap/SaintPetersSquare2/negy.jpg",
        #     "data/cubemap/SaintPetersSquare2/posz.jpg",
        #     "data/cubemap/SaintPetersSquare2/negz.jpg", )

        if len(self.history_p) == 0:
            assert self.cfg.model.rbsolver_iter.start_qpos is not None
            for qpos in self.cfg.model.rbsolver_iter.start_qpos:
                eef_pq = sk.compute_forward_kinematics(qpos, 8)
                self.history_p.append(eef_pq.p)
                self.history_q.append(eef_pq.q)
            self.history_p = np.array(self.history_p)
            self.history_q = np.array(self.history_q)
            os.makedirs(self.cfg.model.space_explorer.ckpt_path, exist_ok=True)
            torch.save({
                "model": {
                    "history_p": self.history_p,
                    "history_q": self.history_q,
                }
            }, osp.join(self.cfg.model.space_explorer.ckpt_path, "model.pth"))

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
                actor_builder.add_box_visual(half_size=half_size, color=color)
                box = actor_builder.build()
                box.set_pose(sapien.Pose(p=pose))

        scene.set_ambient_light([1] * 3)
        scene.add_directional_light([0, 1, -1], [1.0, 1.0, 1.0], position=[0, 0, 2], shadow=True)
        pts = trimesh.primitives.Sphere(radius=2.0).sample(cfg.n_point_light)
        for pt in pts:
            if pt[2] > 0.5:
                scene.add_point_light(pt, [1, 1, 1], shadow=True)

        # ---------------------------------------------------------------------------- #
        # Camera
        # ---------------------------------------------------------------------------- #
        near, far = 0.1, 100
        width, height = 1280, 720

        if cfg.add_chessboard:
            marker_builder = scene.create_actor_builder()
            marker_builder.add_visual_from_file(cfg.chessboard_path)
            checkerboard = marker_builder.build(name="marker")

        # np.savetxt(osp.join(outdir, "K.txt"), K)
        active_joints = robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_properties(stiffness=1e6, damping=1e4)
        # if cfg.camera_ring.enable is False:
        #     print()
        # todo refactor
        Tc_c2es = np.array(cfg.Tc_c2e)[None]

        # todo refactor
        # else:
        #     raise NotImplementedError()
        # todo: multiple cam pose generation?
        # print('Intrinsic matrix\n', camera.get_intrinsic_matrix())
        # np.savetxt(osp.join(outdir, "campose.txt"), Tc_c2e)
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="gen_data_hec_eye_in_hand",
                      auto_increase=True, enable=dbg)
        coord_convert = utils_3d.Rt_to_pose(roty_np(-np.pi / 2) @ rotx_np(np.pi / 2))
        # if cfg.random_qpos:
        #     loguru.logger.info("using random sampled qpos!")
        #     random_qpos_number = cfg.random_qpos_number
        #     pymp_planner = Planner(cfg.urdf_path,
        #                         user_joint_names=None,
        #                         ee_link_name=cfg.move_group,
        #                         srdf=cfg.srdf_path,
        #                         )
        #     qposes = np.random.uniform(*pymp_planner.robot.joint_limits, size=(random_qpos_number, robot.dof))
        #     qposes = list(filter(lambda x: not pymp_planner.robot.computeCollisions(x), qposes))
        # else:
        #     qposes = np.array(cfg.qpos)
        qposes = self.qposes
        nmonitors = number_of_monitors()
        if dbg and nmonitors > 0:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(x=-2, y=0, z=1)
            viewer.set_camera_rpy(r=0, p=-0.3, y=0)
        robot.set_pose(sapien.Pose())
        Tb_b2m = sapien.Pose([0.3, -0.1, 0], [1, 0, 0, 0]).to_transformation_matrix()
        if cfg.add_chessboard:
            checkerboard.set_pose(sapien.Pose.from_transformation_matrix(Tb_b2m))
        for i in range(len(Tc_c2es)):
            outdir = osp.join(cfg.outdir, f"{i:06d}")
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(osp.join(outdir, "color"), exist_ok=True)
            os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
            os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
            os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)
            os.makedirs(osp.join(outdir, "Tc_c2m"), exist_ok=True)

            index = 0
            Tc_c2e = Tc_c2es[i]
            np.savetxt(osp.join(outdir, "Tc_c2e.txt"), Tc_c2e)
            Te_e2c = np.linalg.inv(Tc_c2e) @ coord_convert
            rot = transforms3d.quaternions.mat2quat(Te_e2c[:3, :3])
            trans = Te_e2c[:3, 3]
            camera = scene.add_mounted_camera(
                name="camera",
                actor=robot.get_links()[8],  # todo why 8?
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

            for qpose in tqdm.tqdm(qposes, leave=False):
                robot.set_qpos(np.array(qpose, dtype=np.float32))
                eef_pq = sk.compute_forward_kinematics(qpose.tolist(), 8)
                Tb_b2e = eef_pq.to_transformation_matrix()
                Tc_c2b = Tc_c2e @ np.linalg.inv(Tb_b2e)
                Tc_c2m = Tc_c2b @ Tb_b2m
                np.savetxt(osp.join(outdir, f"Tc_c2m/{index:06d}.txt"), Tc_c2m)

                scene.update_render()
                vis3d.add_xarm((qpose.tolist() + [0, 0])[:9])
                if dbg and nmonitors > 0:
                    viewer.paused = True
                    while not viewer.closed:
                        qf = robot.compute_passive_force(
                            gravity=True,
                            coriolis_and_centrifugal=True,
                        )
                        robot.set_qf(qf)
                        scene.update_render()
                        # scene.step()
                        viewer.render()

                camera.take_picture()

                rgba = camera.get_color_rgba()  # [H, W, 4]
                rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                img = rgba_img[:, :, :3]
                vis3d.add_image(img, name='img')

                # if not cfg.rt.enable:
                #     seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
                #     rendered_mask = np.logical_and(seg_labels[..., 1] > 1, seg_labels[..., 1] < 10)
                # else:
                #     rendered_mask = render_api.nvdiffrast_render_xarm_api(urdf_path, np.linalg.inv(camera_pose),
                #                                                         qpose, height, width, K)

                # vis3d.add_image(rendered_mask, name='mask')
                # imageio.imsave(osp.join(outdir, f"gt_mask/{index:06d}.png"),
                #             np.repeat(rendered_mask[:, :, None], 3, axis=-1).astype(np.uint8) * 255)
                # tmp = plt_utils.vis_mask(img, rendered_mask.astype(np.uint8), color=[255, 0, 0])
                # vis3d.add_image(tmp, name="hover")
                # distort image
                # distort_coeffs = np.array([0.001, 0.001, 0.001, 0.001])
                # distort_img = cv2.undistort(img, K, np.array(distort_coeffs))
                # imageio.imsave(osp.join(outdir, f"color/{index:06d}.png"), distort_img)
                imageio.imsave(osp.join(outdir, f"color/{index:06d}.png"), img)
                np.savetxt(osp.join(outdir, f"qpos/{index:06d}.txt"), qpose)

                position = camera.get_float_texture('Position')  # [H, W, 4]

                points_opengl = position[..., :3].reshape(-1, 3)
                points_color = rgba[..., :3].reshape(-1, 3)
                model_matrix = camera.get_model_matrix()
                points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
                vis3d.add_point_cloud(points_world, points_color, name='points_world')
                pts_cam = utils_3d.transform_points(points_world, Tc_c2b)
                pts_cam[np.linalg.norm(points_world, axis=-1) < 1e-4] = 0  # filter out bg points
                vis3d.add_point_cloud(pts_cam, name='pts_cam')
                fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                depth = pts_cam[:, 2].reshape(height, width)
                pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
                vis3d.add_point_cloud(pts_rect, points_color, name='bp_from_depth')
                depth_image = (depth * 1000.0).astype(np.uint16)
                cv2.imwrite(osp.join(outdir, f"depth/{index:06d}.png"), depth_image)
                vis3d.increase_scene_id()
                index += 1
            scene.remove_camera(camera)

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

        np.savetxt(osp.join(outdir, "Tc_c2b.txt"), np.eye(4))  # dummy

        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg",
                      sequence_name="rbsolver_iter_realsense_capture_data", auto_increase=True, enable=True)
        index = len(self.qposes) - 1
        qpose = self.qposes[-1]
        vis3d.add_xarm(qpose.tolist() + [0, 0])
        arm = self.arm
        speed = self.cfg.model.rbsolver_iter.use_realarm.speed
        if len(self.qposes) == 1:
            arm.set_servo_angle(angle=qpose, is_radian=True, speed=speed, wait=True)
        else:
            # self.cc.planner.robot.set_qpos(self.qposes[-2])
            # retcode, result = self.cc.planner.plan_screw(self.qposes[-1], self.qposes[-2], )
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

        POINTREND_DIR = "./detectron2/projects/PointRend"
        pointrend_cfg_file = self.cfg.model.rbsolver_iter.pointrend_cfg_file
        pointrend_model_weight = self.cfg.model.rbsolver_iter.pointrend_model_weight
        config_file = osp.join(POINTREND_DIR, pointrend_cfg_file)
        model_weight = osp.join(POINTREND_DIR, pointrend_model_weight)
        image_path = osp.join(outdir, f"color/{index:06d}.png")

        from crc.utils.pointrend_api import pointrend_api
        pred_binary_mask = pointrend_api(config_file, model_weight, image_path)
        outpath = osp.join(outdir, "pred_mask", osp.basename(image_path))
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        imageio.imsave(outpath, (pred_binary_mask * 255)[:, :, None].repeat(3, axis=-1))
        tmp = plt_utils.vis_mask(rgb, pred_binary_mask.astype(np.uint8), [255, 0, 0])
        vis3d.add_image(tmp, name='hover_pred_mask')
        print()

    def explore_next_state(self, explore_it, to_zero=False):
        # TODO
        space_explorer = SpaceExplorerDVQF(self.cfg)
        dps = next(iter(self.train_dl))
        if len(self.cfg.model.rbsolver_iter.data_pool) > 0:
            dps['has_selected'] = self.sampled_indices
        dps['to_zero'] = to_zero
        outputs, _ = space_explorer(dps)
        # self.tb_writer.add_scalar("explore/var_max", outputs['var_max'].item(), explore_it)
        # self.tb_writer.add_scalar("explore/var_min", outputs['var_min'].item(), explore_it)
        # self.tb_writer.add_scalar("explore/var_mean", outputs['var_mean'].item(), explore_it)
        # self.tb_writer.add_scalar("explore/variance", outputs['variance'].item(), explore_it)
        if len(self.cfg.model.rbsolver_iter.data_pool) > 0:
            self.sampled_indices.append(outputs['qpos_idx'].item())
        else:
            next_qpos = outputs['qpos']
            self.qposes = np.concatenate([self.qposes, next_qpos[:7][None]], axis=0)
            # plan_result = outputs['plan_result']
            # self.plan_result = plan_result
            print()

    def rebuild(self):
        self.model: nn.Module = build_model(self.cfg).to(torch.device(self.cfg.model.device))
        # pdb.set_trace()
        self.train_dl = make_data_loader(self.cfg, is_train=True)
        self.valid_dl = make_data_loader(self.cfg, is_train=False)
        self.optimizer = make_optimizer(self.cfg, self.model)
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer,
                                           self.cfg.solver.num_epochs * len(self.train_dl))
        self.init_model = HandEyeSolverEIH(self.cfg)

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
