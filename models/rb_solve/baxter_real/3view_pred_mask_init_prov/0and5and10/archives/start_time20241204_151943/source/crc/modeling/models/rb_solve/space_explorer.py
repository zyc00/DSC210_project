import random

import cv2
import os.path as osp
import glob
import itertools
import os

import loguru
import numpy as np
import pytorch3d
import sapien.core as sapien
import torch
import torch.nn as nn
import tqdm
import trimesh
from dl_ext import AverageMeter

from crc.utils.timer import EvalTime
from pymp import Planner

from crc.modeling.models.rb_solve.collision_checker import CollisionChecker
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import utils_3d, render_api
from crc.utils.os_utils import number_of_monitors, red
from crc.utils.pn_utils import random_choice, to_array
from crc.utils.utils_3d import roty_np, rotx_np
from crc.utils.vis3d_ext import Vis3D
from pytorch3d.ops import sample_farthest_points


class SpaceExplorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.space_explorer

        self.mesh_dir = self.cfg.mesh_dir

        self.dbg = self.total_cfg.dbg

        ckpt_path = self.cfg.ckpt_path
        if ckpt_path == "":
            ckpt_dir = self.total_cfg.output_dir
            ckpt_paths = sorted(glob.glob(osp.join(ckpt_dir, "model*.pth")))
            if len(ckpt_paths) > 0:
                ckpt_path = ckpt_paths[-1]
                loguru.logger.info(f"Auto detect ckpt_path {ckpt_path}")

        if ckpt_path != "":
            ckpt = torch.load(ckpt_path, "cpu")
            self.history_dof6 = ckpt['model']['history_ops']
            self.history_losses = ckpt['model']['history_losses']
        self.dummy = nn.Parameter(torch.zeros(1))
        self.sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        if self.cfg.self_collision_check.enable or self.cfg.collision_check.enable:
            # self.pymp_planner = Planner(self.cfg.urdf_path,
            #                             user_joint_names=None,
            #                             ee_link_name=self.cfg.move_group,
            #                             srdf=self.cfg.srdf_path,
            #                             )
            # if self.cfg.collision_check.enable:
            self.planner = CollisionChecker(self.cfg)
            self.pc_added = False
        if self.cfg.max_dist_constraint.enable:
            if self.cfg.max_dist_constraint.max_dist_center == 0:
                self.max_dist_center = self.compute_max_dist_center()
            else:
                self.max_dist_center = np.array([0, 0, self.cfg.max_dist_constraint.max_dist_center])

    def forward(self, dps):
        vis3d = Vis3D(xyz_pattern=("x", "y", "z"), out_folder="dbg", sequence_name="space_explorer", auto_increase=True,
                      enable=self.dbg)
        # print("temporarily enable vis3d")
        # print("temporarily enable vis3d")
        # print("temporarily enable vis3d")
        # print("temporarily enable vis3d")
        # print("temporarily enable vis3d")
        # print("temporarily enable vis3d")
        to_zero = dps.get("to_zero", False)
        if self.cfg.resample_camera_poses.enable is False:
            history_dof6 = self.history_dof6
            keep = ~(history_dof6 == 0).all(dim=1)
            history_dof6 = history_dof6[keep]
            history_dof6 = history_dof6[self.cfg.start:]
        else:
            min_loss_index = self.history_losses.argmin().item()
            sample_center_dof = self.history_dof6[min_loss_index]
            history_dof6 = self.resample_camera_poses(sample_center_dof, dps)
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = self.cfg.urdf_path
        # load as a kinematic articulation
        builder = loader.load_file_as_articulation_builder(urdf_path)
        robot = builder.build(fix_root_link=True)

        active_joints = robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_properties(stiffness=1e6, damping=1e5)
        if to_zero:
            qposes = np.zeros((1, robot.dof))
            loguru.logger.info("using zero qpos choices!")
            has_selected = []
        else:
            if self.cfg.qpos_choices != "":
                loguru.logger.info("using provided qpos choices!")
                qposes = np.loadtxt(self.cfg.qpos_choices)
                has_selected = dps['has_selected']
            else:
                loguru.logger.info("using sampled qpos choices!")
                if self.cfg.qpos_sample_method_switch is False:
                    qpos_sample_method = self.cfg.qpos_sample_method
                    n_sample_qposes = self.cfg.n_sample_qposes
                else:
                    random_number = random.random()
                    print("random_number", random_number)
                    if random_number < 0.5:
                        qpos_sample_method = "random"
                        n_sample_qposes = self.cfg.n_sample_qposes
                    else:
                        qpos_sample_method = "random_eef"
                        n_sample_qposes = 100
                qposes = self.sample_qposes(self.cfg.sample_dof, robot.dof, n_sample_qposes,
                                            dps.get("center_qpos", None),
                                            qpos_sample_method=qpos_sample_method)
                has_selected = []
        width, height = 960, 480
        fx, fy, cx, cy = self.cfg.camera_intrinsic
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        if self.cfg.sample_camera_poses_method == "random":
            history_Tc_c2b = utils_3d.se3_exp_map(history_dof6).permute(0, 2, 1).numpy()
            history_Tc_c2b, _ = random_choice(history_Tc_c2b,
                                              size=self.cfg.sample,
                                              dim=0, replace=False)
        elif self.cfg.sample_camera_poses_method == "fps":
            ret = sample_farthest_points(history_dof6[None], K=self.cfg.sample)
            history_Tc_c2b = utils_3d.se3_exp_map(ret[0][0]).permute(0, 2, 1).numpy()
        history_cam_poses = np.linalg.inv(history_Tc_c2b)
        variances = []
        evaltime = EvalTime(disable=not self.total_cfg.evaltime)
        n_self_collision = 0
        n_has_selected = 0
        n_exceed_max_dist = 0
        n_collision = 0
        plan_results = {}

        time_ams = {
            "begin_loop_time": AverageMeter(),
            "robot_set_qpos_time": AverageMeter(),
            "vis3d_add_time": AverageMeter(),
            "self_collision_time": AverageMeter(),
            "max_dist_cons_time": AverageMeter(),
            "collision_check_time": AverageMeter(),
            "render_mask_time": AverageMeter(),
            "var_time": AverageMeter()}
        evaltime("")
        pts_base = self.get_workspace_boundary()
        self.planner.add_point_cloud(pts_base)
        for qpos_idx in tqdm.trange(qposes.shape[0]):
            vis3d.set_scene_id(qpos_idx)
            time_ams["begin_loop_time"].update(evaltime("begin loop"))
            if qpos_idx in has_selected:
                n_has_selected += 1
                variances.append(0)
                continue
            qpos = qposes[qpos_idx]
            qpos = [0] * self.cfg.qpos_choices_pad_left + qpos.tolist() + [0] * self.cfg.qpos_choices_pad_right
            robot.set_qpos(np.array(qpos, dtype=np.float32))
            time_ams["robot_set_qpos_time"].update(evaltime("robot set qpos"))
            if "xarm" in self.cfg.urdf_path:
                vis3d.add_xarm(qpos)
            elif "baxter" in self.cfg.urdf_path:
                vis3d.add_baxter(qpos)
            elif 'hillbot' in self.cfg.urdf_path:
                vis3d.add_realman(np.array(qpos))
            else:
                raise NotImplementedError()
            time_ams["vis3d_add_time"].update(evaltime("vis3d add"))
            if self.cfg.self_collision_check.enable:
                self_collision = self.planner.planner.check_for_self_collision(qpos)
                self_collision_time = evaltime("self collision compute")
                time_ams["self_collision_time"].update(self_collision_time)
                if self_collision:
                    variances.append(-2)
                    n_self_collision += 1
                    continue
            if self.cfg.max_dist_constraint.enable is True:
                lbase = self.sk.compute_forward_kinematics(qpos, 3).to_transformation_matrix()
                vis3d.add_spheres(lbase[:3, 3] + self.max_dist_center, radius=self.cfg.max_dist_constraint.max_dist)
                exceed_max_dist_constraint = False
                # for link in range(len(self.sk.robot.get_links())):
                for link in [10, 12, 14, 16]:
                    pq = self.sk.compute_forward_kinematics(qpos, link).to_transformation_matrix()

                    pq = np.linalg.inv(lbase) @ pq
                    if np.linalg.norm(pq[:3, 3] - self.max_dist_center) > self.cfg.max_dist_constraint.max_dist:
                        exceed_max_dist_constraint = True
                        break
                max_dist_cons_time = evaltime("max dist constraint compute")
                time_ams["max_dist_cons_time"].update(max_dist_cons_time)
                if exceed_max_dist_constraint:
                    variances.append(-1)
                    n_exceed_max_dist += 1
                    continue
            if self.cfg.collision_check.enable:
                assert not self.cfg.collision_check.use_pointcloud
                curr_qpos = dps['qpos'][-1].cpu().numpy()
                pad_qpos = np.zeros([self.planner.robot.dof - curr_qpos.shape[0]])
                curr_qpos = np.concatenate([curr_qpos, pad_qpos])
                self.planner.robot.set_qpos(curr_qpos)
                assert self.cfg.collision_check.by_eef_pose is False
                timestep = self.cfg.collision_check.timestep
                # code, result = self.planner.move_to_qpos(qpos, time_step=timestep, use_point_cloud=True,
                #                                          planning_time=self.cfg.collision_check.planning_time)
                # self.planner.planner.planning_world.set_qpos(curr_qpos)
                result = self.planner.planner.plan_qpos([np.array(qpos)], curr_qpos, time_step=timestep,
                                                        planning_time=self.cfg.collision_check.planning_time)
                collision_check_time = evaltime("collision check compute done")
                time_ams["collision_check_time"].update(collision_check_time)
                if result['status'] != 'Success':
                    n_collision += 1
                    vis3d.add_point_cloud(pts_base)
                    vis3d.add_xarm(curr_qpos, name='curr')
                    vis3d.add_xarm(qpos, name='target')
                    variances.append(0)
                    continue
                else:
                    plan_results[qpos_idx] = result
            masks = []
            for cam_pose in tqdm.tqdm(history_cam_poses, leave=False, disable="PYCHARM_HOSTED" in os.environ):
                if "xarm" in osp.basename(self.cfg.urdf_path):
                    render_resize_factor = self.cfg.render_resize_factor
                    K_render = to_array(K)
                    K_render[:2] /= render_resize_factor
                    height_render = int(height / render_resize_factor)
                    width_render = int(width / render_resize_factor)
                    if not self.cfg.parallel_rendering:
                        rendered_mask = render_api.nvdiffrast_render_xarm_api(self.cfg.urdf_path,
                                                                              np.linalg.inv(cam_pose),
                                                                              qpos[:7] + [0, 0],
                                                                              height_render, width_render,
                                                                              K_render,
                                                                              return_ndarray=False)
                    else:
                        rendered_mask = render_api.nvdiffrast_parallel_render_xarm_api(self.cfg.urdf_path,
                                                                                       np.linalg.inv(cam_pose),
                                                                                       qpos[:7] + [0, 0],
                                                                                       height_render, width_render,
                                                                                       K_render,
                                                                                       return_ndarray=False)
                elif "baxter" in osp.basename(self.cfg.urdf_path):
                    tmp_qpos = qpos.copy()
                    if self.cfg.qpos_choices_pad_left > 0:
                        tmp_qpos = tmp_qpos[self.cfg.qpos_choices_pad_left:]
                    if self.cfg.qpos_choices_pad_right > 0:
                        tmp_qpos = tmp_qpos[:-self.cfg.qpos_choices_pad_right]
                    rendered_mask = render_api.nvdiffrast_render_baxter_api(self.cfg.urdf_path, np.linalg.inv(cam_pose),
                                                                            tmp_qpos,
                                                                            height, width,
                                                                            to_array(K), return_ndarray=False)
                elif 'hillbot' in osp.basename(self.cfg.urdf_path):
                    rendered_mask = render_api.nvdiffrast_render_realman_api(self.cfg.urdf_path,
                                                                             np.linalg.inv(cam_pose) @ np.linalg.inv(
                                                                                 self.sk.compute_forward_kinematics(
                                                                                     qpos,
                                                                                     3).to_transformation_matrix()),
                                                                             qpos[:7] + [0, 0],
                                                                             height, width,
                                                                             K,
                                                                             return_ndarray=False)
                else:
                    raise NotImplementedError()
                render_mask_time = evaltime("render mask")
                time_ams["render_mask_time"].update(render_mask_time)
                vis3d.add_image(rendered_mask)
                masks.append(rendered_mask)
            masks = torch.stack(masks)
            var = torch.var(masks.reshape(masks.shape[0], -1).float(), dim=0).sum()
            var_time = evaltime("var")
            time_ams["var_time"].update(var_time)
            variances.append(var)
        variances = torch.tensor(variances)
        print(f"space exploring finished.")
        loguru.logger.info(f"total {qposes.shape[0]} qposes")
        loguru.logger.info(f"total {variances.shape[0]} variances")
        loguru.logger.info(f"total {n_has_selected} has selected")
        loguru.logger.info(f"total {n_self_collision} self_collision")
        loguru.logger.info(f"total {n_exceed_max_dist} exceed max dist constraint")
        loguru.logger.info(f"total {n_collision} collision")
        loguru.logger.info(f"valid qposes {(variances >= 0).sum().item()}.\n")
        for k, v in time_ams.items():
            loguru.logger.info(f"time avg {k}: {v.avg} sum: {v.sum}")
        top_ids = variances.argsort(descending=True)
        vis3d.set_scene_id(0)
        rotx = utils_3d.rotx_np(-np.pi / 2)
        rotx = utils_3d.Rt_to_pose(rotx)
        if self.cfg.variance_based_sampling and qpos_sample_method == 'random':
            tid = top_ids[0].item()
            if variances[tid] > 0 or to_zero is True:
                if "xarm" in self.cfg.urdf_path:
                    vis3d.add_xarm(qposes[tid], Tw_w2B=rotx, name=f'xarm')
                elif "baxter" in self.cfg.urdf_path:
                    vis3d.add_baxter(qposes[tid], Tw_w2B=rotx, name=f'baxter')
                elif 'hillbot' in self.cfg.urdf_path:
                    vis3d.add_realman(qposes[tid], name=f'realman')
                else:
                    raise NotImplementedError()
                vis3d.increase_scene_id()
                next_qpos = qposes[tid]
                variance = variances[tid]
            else:
                raise RuntimeError("no valid qpos found!")
        else:
            keep = variances > 0
            variances = variances[keep]
            qposes = qposes[keep]
            rand_var, rand_idx = random_choice(variances, 1, dim=0, replace=False)
            print(red("randomly sample a qpos"))
            # todo: bug???
            tid = rand_idx[0]
            next_qpos = qposes[tid]
            variance = variances[tid]
            tid = keep.nonzero()[tid].item()

        outputs = {
            "qpos": next_qpos,
            "qpos_idx": tid,
            "variance": variance,
            "var_max": variances[variances > 0].max(),
            "var_min": variances[variances > 0].min(),
            "var_mean": variances[variances > 0].mean(),
            "plan_result": plan_results[tid]
        }
        return outputs, {}

    def sample_qposes(self, dof, total_dof, n_sample_qposes, center_qpos=None, qpos_sample_method="random"):
        if not self.cfg.sample_limit:
            raise ValueError("sample_limit is False!")
            loguru.logger.error("sample_limit is False!")
            joint_limits = [np.full(dof, -np.pi), np.full(dof, np.pi)]
        else:
            joint_limits = [self.planner.planner.joint_limits[:, 0][:dof],
                            self.planner.planner.joint_limits[:, 1][:dof]]
        if qpos_sample_method == "grid":
            raise NotImplementedError()
            n_sample_qpos_each_joint = self.cfg.n_sample_qpos_each_joint
            random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], n_sample_qpos_each_joint)  # 7,2
            random_qpos = (random_qpos * 2 - 1) * np.pi

            random_qpos = list(itertools.product(*random_qpos.tolist()))
            random_qpos = np.array(random_qpos)
        elif qpos_sample_method == "random_eef":
            vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="sample_qposes_random_eef",
                          enable=self.dbg)
            # print("temporarily enable vis3d")
            # print("temporarily enable vis3d")
            # print("temporarily enable vis3d")
            # print("temporarily enable vis3d")
            # print("temporarily enable vis3d")
            xmin, ymin, zmin, xmax, ymax, zmax = 0.13, -0.36, 0.25, 0.36, -0.06, 0.44
            sampled_eef_position = np.random.uniform(np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]),
                                                     size=[n_sample_qposes, 3])
            random_qposes = []
            for sep in sampled_eef_position:
                sapien_pose = sapien.Pose(sep, np.array([0, 1, 0, 0]))
                next_angle, success, err = self.sk.model.compute_inverse_kinematics(8, sapien_pose,
                                                                                    active_qmask=[1, 1, 1, 1, 1, 1, 0,
                                                                                                  0, 0, 0, 0, 0, 0])
                if success:
                    random_qposes.append(next_angle[:6])
                    vis3d.add_xarm(next_angle)
                    vis3d.add_point_cloud(sampled_eef_position, name="sampled_eef_position")
                    vis3d.increase_scene_id()
            random_qpos = np.array(random_qposes)
        elif qpos_sample_method == "random":
            if center_qpos is None:
                random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], [n_sample_qposes, dof])
            else:
                random_qpos = np.random.uniform(center_qpos[:dof] - self.cfg.qpos_sample_range,
                                                center_qpos[:dof] + self.cfg.qpos_sample_range,
                                                [n_sample_qposes, dof])
                random_qpos = np.clip(random_qpos, joint_limits[0], joint_limits[1])
            random_qpos[:, 0] = self.total_cfg.model.rbsolver_iter.start_qpos[0]
            random_qpos[:, [1, 2]] = self.total_cfg.model.rbsolver_iter.start_qpos[1:3]
            random_qpos[:, 4::2] = self.total_cfg.model.rbsolver_iter.start_qpos[4::2]
        else:
            raise NotImplementedError()
        pad_qpos = np.zeros([random_qpos.shape[0], total_dof - dof])
        random_qpos = np.concatenate([random_qpos, pad_qpos], axis=1)

        return random_qpos

    def compute_max_dist_center(self):
        wis3d = Vis3D(out_folder='dbg', sequence_name='compute_max_dist_center')
        pts = []
        qposes = np.random.uniform(*self.pymp_planner.robot.joint_limits,
                                   size=(self.cfg.max_dist_constraint.max_dist_center_compute_n,
                                         self.sk.robot.dof))
        qposes[:, 0] = 0.6
        qposes[:, [1, 2]] = 0
        qposes[:, 4::2] = 0
        loguru.logger.info("computing max dist center")
        for qpos in tqdm.tqdm(qposes):
            ret = self.pymp_planner.robot.computeCollisions(qpos)
            if not ret:
                curr_pts = []
                for link in range(len(self.sk.robot.get_links())):
                    pq = self.sk.compute_forward_kinematics(qpos, link)
                    left_base = self.sk.compute_forward_kinematics(qpos, 3)
                    pq = np.linalg.inv(left_base.to_transformation_matrix()) @ pq.to_transformation_matrix()
                    curr_pts.append(pq[:3, 3])
                curr_pts = np.array(curr_pts)
                maxid = np.argmax(np.linalg.norm(curr_pts, axis=-1))
                wis3d.add_realman(qpos, )
                pts.append(curr_pts[maxid])
            else:
                pts.append([0, 0, 0])
        pts = np.array(pts)
        wis3d.add_point_cloud(pts)
        maxi0 = pts[:, 1].argmax()
        mini0 = pts[:, 1].argmin()
        est_centerz = ((pts[maxi0] + pts[mini0]) / 2)[1]
        center = np.array([0, 0, est_centerz])
        loguru.logger.info("using center: " + str(center))
        return center

    def get_workspace_boundary(self):
        # generate point cloud describing the edges of the workspace: x from -0.1 to 1.5, y from -0.4 t 0.4, z from -2 to 1.5
        vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="get_workspace_boundary",
                      enable=self.dbg)
        # xmin, ymin, zmin, xmax, ymax, zmax = -0.4, -0.5, 0.1, 1.0, 0.4, 1.0
        # box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
        # box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        # pts_ws = box.sample(20000)
        #
        # norm = np.linalg.norm(pts_ws, axis=1)
        # keep = norm > 0.13
        # pts_ws = pts_ws[keep]
        #
        # pts_plane = np.zeros([20000, 3])
        # pts_plane[:, 0] = np.random.uniform(-0.2, 0.15, size=20000)
        # pts_plane[:, 1] = np.random.uniform(-0.4, 0.4, size=20000)
        # pts_plane[:, 2] = 0
        # norm = np.linalg.norm(pts_plane, axis=1)
        # keep = norm > 0.13
        # pts_plane = pts_plane[keep]
        #
        # pts_base = np.concatenate([pts_ws, pts_plane], axis=0)
        #
        # pts_base, _ = random_choice(pts_base, 5000, dim=0, replace=False)
        # vis3d.add_point_cloud(pts_base)
        # vis3d.add_xarm(np.zeros(13))
        xmin, ymin, zmin, xmax, ymax, zmax = -1.3, -1.3, -0.1, 1.3, 1.3, 2
        box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
        box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        pts_ws = box.sample(20000)

        pts_plane = np.zeros([20000, 3])
        pts_plane[:, 0] = -0.4
        pts_plane[:, 1] = np.random.uniform(-0.7, 0.7, size=20000)
        pts_plane[:, 2] = np.random.uniform(0, 1.8, size=20000)

        pts1 = np.zeros([10000, 3])
        pts1[:, 0] = np.random.uniform(-0.4, 0.4, size=10000)
        pts1[:, 1] = 0.25
        pts1[:, 2] = np.random.uniform(0.4, 1.0, size=10000)

        pts2 = np.zeros([10000, 3])
        pts2[:, 0] = np.random.uniform(-0.4, 0.4, size=10000)
        pts2[:, 1] = -0.25
        pts2[:, 2] = np.random.uniform(0.4, 1.0, size=10000)
        # norm = np.linalg.norm(pts_plane, axis=1)
        # keep = norm > 0.1
        # pts_plane = pts_plane[keep]
        #
        # pts_plane2 = np.random.uniform(np.array([0.1, -0.3, 0.2], ), np.array([0.5, 0.3, 0.22]), size=(1000, 3))
        #
        # pts3 = trimesh.primitives.Cylinder(0.1, 0.6).sample(1000)
        # keep = np.logical_and(pts3[:, 2] > 0, pts3[:, 2] < 0.29)
        # pts3 = pts3[keep]
        #
        pts_ws = np.concatenate([pts_ws, pts_plane, pts1, pts2], axis=0)
        #
        pts_ws, _ = random_choice(pts_ws, 5000, dim=0, replace=False)
        vis3d.add_point_cloud(pts_ws)
        # vis3d.add_xarm(np.zeros(9))
        qpos = np.zeros(19)
        qpos[0] = 0.62
        vis3d.add_realman(qpos)

        return pts_ws

    def resample_camera_poses(self, center_pose_dof6, dps):
        vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="resample_camera_poses")
        maxn = self.cfg.resample_camera_poses.maxn
        # trans_noise = self.cfg.resample_camera_poses.trans_noise
        # rot_noise = self.cfg.resample_camera_poses.rot_noise
        sampled_dof6 = []

        history_dof6 = self.history_dof6
        keep = ~(history_dof6 == 0).all(dim=1)
        history_losses = self.history_losses[keep][self.cfg.start:]
        history_dof6 = history_dof6[keep][self.cfg.start:]
        last_dof6 = history_dof6[-1]
        vis3d.add_point_cloud((history_dof6[:, :3] - last_dof6[:3]) * 50, name='history_dof6')
        vis3d.add_spheres(center_pose_dof6[:3] - last_dof6[:3] * 50, radius=1, name='center_pose_dof6')
        gap = history_dof6.max(0).values - history_dof6.min(0).values
        trans_noise = gap[:3].mean().item()
        rot_noise = gap[3:].mean().item()
        max_history_threshold = history_losses.max().item()

        H, W = dps['rgb'].shape[1:3]
        K = to_array(dps['K'][0])
        batch_size = dps['qpos'].shape[0]
        for i in tqdm.trange(maxn, desc="resample camera poses"):
            new_dof6 = np.concatenate([center_pose_dof6[:3] + np.random.random(3) * trans_noise,
                                       center_pose_dof6[3:] + np.random.random(3) * rot_noise])
            new_dof6 = torch.from_numpy(new_dof6).float()
            new_pose = utils_3d.se3_exp_map(new_dof6[None]).permute(0, 2, 1)[0]
            new_pose = to_array(new_pose)
            # compute losses
            loss_batch = 0
            for batch_index in range(batch_size):
                qpos = dps['qpos'][batch_index].tolist() + [0, 0]
                ref_mask = dps['mask_pred'][batch_index]
                rendered_mask = render_api.nvdiffrast_parallel_render_xarm_api(self.cfg.urdf_path,
                                                                               new_pose, to_array(qpos), H, W, K,
                                                                               return_ndarray=False)
                loss_i = torch.sum((ref_mask.cuda() - rendered_mask.float()) ** 2)
                loss_batch += loss_i
            if loss_batch < max_history_threshold:
                sampled_dof6.append(new_dof6)
            if len(sampled_dof6) >= self.cfg.sample:
                break
        sampled_dof6 = torch.stack(sampled_dof6, dim=0)
        vis3d.add_point_cloud((sampled_dof6[:, :3] - last_dof6[:3]) * 50, name='sampled_dof6')
        return sampled_dof6

    def plan_traj(self, start_qpos, end_qpos):
        self.planner.robot.set_qpos(start_qpos)
        timestep = self.cfg.collision_check.timestep
        code, result = self.planner.move_to_qpos(end_qpos, time_step=timestep, use_point_cloud=True,
                                                 planning_time=self.cfg.collision_check.planning_time)
        return code, result
