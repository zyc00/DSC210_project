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


class SpaceExplorerEih(nn.Module):
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
            ckpt_path = ckpt_paths[-1]
            loguru.logger.info(f"Auto detect ckpt_path {ckpt_path}")
        ckpt = torch.load(ckpt_path, "cpu")
        self.history_dof6 = ckpt['model']['history_ops']
        self.history_losses = ckpt['model']['history_losses']
        self.dummy = nn.Parameter(torch.zeros(1))
        self.sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        if self.cfg.self_collision_check.enable:
            self.pymp_planner = Planner(self.cfg.urdf_path,
                                        user_joint_names=None,
                                        ee_link_name=self.cfg.move_group,
                                        srdf=self.cfg.srdf_path,
                                        )
        if self.cfg.collision_check.enable:
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

        width, height = 1280, 720
        fx, fy, cx, cy = self.cfg.camera_intrinsic
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        if self.cfg.sample_camera_poses_method == "random":
            history_Tc_c2es = utils_3d.se3_exp_map(history_dof6).permute(0, 2, 1).numpy()
            history_Tc_c2es, _ = random_choice(history_Tc_c2es, size=self.cfg.sample, dim=0, replace=False)
        elif self.cfg.sample_camera_poses_method == "fps":
            ret = sample_farthest_points(history_dof6[None], K=self.cfg.sample)
            history_Tc_c2es = utils_3d.se3_exp_map(ret[0][0]).permute(0, 2, 1).numpy()
        else:
            raise NotImplementedError()

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
                qposes = self.sample_qposes(self.cfg.sample_dof, robot.dof,
                                            dps['link_poses'][0, 7])  # todo hard code 0,7
                has_selected = []

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

        # qposes = np.array([[1.91139, -0.26715, 4.98315, 1.32200, -0.80595, 2.61909, 0.00000, 0.00000, 0.00000],
        #                    [1.91139, -0.26715, 4.98315, 1.32200, -0.80595, 2.63, 0.00000, 0.00000, 0.00000],
        #                    ])
        # print("overwritten qposes for debug.!!!!!!")
        # all_masks = {}
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
            else:
                raise NotImplementedError()
            time_ams["vis3d_add_time"].update(evaltime("vis3d add"))
            if self.cfg.self_collision_check.enable:
                self_collision = self.pymp_planner.robot.computeCollisions(qpos)
                self_collision_time = evaltime("self collision compute")
                time_ams["self_collision_time"].update(self_collision_time)
                if self_collision:
                    variances.append(0)
                    n_self_collision += 1
                    continue
            if self.cfg.max_dist_constraint.enable is True:
                vis3d.add_spheres(self.max_dist_center, radius=self.cfg.max_dist_constraint.max_dist)
                exceed_max_dist_constraint = False
                for link in range(len(self.sk.robot.get_links())):
                    pq = self.sk.compute_forward_kinematics(qpos, link)
                    if np.linalg.norm(pq.p - self.max_dist_center) > self.cfg.max_dist_constraint.max_dist:
                        exceed_max_dist_constraint = True
                        break
                max_dist_cons_time = evaltime("max dist constraint compute")
                time_ams["max_dist_cons_time"].update(max_dist_cons_time)
                if exceed_max_dist_constraint:
                    variances.append(0)
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
                code, result = self.planner.move_to_qpos(qpos, time_step=timestep, use_point_cloud=True,
                                                         planning_time=self.cfg.collision_check.planning_time)
                collision_check_time = evaltime("collision check compute done")
                time_ams["collision_check_time"].update(collision_check_time)
                if code != 0:
                    n_collision += 1
                    vis3d.add_point_cloud(pts_base)
                    variances.append(0)
                    continue
                else:
                    plan_results[qpos_idx] = result
            masks = []
            for history_Tc_c2e in tqdm.tqdm(history_Tc_c2es, leave=False, disable="PYCHARM_HOSTED" in os.environ):
                if "xarm" in osp.basename(self.cfg.urdf_path):
                    Tb_b2e = self.sk.compute_forward_kinematics(qpos, 8).to_transformation_matrix()
                    Tc_c2b = history_Tc_c2e @ np.linalg.inv(Tb_b2e)
                    render_resize_factor = self.cfg.render_resize_factor
                    K_render = to_array(K)
                    K_render[:2] /= render_resize_factor
                    height_render = int(height / render_resize_factor)
                    width_render = int(width / render_resize_factor)
                    if not self.cfg.parallel_rendering:
                        rendered_mask = render_api.nvdiffrast_render_xarm_api(self.cfg.urdf_path,
                                                                              Tc_c2b,
                                                                              qpos[:7] + [0, 0],
                                                                              height_render, width_render,
                                                                              K_render,
                                                                              return_ndarray=False)
                    else:
                        rendered_mask = render_api.nvdiffrast_parallel_render_xarm_api(self.cfg.urdf_path,
                                                                                       Tc_c2b,
                                                                                       qpos[:7] + [0, 0],
                                                                                       height_render, width_render,
                                                                                       K_render,
                                                                                       return_ndarray=False)
                    # if rendered_mask.sum() != 0:
                    #     print()
                    # if rendered_mask.sum() == 0:
                    #     n_invisible += 1
                    #     variances.append(0)
                    #     continue
                # elif "baxter" in osp.basename(self.cfg.urdf_path):
                #     tmp_qpos = qpos.copy()
                #     if self.cfg.qpos_choices_pad_left > 0:
                #         tmp_qpos = tmp_qpos[self.cfg.qpos_choices_pad_left:]
                #     if self.cfg.qpos_choices_pad_right > 0:
                #         tmp_qpos = tmp_qpos[:-self.cfg.qpos_choices_pad_right]
                #     rendered_mask = render_api.nvdiffrast_render_baxter_api(self.cfg.urdf_path, np.linalg.inv(cam_pose),
                #                                                             tmp_qpos,
                #                                                             height, width,
                #                                                             to_array(K), return_ndarray=False)
                else:
                    raise NotImplementedError()
                render_mask_time = evaltime("render mask")
                time_ams["render_mask_time"].update(render_mask_time)
                vis3d.add_image(rendered_mask)
                masks.append(rendered_mask)
            masks = torch.stack(masks)
            # all_masks[qpos_idx] = masks
            if not self.cfg.var_in_valid_mask:
                var = torch.var(masks.reshape(masks.shape[0], -1).float(), dim=0).sum()
            else:
                valid_flag = (masks.sum(1).sum(1) > 0) & (masks.sum(1).sum(1) < masks.shape[1] * masks.shape[2])
                if valid_flag.sum() == 0:
                    var = torch.zeros(1)
                else:
                    ms = masks[valid_flag]
                    var = torch.var(ms.reshape(ms.shape[0], -1).float(), dim=0).sum()
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
        loguru.logger.info(f"valid qposes {(variances != 0).sum().item()}.\n")
        for k, v in time_ams.items():
            loguru.logger.info(f"time avg {k}: {v.avg} sum: {v.sum}")
        top_ids = variances.argsort(descending=True)
        vis3d.set_scene_id(0)
        rotx = utils_3d.rotx_np(-np.pi / 2)
        rotx = utils_3d.Rt_to_pose(rotx)
        if self.cfg.variance_based_sampling:
            tid = top_ids[0]
            if variances[tid] > 0:
                if "xarm" in self.cfg.urdf_path:
                    vis3d.add_xarm(qposes[tid], Tw_w2B=rotx, name=f'xarm')
                elif "baxter" in self.cfg.urdf_path:
                    vis3d.add_baxter(qposes[tid], Tw_w2B=rotx, name=f'baxter')
                else:
                    raise NotImplementedError()
                vis3d.increase_scene_id()
                next_qpos = qposes[tid]
                variance = variances[tid]
            elif dps['to_zero'] is False:
                raise RuntimeError("no valid qpos found!")
            else:
                next_qpos = qposes[0]
                variance = np.zeros([1])
            plan_result = plan_results[tid.item()]
        else:
            keep = variances > 0
            variances = variances[keep]
            qposes = qposes[keep]
            rand_var, rand_idx = random_choice(variances, 1, dim=0, replace=False)
            print(red("randomly sample a qpos"))
            tid = rand_idx[0]
            next_qpos = qposes[tid]
            variance = variances[tid]
            # plan_result = None

        outputs = {
            "qpos": next_qpos,
            "qpos_idx": tid,
            "variance": variance,
            "plan_result": plan_result
        }
        return outputs, {}

    def sample_qposes(self, dof, total_dof, ref_eef_pose):
        if not self.cfg.sample_limit:
            raise ValueError("sample_limit is False!")
            loguru.logger.error("sample_limit is False!")
            joint_limits = [np.full(dof, -np.pi), np.full(dof, np.pi)]
        else:
            joint_limits = [self.pymp_planner.robot.joint_limits[0][:dof],
                            self.pymp_planner.robot.joint_limits[1][:dof]]
        # if 'PYCHARM_HOSTED' in os.environ and self.total_cfg.deterministic:
        #     np.random.seed(0)
        if self.cfg.qpos_sample_method == "grid":
            raise NotImplementedError()
            n_sample_qpos_each_joint = self.cfg.n_sample_qpos_each_joint
            random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], n_sample_qpos_each_joint)  # 7,2
            random_qpos = (random_qpos * 2 - 1) * np.pi

            random_qpos = list(itertools.product(*random_qpos.tolist()))
            random_qpos = np.array(random_qpos)
        elif self.cfg.qpos_sample_method == "random_eef":
            vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="sample_qposes_random_eef",
                          enable=self.dbg)
            xmin, ymin, zmin, xmax, ymax, zmax = 0.26, -0.28, 0.07, 0.46, -0.06, 0.165
            sampled_eef_position = np.random.uniform(np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]),
                                                     size=[self.cfg.n_sample_qposes, 3])
            random_qposes = []
            for sep in sampled_eef_position:
                sapien_pose = sapien.Pose(sep, np.array([0, 1, 0, 0]))
                next_angle, success, err = self.sk.model.compute_inverse_kinematics(8, sapien_pose,
                                                                                    active_qmask=[1, 1, 1, 1, 1, 1, 0])
                if success:
                    random_qposes.append(next_angle[:6])
                    vis3d.add_xarm(next_angle)
                    vis3d.increase_scene_id()
            random_qpos = np.array(random_qposes)
        elif self.cfg.qpos_sample_method == "fan_shaped_eef":
            vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="sample_qposes_fan_shaped_eef",
                          enable=self.dbg)
            eef_position = ref_eef_pose[:3, 3]
            vis3d.add_spheres(eef_position, radius=0.01)

            qpos, success, err = self.sk.model.compute_inverse_kinematics(8, sapien.Pose(ref_eef_pose),
                                                                          active_qmask=[1, 1, 1, 1, 1, 1] + [0] * (self.sk.robot.dof - 6))
            assert success
            vis3d.add_xarm(qpos, name='xarm')
            # x, y, z = eef_position.tolist()
            radius_min = 0.35
            radius_max = 0.5
            elevation_min = -6
            elevation_max = 40
            azimuth_min = -80
            azimuth_max = 80
            center_min = np.array([-0.02, -0.02, 0.15])
            center_max = np.array([0.02, 0.02, 0.42])

            start_azim = azimuth_min / 180 * np.pi
            end_azim = azimuth_max / 180 * np.pi
            centers = np.random.uniform(center_min, center_max, [self.cfg.n_sample_qposes, 3])
            centers = np.array(centers)
            elev = np.random.uniform(elevation_min, elevation_max, self.cfg.n_sample_qposes)
            dist = np.random.uniform(radius_min, radius_max, self.cfg.n_sample_qposes)
            thetas = np.random.uniform(start_azim, end_azim, self.cfg.n_sample_qposes)
            angle_zs = np.deg2rad(elev)
            radiuss = dist * np.cos(angle_zs)
            heights = dist * np.sin(angle_zs)
            RTs = []
            for theta, radius, height, center, angle_z in zip(thetas, radiuss, heights, centers, angle_zs):
                st = np.sin(theta)
                ct = np.cos(theta)
                center_ = np.array([radius * ct, radius * st, height])
                center_[0] += center[0]
                center_[1] += center[1]
                center_[2] += center[2]
                R = np.array([
                    [-st, ct, 0],
                    [0, 0, -1],
                    [-ct, -st, 0]
                ])
                # Rotx = cv2.Rodrigues(angle_z * np.array([1., 0., 0.]))[0]
                up_vector = np.array([1.0, 0, 0])
                noise = np.random.normal(0, 0.1, 3)
                up_vector += noise
                up_vector /= np.linalg.norm(up_vector)
                Rotx = cv2.Rodrigues(angle_z * up_vector)[0]
                R = Rotx @ R
                T = - R @ center_
                RT = utils_3d.Rt_to_pose(R, T)
                RTs.append(RT)
            # return np.array(RTs)
            RTs = np.linalg.inv(RTs)
            random_qposes = []
            for pose in tqdm.tqdm(RTs, desc="compute ik"):
                rz = utils_3d.rotz_np(np.pi / 2)[0]
                rz = utils_3d.Rt_to_pose(rz)
                pose = pose @ rz
                random_rz = utils_3d.Rt_to_pose(utils_3d.rotz_np(np.random.uniform(-np.pi, np.pi))[0])
                pose = pose @ random_rz
                sapien_pose = sapien.Pose(pose)
                next_angle, success, err = self.sk.model.compute_inverse_kinematics(8, sapien_pose,
                                                                                    active_qmask=[1, 1, 1, 1, 1, 1] + [0] * (self.sk.robot.dof - 6))
                # active_qmask=[0, 1, 1, 1, 1, 1, 1,
                #               0, 0])
                vis3d.add_xarm(next_angle)
                vis3d.increase_scene_id()
                if success:
                    random_qposes.append(next_angle[:7])

            random_qpos = np.array(random_qposes)
        elif self.cfg.qpos_sample_method == 'random':
            random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], [self.cfg.n_sample_qposes, dof])
        else:
            raise NotImplementedError()
        pad_qpos = np.zeros([random_qpos.shape[0], total_dof - dof])
        random_qpos = np.concatenate([random_qpos, pad_qpos], axis=1)

        return random_qpos

    def compute_max_dist_center(self):
        pts = []
        qposes = np.random.uniform(*self.pymp_planner.robot.joint_limits,
                                   size=(self.cfg.max_dist_constraint.max_dist_center_compute_n,
                                         self.sk.robot.dof))
        loguru.logger.info("computing max dist center")
        for qpos in tqdm.tqdm(qposes):
            ret = self.pymp_planner.robot.computeCollisions(qpos)
            if not ret:
                curr_pts = []
                for link in range(len(self.sk.robot.get_links())):
                    pq = self.sk.compute_forward_kinematics(qpos, link)
                    curr_pts.append(pq.p)
                curr_pts = np.array(curr_pts)
                maxid = np.argmax(np.linalg.norm(curr_pts, axis=-1))
                pts.append(curr_pts[maxid])
            else:
                pts.append([0, 0, 0])
        pts = np.array(pts)
        maxi0 = pts[:, 1].argmax()
        mini0 = pts[:, 1].argmin()
        est_centerz = ((pts[maxi0] + pts[mini0]) / 2)[2]
        center = np.array([0, 0, est_centerz])
        loguru.logger.info("using center: " + str(center))
        return center

    def get_workspace_boundary(self):
        vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="get_workspace_boundary",
                      enable=self.dbg)
        xmin, ymin, zmin, xmax, ymax, zmax = -0.4, -0.5, 0.1, 1.0, 0.4, 1.0
        box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
        box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        pts_ws = box.sample(20000)

        norm = np.linalg.norm(pts_ws, axis=1)
        keep = norm > 0.13
        pts_ws = pts_ws[keep]

        pts_plane = np.zeros([20000, 3])
        pts_plane[:, 0] = np.random.uniform(-0.2, 0.15, size=20000)
        pts_plane[:, 1] = np.random.uniform(-0.4, 0.4, size=20000)
        pts_plane[:, 2] = 0
        norm = np.linalg.norm(pts_plane, axis=1)
        keep = norm > 0.13
        pts_plane = pts_plane[keep]

        # pts_plane2 = np.random.uniform(np.array([0.1, -0.3, 0.2], ), np.array([0.5, 0.3, 0.22]), size=(1000, 3))
        #
        # pts3 = trimesh.primitives.Cylinder(0.1, 0.4).sample(1000)
        # keep = np.logical_and(pts3[:, 2] > 0, pts3[:, 2] < 0.19)
        # pts3 = pts3[keep]

        pts_base = np.concatenate([pts_ws, pts_plane], axis=0)

        pts_base, _ = random_choice(pts_base, 5000, dim=0, replace=False)
        vis3d.add_point_cloud(pts_base)
        vis3d.add_xarm(np.zeros(13))
        return pts_base

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
