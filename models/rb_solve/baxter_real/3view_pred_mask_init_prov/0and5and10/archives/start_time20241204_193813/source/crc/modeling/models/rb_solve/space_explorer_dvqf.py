import cv2
import os.path as osp
import glob
import os
import math
import random

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
from scipy.spatial.transform import Rotation as R


class SpaceExplorerDVQF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.space_explorer

        self.mesh_dir = self.cfg.mesh_dir

        self.dbg = self.total_cfg.dbg

        ckpt_dir = self.cfg.ckpt_path
        ckpt_paths = sorted(glob.glob(osp.join(ckpt_dir, "model*.pth")))
        loguru.logger.info(f"Auto detect ckpt_path")
        if len(ckpt_paths) == 0:
            self.history_p = None
            self.history_q = None
        else:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path, "cpu")
            self.history_p = ckpt['model']['history_p']
            self.history_q = ckpt['model']['history_q']
        # self.history_losses = ckpt['model']['history_losses']
        self.dummy = nn.Parameter(torch.zeros(1))
        self.sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        if self.cfg.self_collision_check.enable:
            self.pymp_planner = Planner(self.cfg.urdf_path,
                                        user_joint_names=None,
                                        # ee_link_name=self.cfg.move_group,
                                        ee_link_name=None,
                                        srdf=self.cfg.srdf_path,
                                        )
        if self.cfg.collision_check.enable:
            self.planner = CollisionChecker(self.cfg)
            self.pc_added = False
        if self.cfg.max_dist_constraint.enable:
            self.max_dist_center = self.compute_max_dist_center()

    def forward(self, dps):
        vis3d = Vis3D(xyz_pattern=("x", "y", "z"), out_folder="dbg", sequence_name="space_explorer", auto_increase=True,
                      enable=self.dbg)
        to_zero = dps.get("to_zero", False)

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

        if to_zero:
            qposes = np.zeros((1, robot.dof))
            loguru.logger.info("using zero qpos choices!")
        else:
            if self.cfg.qpos_choices != "":
                loguru.logger.info("using provided qpos choices!")
                qposes = np.loadtxt(self.cfg.qpos_choices)
                has_selected = dps['has_selected']
            else:
                loguru.logger.info("using sampled qpos choices!")
                qposes, eef_ps, eef_qs = self.sample_qposes(self.cfg.sample_dof, robot.dof)

        tid = -1
        if to_zero:
            tid = 0
            next_qpos = qposes[tid]
            outputs = {
                "qpos": next_qpos,
                "qpos_idx": tid,
            }
            return outputs, {}
        else:
            highest_dvqf_score = -10000
            # change batch
            for key in range(len(eef_ps)):
                eef_p = eef_ps[key]
                eef_q = eef_qs[key]
                dvqf_score = self.calc_dvqfscore(eef_p, eef_q, self.history_p, self.history_q)
                if dvqf_score > highest_dvqf_score:
                    highest_dvqf_score = dvqf_score
                    tid = key
        assert tid != -1
        next_qpos = qposes[tid]
        eef_p = eef_ps[tid]
        eef_q = eef_qs[tid]
        self.history_p = torch.cat([torch.tensor(self.history_p), torch.tensor(eef_p[None])], dim=0)
        self.history_q = torch.cat([torch.tensor(self.history_q), torch.tensor(eef_q[None])], dim=0)
        torch.save({
            "model": {
                "history_p": self.history_p,
                "history_q": self.history_q,
            }
        }, osp.join(self.cfg.ckpt_path, "model.pth"))

        outputs = {
            "qpos": next_qpos,
            "qpos_idx": tid,
            # "variance": variance,
            # "var_max": variances[variances > 0].max(),
            # "var_min": variances[variances > 0].min(),
            # "var_mean": variances[variances > 0].mean(),
            # "plan_result": plan_results[tid]
        }
        return outputs, {}

    def calc_dvqfscore(self, eef_p, eef_q, history_p, history_q):
        eqal = 0
        for i in range(history_p.shape[0]):
            if torch.sum(torch.tensor(history_p[i] - eef_p) ** 2) < 0.005 and torch.sum(torch.tensor(history_q[i] - eef_q) ** 2) < 0.005:
                eqal = -1
        cat_p = torch.cat([torch.tensor(history_p), torch.tensor(eef_p[None])], dim=0)
        cat_q = torch.cat([torch.tensor(history_q), torch.tensor(eef_q[None])], dim=0)
        n = cat_p.shape[0]
        mean_p = torch.sum(cat_p, dim=0) / n
        mean_q = torch.sum(cat_q, dim=0) / n
        uncertainty = torch.sum((cat_p - mean_p) ** 2) / n + torch.sum((cat_q - mean_q) ** 2) / n
        pose_diversity = 0
        epsi = 0.2
        mu = 0.22
        alpha = 0.09
        for i in range(n):
            if torch.sum((cat_p[i] - mean_p) ** 2) < epsi and torch.sum((cat_q[i] - mean_q) ** 2) < mu:
                pose_diversity = -1
                break
        return alpha * uncertainty + (1 - alpha) * pose_diversity + eqal

    def sample_qposes(self, dof, total_dof):
        # TODO select views from trapezoidal field

        # if not self.cfg.sample_limit:
        #     raise ValueError("sample_limit is False!")
        #     loguru.logger.error("sample_limit is False!")
        #     joint_limits = [np.full(dof, -np.pi), np.full(dof, np.pi)]
        # else:
        #     joint_limits = [self.pymp_planner.robot.joint_limits[0][:dof],
        #                     self.pymp_planner.robot.joint_limits[1][:dof]]
        # if 'PYCHARM_HOSTED' in os.environ and self.total_cfg.deterministic:
        #     np.random.seed(0)
        # if self.cfg.qpos_sample_method == "grid":
        #     raise NotImplementedError()
        #     n_sample_qpos_each_joint = self.cfg.n_sample_qpos_each_joint
        #     random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], n_sample_qpos_each_joint)  # 7,2
        #     random_qpos = (random_qpos * 2 - 1) * np.pi

        #     random_qpos = list(itertools.product(*random_qpos.tolist()))
        #     random_qpos = np.array(random_qpos)
        # elif self.cfg.qpos_sample_method == "random_eef":
        vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="sample_qposes_random_eef",
                      enable=self.dbg)
        # xmin, ymin, zmin, xmax, ymax, zmax = 0.26, -0.28, 0.07, 0.46, -0.06, 0.165
        # sampled_eef_position = np.random.uniform(np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]),
        #                                          size=[self.cfg.n_sample_qposes, 3])
        step = 0.1
        sampled_cam_position = []
        zs = np.arange(0.4, 0.8, 0.1)
        for z_val in zs:
            for x_val in np.arange(-z_val / 4, z_val / 4, 0.1):
                for y_val in np.arange(-z_val / 4, z_val / 4, 0.1):
                    # sampled_cam_position.append([x_val + 0.3, y_val, z_val])
                    sampled_cam_position.append([x_val + 0.4, y_val, z_val])
        angles_deg = [0]
        angles_rad = [math.radians(angle) for angle in angles_deg]
        quaternions = [R.from_euler('xyz', [0, 0, angle], degrees=True).as_quat() for angle in angles_deg]

        random_qposes = []
        random_p = []
        random_q = []
        for scp in sampled_cam_position:
            # for quaternion in quaternions:
            sapien_pose = sapien.Pose(scp, [0, 1, 0, 0])
            next_angle, success, err = self.sk.model.compute_inverse_kinematics(8, sapien_pose,
                                                                                active_qmask=[1, 1, 1, 1, 1, 1, 0])
            if success:
                random_qposes.append(next_angle[:7])
                random_p.append(scp)
                random_q.append([0, 1, 0, 0])
                vis3d.add_xarm(next_angle.tolist() + [0, 0])
                vis3d.increase_scene_id()
        random_qpos = np.array(random_qposes)
        random_p = np.array(random_p)
        random_q = np.array(random_q)
        # else:
        #     random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], [self.cfg.n_sample_qposes, dof])

        pad_qpos = np.zeros([random_qpos.shape[0], total_dof - dof])
        random_qpos = np.concatenate([random_qpos, pad_qpos], axis=1)
        return random_qpos, random_p, random_q

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
        # generate point cloud describing the edges of the workspace: x from -0.1 to 1.5, y from -0.4 t 0.4, z from -2 to 1.5
        xmin, ymin, zmin, xmax, ymax, zmax = -0.2, -0.5, -0.5, 1.0, 0.5, 1.0
        box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
        box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        pts_ws = box.sample(20000)

        pts_plane = np.zeros([20000, 3])
        pts_plane[:, 0] = np.random.uniform(-0.2, 0.15, size=20000)
        pts_plane[:, 1] = np.random.uniform(-0.4, 0.4, size=20000)
        pts_plane[:, 2] = 0
        norm = np.linalg.norm(pts_plane, axis=1)
        keep = norm > 0.1
        pts_plane = pts_plane[keep]
        pts_base = np.concatenate([pts_ws, pts_plane], axis=0)

        pts_base, _ = random_choice(pts_base, 5000, dim=0, replace=False)
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
