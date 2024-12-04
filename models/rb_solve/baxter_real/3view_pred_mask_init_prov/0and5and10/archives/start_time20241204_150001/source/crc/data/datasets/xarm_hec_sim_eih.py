import matplotlib.pyplot as plt
import glob
import cv2
import imageio
import numpy as np
import torch
import transforms3d

from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import utils_3d, plt_utils
from crc.utils.vis3d_ext import Vis3D


class XarmHecSimEihDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.xarm_hec_sim
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.png"))[:ds_len]
        gt_mask_paths = sorted(glob.glob(f"{data_dir}/gt_mask/*.png"))[:ds_len]
        pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        self.masks = []
        self.masks_pred = []
        # self.masks_anno = []
        self.qpos = []
        # self.depths = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        # for depth_path in depth_paths:
        #     depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        #     self.depths.append(depth)
        for gt_mask_path in gt_mask_paths:
            gt_mask = np.array(imageio.imread_v2(gt_mask_path))[..., 0] > 0
            self.masks.append(gt_mask)
        self.masks = np.stack(self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        for pred_mask_path in pred_mask_paths:
            pred_mask = imageio.imread_v2(pred_mask_path)[..., 0] > 0
            self.masks_pred.append(pred_mask)
        if len(self.masks_pred) > 0:
            self.masks_pred = np.stack(self.masks_pred)
            self.masks_pred = torch.from_numpy(self.masks_pred).float()
        # for anno_mask_path in anno_mask_paths:
        #     anno_mask = imageio.imread_v2(anno_mask_path)[..., 0] > 0
        #     self.masks_anno.append(anno_mask)
        # if len(self.masks_anno) > 0:
        #     self.masks_anno = np.stack(self.masks_anno)
        #     self.masks_anno = torch.from_numpy(self.masks_anno).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            self.qpos.append(qpos)
            link_poses = []
            for link in self.cfg.use_links:
                pad = np.zeros(sk.robot.dof - qpos.shape[0])
                pq = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), link)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.Rt_to_pose(R, t)
                link_poses.append(pose_eb)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/K.txt")
        self.K = torch.from_numpy(self.K).float()
        self.Tc_c2e = np.loadtxt(f"{data_dir}/Tc_c2e.txt")
        self.Tc_c2e = torch.from_numpy(self.Tc_c2e).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]
        # depth = self.depths[idx]
        mask = self.masks[idx]

        qpos = self.qpos[idx]
        K = self.K
        Tc_c2e = self.Tc_c2e
        link_poses = self.link_poses[idx]
        data_dict = {
            "rgb": rgb,
            # "depth": depth,
            "mask": mask,
            "qpos": qpos,
            "K": K,
            "link_poses": link_poses,
            "Tc_c2e": Tc_c2e,
        }
        if self.cfg.load_mask_pred:
            mask_pred = self.masks_pred[idx]
            data_dict["mask_pred"] = mask_pred
        # if self.cfg.load_mask_anno:
        #     mask_anno = self.masks_anno[idx]
        #     data_dict["mask_anno"] = mask_anno
        return data_dict
