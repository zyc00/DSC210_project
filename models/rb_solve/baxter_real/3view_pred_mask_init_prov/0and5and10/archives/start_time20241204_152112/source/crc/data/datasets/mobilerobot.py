import os.path as osp
import matplotlib.pyplot as plt
import glob
import cv2
import imageio
import numpy as np
import torch
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone


class MobileRobotDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.mobilerobot
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.png"))[:ds_len]
        gt_mask_paths = sorted(glob.glob(f"{data_dir}/gt_mask/*.png"))[:ds_len]
        anno_mask_paths = sorted(glob.glob(f"{data_dir}/anno_mask/*.png"))[:ds_len]
        pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/*.png"))[:ds_len]
        gripper_mask_paths = sorted(glob.glob(f"{data_dir}/pred_gripper_mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]

        if self.cfg.selected_indices != []:
            rgb_paths = [rgb_paths[i] for i in self.cfg.selected_indices]
            if len(gt_mask_paths) > 0:
                gt_mask_paths = [gt_mask_paths[i] for i in self.cfg.selected_indices]
            if len(anno_mask_paths) > 0:
                anno_mask_paths = [anno_mask_paths[i] for i in self.cfg.selected_indices]
            pred_mask_paths = [pred_mask_paths[i] for i in self.cfg.selected_indices]
            if len(gripper_mask_paths) > 0:
                gripper_mask_paths = [gripper_mask_paths[i] for i in self.cfg.selected_indices]
            qpos_paths = [qpos_paths[i] for i in self.cfg.selected_indices]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        self.masks_pred = []
        self.masks_gripper = []
        self.masks_anno = []
        self.masks_gt = []
        self.qpos = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        for pred_mask_path in pred_mask_paths:
            pred_mask = cv2.imread(pred_mask_path, 2) > 0
            self.masks_pred.append(pred_mask)
        if len(self.masks_pred) > 0:
            self.masks_pred = np.stack(self.masks_pred)
            self.masks_pred = torch.from_numpy(self.masks_pred).float()
        for anno_mask_path in anno_mask_paths:
            anno_mask = cv2.imread(anno_mask_path, 2) > 0
            self.masks_anno.append(anno_mask)
        if len(self.masks_anno) > 0:
            self.masks_anno = np.stack(self.masks_anno)
            self.masks_anno = torch.from_numpy(self.masks_anno).float()
        for gripper_mask_path in gripper_mask_paths:
            gripper_mask = cv2.imread(gripper_mask_path, 2) > 0
            self.masks_gripper.append(gripper_mask)
        if len(self.masks_gripper) > 0:
            self.masks_gripper = np.stack(self.masks_gripper)
            self.masks_gripper = torch.from_numpy(self.masks_gripper).float()
        for gt_mask_path in gt_mask_paths:
            mask_gt = cv2.imread(gt_mask_path, 2) > 0
            self.masks_gt.append(mask_gt)
        if len(self.masks_gt) > 0:
            self.masks_gt = np.stack(self.masks_gt)
            self.masks_gt = torch.from_numpy(self.masks_gt).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            self.qpos.append(qpos)
            link_poses = []
            for link in self.cfg.use_links:
                pad = np.zeros(sk.robot.dof - qpos.shape[0])
                Tb_b2l = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), link).to_transformation_matrix()
                Te_e2ep = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0.005],
                                    [0, 0, 0, 1]])
                if link > 8 and self.cfg.shift_gripper is True:
                    Tb_b2e = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), 8).to_transformation_matrix()
                    Te_e2l = np.linalg.inv(Tb_b2e) @ Tb_b2l
                    Tb_b2l = Tb_b2e @ Te_e2ep @ Te_e2l
                link_poses.append(Tb_b2l)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        if osp.exists(osp.join(self.data_dir, "link78regions.txt")):
            self.large_mask_regions = np.loadtxt(osp.join(self.data_dir, "link78regions.txt")).astype(int)
        else:
            self.large_mask_regions = np.zeros([100, 4])
        self.K = np.loadtxt(f"{data_dir}/K.txt")
        self.K = torch.from_numpy(self.K).float()
        if self.cfg.eye_in_hand is False:
            if osp.exists(osp.join(self.data_dir, "Tc_c2b.txt")):
                self.Tc_c2b = np.loadtxt(f"{data_dir}/Tc_c2b.txt")
            else:
                self.Tc_c2b = np.eye(4)
            self.Tc_c2b = torch.from_numpy(self.Tc_c2b).float()

        else:
            if osp.exists(osp.join(self.data_dir, "Tc_c2e.txt")):
                self.Tc_c2e = np.loadtxt(f"{data_dir}/Tc_c2e.txt")
            else:
                self.Tc_c2e = np.eye(4)
            self.Tc_c2e = torch.from_numpy(self.Tc_c2e).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]

        qpos = self.qpos[idx]
        K = self.K
        link_poses = self.link_poses[idx]
        data_dict = {
            "rgb": rgb,
            "qpos": qpos,
            "K": K,
            "link_poses": link_poses,
        }
        if self.cfg.load_mask_pred:
            mask_pred = self.masks_pred[idx]
            data_dict["mask_pred"] = mask_pred
        if self.cfg.load_mask_anno:
            mask_anno = self.masks_anno[idx]
            data_dict["mask_anno"] = mask_anno
        if self.cfg.load_mask_gripper:
            mask_gripper = self.masks_gripper[idx]
            data_dict["mask_gripper"] = mask_gripper
        if self.cfg.load_mask_gt:
            data_dict["mask_gt"] = self.masks_gt[idx]

        if self.cfg.eye_in_hand is False:
            Tc_c2b = self.Tc_c2b
            data_dict["Tc_c2b"] = Tc_c2b
        else:
            Tc_c2e = self.Tc_c2e
            data_dict["Tc_c2e"] = Tc_c2e

        data_dict['link78regions'] = self.large_mask_regions
        return data_dict
