import os.path as osp
import glob
import cv2
import imageio
import numpy as np
import torch
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone


class RealmanDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.realman
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        self.camera_id = self.cfg.camera_id
        if self.cfg.resize[0] > 0 and self.cfg.resize[1] > 0:
            self.resize_h, self.resize_w = self.cfg.resize
        else:
            self.resize_w = self.resize_h = None
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/{self.camera_id}/*.png"))[:ds_len]
        gt_mask_paths = sorted(glob.glob(f"{data_dir}/gt_mask/{self.camera_id}/*.png"))[:ds_len]
        anno_mask_paths = sorted(glob.glob(f"{data_dir}/anno_mask/{self.camera_id}/*.png"))[:ds_len]
        pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/{self.camera_id}/*.png"))[:ds_len]
        # gripper_mask_paths = sorted(glob.glob(f"{data_dir}/pred_gripper_mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]

        if self.cfg.selected_indices != []:
            rgb_paths = [rgb_paths[i] for i in self.cfg.selected_indices]
            if len(gt_mask_paths) > 0:
                gt_mask_paths = [gt_mask_paths[i] for i in self.cfg.selected_indices]
            if len(anno_mask_paths) > 0:
                anno_mask_paths = [anno_mask_paths[i] for i in self.cfg.selected_indices]
            if len(pred_mask_paths) > 0:
                pred_mask_paths = [pred_mask_paths[i] for i in self.cfg.selected_indices]
            # if len(gripper_mask_paths) > 0:
            #     gripper_mask_paths = [gripper_mask_paths[i] for i in self.cfg.selected_indices]
            qpos_paths = [qpos_paths[i] for i in self.cfg.selected_indices]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        self.masks_pred = []
        self.masks_anno = []
        self.masks_gt = []
        self.qpos = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        self.original_img_size = None
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.original_img_size = rgb.shape[:2]
            if self.resize_w is not None:
                rgb = cv2.resize(rgb, (self.resize_w, self.resize_h))
            self.images.append(rgb)
        for pred_mask_path in pred_mask_paths:
            pred_mask = cv2.imread(pred_mask_path, 2) > 0
            self.masks_pred.append(pred_mask)
        if len(self.masks_pred) > 0:
            self.masks_pred = np.stack(self.masks_pred)
            self.masks_pred = torch.from_numpy(self.masks_pred).float()
        for anno_mask_path in anno_mask_paths:
            anno_mask = cv2.imread(anno_mask_path, 2) > 0
            if self.resize_w is not None:
                anno_mask = cv2.resize(anno_mask.astype(np.uint8), (self.resize_w, self.resize_h)) > 0
            self.masks_anno.append(anno_mask)
        if len(self.masks_anno) > 0:
            self.masks_anno = np.stack(self.masks_anno)
            self.masks_anno = torch.from_numpy(self.masks_anno).float()
        for gt_mask_path in gt_mask_paths:
            mask_gt = cv2.imread(gt_mask_path, 2) > 0
            self.masks_gt.append(mask_gt)
        if len(self.masks_gt) > 0:
            self.masks_gt = np.stack(self.masks_gt)
            self.masks_gt = torch.from_numpy(self.masks_gt).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            qpos[0] += 0.0618
            self.qpos.append(qpos)
            link_poses = []
            relative_to = self.cfg.relative_to
            if relative_to == 'base_link':
                relative_to_link_index = 0
            elif relative_to == 'Link_lbase':
                relative_to_link_index = 3
            elif relative_to == 'Link_right6':
                relative_to_link_index = 17
            elif relative_to == 'Link_rbase':
                relative_to_link_index = 4
            elif relative_to == 'Link_left6':
                relative_to_link_index = 16
            else:
                raise NotImplementedError()
            pad = np.zeros(sk.robot.dof - qpos.shape[0])
            relative_link_pose = sk.compute_forward_kinematics(np.concatenate([qpos, pad]),
                                                               relative_to_link_index).to_transformation_matrix()
            for link in self.cfg.use_links:
                Tb_b2l = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), link).to_transformation_matrix()
                Tb_b2l = np.linalg.inv(relative_link_pose) @ Tb_b2l
                link_poses.append(Tb_b2l)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        # if osp.exists(osp.join(self.data_dir, "link78regions.txt")):
        #     self.large_mask_regions = np.loadtxt(osp.join(self.data_dir, "link78regions.txt")).astype(int)
        # else:
        #     self.large_mask_regions = np.zeros([100, 4])
        self.K = np.loadtxt(f"{data_dir}/K_{self.camera_id}.txt")
        if self.resize_w is not None:
            self.K[0] *= self.resize_w / self.original_img_size[1]
            self.K[1] *= self.resize_h / self.original_img_size[0]
        self.K = torch.from_numpy(self.K).float()

        # dummy gt poses
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
        # if self.cfg.load_mask_gripper:
        #     mask_gripper = self.masks_gripper[idx]
        #     data_dict["mask_gripper"] = mask_gripper
        if self.cfg.load_mask_gt:
            data_dict["mask_gt"] = self.masks_gt[idx]

        if self.cfg.eye_in_hand is False:
            Tc_c2b = self.Tc_c2b
            data_dict["Tc_c2b"] = Tc_c2b
        else:
            Tc_c2e = self.Tc_c2e
            data_dict["Tc_c2e"] = Tc_c2e

        # data_dict['link78regions'] = self.large_mask_regions
        return data_dict

# 0 base_link
# 1 Link_rail
# 2 Link_head1
# 3 Link_lbase
# 4 Link_rbase
# 5 Link_head2
# 6 Link_left1
# 7 Link_right1
# 8 Link_left2
# 9 Link_right2
# 10 Link_left3
# 11 Link_right3
# 12 Link_left4
# 13 Link_right4
# 14', 'Link_left5', '
# 15 Link_right5',
# 16'Link_left6',
# 17    'Link_right6',
# 18 'gripper_base_link_left',
# 19 'gripper_base_link_right',
# 20 'finger_link_left1',
# 21 'finger_link_left2',
# 22 'finger_link_right1',
# 23 'finger_link_right2'
