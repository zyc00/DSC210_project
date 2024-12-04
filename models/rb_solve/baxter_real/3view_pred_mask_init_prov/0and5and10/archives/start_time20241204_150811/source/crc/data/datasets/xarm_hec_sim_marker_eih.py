import os.path as osp

import loguru
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


class XarmHecSimMarkerEIHDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.xarm_hec_sim_marker
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.png"))[:ds_len]
        depth_paths = sorted(glob.glob(f"{data_dir}/depth/*.png"))[:ds_len]
        # gt_mask_paths = sorted(glob.glob(f"{data_dir}/gt_mask/*.png"))[:ds_len]
        # anno_mask_paths = sorted(glob.glob(f"{data_dir}/anno_mask/*.png"))[:ds_len]
        # pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]
        Tc_c2m_paths = sorted(glob.glob(f"{data_dir}/Tc_c2m/*.txt"))[:ds_len]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        # links = sk.robot.get_links()
        # links = [links[ul] for ul in self.cfg.use_links]
        self.images = []
        # self.masks = []
        # self.masks_pred = []
        # self.masks_anno = []
        self.qpos = []
        self.depths = []
        self.link_poses = []
        self.Tc_c2ms = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        for depth_path in depth_paths:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            self.depths.append(depth)
        # for gt_mask_path in gt_mask_paths:
        #     gt_mask = np.array(imageio.imread_v2(gt_mask_path))[..., 0] > 0
        #     self.masks.append(gt_mask)
        # self.masks = np.stack(self.masks)
        # self.masks = torch.from_numpy(self.masks).float()
        # for pred_mask_path in pred_mask_paths:
        #     pred_mask = imageio.imread_v2(pred_mask_path)[..., 0] > 0
        #     self.masks_pred.append(pred_mask)
        # if len(self.masks_pred) > 0:
        #     self.masks_pred = np.stack(self.masks_pred)
        #     self.masks_pred = torch.from_numpy(self.masks_pred).float()
        # for anno_mask_path in anno_mask_paths:
        #     anno_mask = imageio.imread_v2(anno_mask_path)[..., 0] > 0
        #     self.masks_anno.append(anno_mask)
        # if len(self.masks_anno) > 0:
        #     self.masks_anno = np.stack(self.masks_anno)
        #     self.masks_anno = torch.from_numpy(self.masks_anno).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            self.qpos.append(qpos)
            # link_poses = []
            # for link in self.cfg.use_links:
            pad = np.zeros(sk.robot.dof - qpos.shape[0])
            pq = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), 8)
            R = transforms3d.quaternions.quat2mat(pq.q)
            t = pq.p
            Tb_b2e = utils_3d.Rt_to_pose(R, t)
            # link_poses.append(pose_eb)
            # link_poses = np.stack(link_poses)
            self.link_poses.append(Tb_b2e)
        for Tc_c2m_path in Tc_c2m_paths:
            Tc_c2m = np.loadtxt(Tc_c2m_path)
            self.Tc_c2ms.append(Tc_c2m)

        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/K.txt")
        self.K = torch.from_numpy(self.K).float()
        # if osp.exists(f"{data_dir}/Tc_c2e.txt"):
        self.Tc_c2e = np.loadtxt(f"{data_dir}/Tc_c2e.txt")
        # else:
        #     loguru.logger.warning(f"{data_dir}/Tc_c2e.txt not found, using default value")

        self.Tc_c2e = torch.from_numpy(self.Tc_c2e).float()
        if len(self.Tc_c2ms) > 0:
            self.Tc_c2ms = np.stack(self.Tc_c2ms)
            self.Tc_c2ms = torch.from_numpy(self.Tc_c2ms).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]
        # depth = self.depths[idx]
        # mask = self.masks[idx]

        qpos = self.qpos[idx]
        K = self.K
        Tc_c2e = self.Tc_c2e
        link_poses = self.link_poses[idx]
        data_dict = {
            "rgb": rgb,
            # "depth": depth,
            # "mask": mask,
            # "Tc_c2m": Tc_c2m,
            "qpos": qpos,
            "K": K,
            "link_poses": link_poses,
            "Tc_c2e": Tc_c2e,
        }
        if len(self.Tc_c2ms) > 0:
            Tc_c2m = self.Tc_c2ms[idx]
            data_dict["Tc_c2m"] = Tc_c2m

        # if self.cfg.load_mask_pred:
        #     mask_pred = self.masks_pred[idx]
        #     data_dict["mask_pred"] = mask_pred
        # if self.cfg.load_mask_anno:
        #     mask_anno = self.masks_anno[idx]
        #     data_dict["mask_anno"] = mask_anno
        return data_dict


def main():
    from crc.engine.defaults import setup
    from crc.engine.defaults import default_argument_parser
    from crc.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/hand_eye_solver/version1_000000.yaml'
    cfg = setup(args, do_archive=False)
    dl = make_data_loader(cfg, is_train=True)
    ds = dl.dataset
    vis3d = Vis3D(xyz_pattern=('x', '-y', '-z'), out_folder="dbg", sequence_name="xarm_hec_sim_marker_loader")
    for i in range(len(ds)):
        dd = ds[i]
        rgb = dd['rgb']
        if "mask_anno" in dd:
            mask_anno = dd['mask_anno']
            tmp = plt_utils.vis_mask(rgb, mask_anno.numpy().astype(np.uint8))
            plt.imshow(tmp)
            plt.show()
        # depth = dd["depth"]
        K = dd["K"].numpy()
        # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
        # vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3))
        vis3d.add_xarm(dd['qpos'].tolist() + [0, 0], dd['Tc_c2e'].numpy())
        vis3d.increase_scene_id()
        print()


if __name__ == '__main__':
    main()
