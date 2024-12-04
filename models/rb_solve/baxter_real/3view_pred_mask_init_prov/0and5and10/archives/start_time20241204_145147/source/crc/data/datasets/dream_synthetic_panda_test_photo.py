import os.path as osp
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


class DreamSyntheticPandaTestPhotoDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.dream_synthetic_panda_test_photo
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.jpg"))[:ds_len]
        # depth_paths = sorted(glob.glob(f"{data_dir}/depth/*.png"))[:ds_len]
        gt_mask_paths = sorted(glob.glob(f"{data_dir}/mask/*.png"))[:ds_len]
        anno_mask_paths = sorted(glob.glob(f"{data_dir}/anno_mask/*.png"))[:ds_len]
        pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/*.jpg"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        self.masks_gt = []
        self.masks_pred = []
        self.masks_anno = []
        self.qpos = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        for gt_mask_path in gt_mask_paths:
            gt_mask = imageio.imread_v2(gt_mask_path) > 0
            self.masks_gt.append(gt_mask)
        if len(self.masks_gt) > 0:
            self.masks_gt = np.stack(self.masks_gt)
            self.masks_gt = torch.from_numpy(self.masks_gt).float()
        for pred_mask_path in pred_mask_paths:
            pred_mask = imageio.imread_v2(pred_mask_path)[..., 0] > 0
            self.masks_pred.append(pred_mask)
        if len(self.masks_pred) > 0:
            self.masks_pred = np.stack(self.masks_pred)
            self.masks_pred = torch.from_numpy(self.masks_pred).float()
        for anno_mask_path in anno_mask_paths:
            anno_mask = imageio.imread_v2(anno_mask_path)[..., 0] > 0
            self.masks_anno.append(anno_mask)
        if len(self.masks_anno) > 0:
            self.masks_anno = np.stack(self.masks_anno)
            self.masks_anno = torch.from_numpy(self.masks_anno).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            self.qpos.append(qpos)
            link_poses = []
            for link in self.cfg.use_links:
                # pad = np.zeros(sk.robot.dof - qpos.shape[0])
                pq = sk.compute_forward_kinematics(qpos, link)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.Rt_to_pose(R, t)
                link_poses.append(pose_eb)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/../K.txt")
        self.K = torch.from_numpy(self.K).float()
        self.Tc_c2b = np.loadtxt(f"{data_dir}/Tc_c2b.txt")
        self.Tc_c2b = torch.from_numpy(self.Tc_c2b).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]
        # depth = self.depths[idx]
        mask = self.masks_gt[idx]

        qpos = self.qpos[idx]
        K = self.K
        Tc_c2b = self.Tc_c2b
        link_poses = self.link_poses[idx]
        data_dict = {
            "rgb": rgb,
            # "depth": depth,
            "mask": mask,
            "qpos": qpos,
            "K": K,
            "link_poses": link_poses,
            "Tc_c2b": Tc_c2b,
        }
        if self.cfg.load_mask_pred:
            mask_pred = self.masks_pred[idx]
            data_dict["mask_pred"] = mask_pred
        if self.cfg.load_mask_anno:
            mask_anno = self.masks_anno[idx]
            data_dict["mask_anno"] = mask_anno
        return data_dict


def main():
    from crc.engine.defaults import setup
    from crc.engine.defaults import default_argument_parser
    from crc.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/rb_solve/dream/real/panda_3cam_realsense/0.yaml'
    cfg = setup(args, do_archive=False)
    dl = make_data_loader(cfg, is_train=True)
    ds = dl.dataset
    vis3d = Vis3D(xyz_pattern=('x', '-y', '-z'), out_folder="dbg",
                  sequence_name="dream_real_panda_3cam_realsense_loader")
    for i in range(len(ds)):
        dd = ds[i]
        rgb = dd['rgb']
        if "mask_anno" in dd:
            mask_anno = dd['mask_anno']
            tmp = plt_utils.vis_mask(rgb, mask_anno.numpy().astype(np.uint8), [255, 0, 0])
            plt.imshow(tmp)
            plt.show()
        # depth = dd["depth"]
        # K = dd["K"].numpy()
        # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
        # vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3))
        vis3d.add_panda(dd['qpos'].tolist(), dd['Tc_c2b'].numpy())
        vis3d.increase_scene_id()
        print()


if __name__ == '__main__':
    main()
