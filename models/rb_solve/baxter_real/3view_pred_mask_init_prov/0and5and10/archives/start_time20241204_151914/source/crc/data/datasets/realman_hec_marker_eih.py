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


class RealmanHecMarkerEIHDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.realman_hec_marker
        self.camera_id = self.cfg.camera_id
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/{self.camera_id}/*.png"))[:ds_len]
        depth_paths = sorted(glob.glob(f"{data_dir}/depth/{self.camera_id}*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]
        Tc_c2m_paths = sorted(glob.glob(f"{data_dir}/Tc_c2m/{self.camera_id}/*.txt"))[:ds_len]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
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

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            qpos[0] += 0.0618
            self.qpos.append(qpos)
            pad = np.zeros(sk.robot.dof - qpos.shape[0])
            if self.camera_id == 'right':
                pq = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), 17)  # right link 6
            elif self.camera_id == 'left':
                pq = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), 16)  # left link 6
            else:
                raise NotImplementedError()
            Tb_b2e = pq.to_transformation_matrix()
            self.link_poses.append(Tb_b2e)
        for Tc_c2m_path in Tc_c2m_paths:
            Tc_c2m = np.loadtxt(Tc_c2m_path)
            self.Tc_c2ms.append(Tc_c2m)

        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/K_{self.camera_id}.txt")
        self.K = torch.from_numpy(self.K).float()
        self.Tc_c2e = np.eye(4)

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
