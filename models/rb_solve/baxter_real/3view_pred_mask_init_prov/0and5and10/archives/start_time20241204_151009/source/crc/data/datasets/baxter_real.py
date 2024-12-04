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


class BaxterRealDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.baxter_real
        selected_indices = cfg.dataset.baxter_real.selected_indices
        self.data_dir = data_dir
        print(data_dir)
        exit()
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.png"))[:ds_len]
        # depth_paths = sorted(glob.glob(f"{data_dir}/depth/*.png"))[:ds_len]
        # gt_mask_paths = sorted(glob.glob(f"{data_dir}/gt_mask/*.png"))[:ds_len]
        anno_mask_paths = sorted(glob.glob(f"{data_dir}/anno_mask/*.png"))[:ds_len]
        pred_mask_paths = sorted(glob.glob(f"{data_dir}/pred_mask/*.png"))[:ds_len]
        sam_mask_paths = sorted(glob.glob(f"{data_dir}/sam_mask/*.png"))[:ds_len]
        gsam_mask_paths = sorted(glob.glob(f"{data_dir}/gsam_mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]
        if selected_indices == []:
            selected_indices = range(len(rgb_paths))
        if len(rgb_paths) > 0:
            rgb_paths = [rgb_paths[i] for i in selected_indices]
        if len(anno_mask_paths) > 0:
            anno_mask_paths = [anno_mask_paths[i] for i in selected_indices]
        if len(pred_mask_paths) > 0:
            pred_mask_paths = [pred_mask_paths[i] for i in selected_indices]
        if len(sam_mask_paths) > 0:
            sam_mask_paths = [sam_mask_paths[i] for i in selected_indices]
        if len(gsam_mask_paths) > 0:
            gsam_mask_paths = [gsam_mask_paths[i] for i in selected_indices]
        if len(qpos_paths) > 0:
            qpos_paths = [qpos_paths[i] for i in selected_indices]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        # self.masks = []
        self.masks_pred = []
        self.masks_sam = []
        self.masks_anno = []
        self.masks_gsam = []
        self.qpos = []
        # self.depths = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        for pred_mask_path in pred_mask_paths:
            pred_mask = imageio.imread_v2(pred_mask_path)[..., 0] > 0
            self.masks_pred.append(pred_mask)
        if len(self.masks_pred) > 0:
            self.masks_pred = np.stack(self.masks_pred)
            self.masks_pred = torch.from_numpy(self.masks_pred).float()

        for sam_mask_path in sam_mask_paths:
            sam_mask = imageio.imread_v2(sam_mask_path)[..., 0] > 0
            self.masks_sam.append(sam_mask)
        if len(self.masks_sam) > 0:
            self.masks_sam = np.stack(self.masks_sam)
            self.masks_sam = torch.from_numpy(self.masks_sam).float()

        for gsam_mask_path in gsam_mask_paths:
            gsam_mask = cv2.imread(gsam_mask_path, 2) > 0
            self.masks_gsam.append(gsam_mask)
        if len(self.masks_gsam) > 0:
            self.masks_gsam = np.stack(self.masks_gsam)
            self.masks_gsam = torch.from_numpy(self.masks_gsam).float()

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
                pad = np.zeros(sk.robot.dof - qpos.shape[0])
                pq = sk.compute_forward_kinematics(np.concatenate([pad, qpos]), link)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.Rt_to_pose(R, t)
                link_poses.append(pose_eb)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/newK.txt")
        self.K = torch.from_numpy(self.K).float()
        self.Tc_c2b = np.loadtxt(f"{data_dir}/Tc_c2b.txt")
        self.Tc_c2b = torch.from_numpy(self.Tc_c2b).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]
        # depth = self.depths[idx]
        # mask = self.masks[idx]

        qpos = self.qpos[idx]
        K = self.K
        Tc_c2b = self.Tc_c2b
        link_poses = self.link_poses[idx]
        data_dict = {
            "rgb": rgb,
            # "depth": depth,
            # "mask": mask,
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
        if self.cfg.load_mask_sam:
            mask_sam = self.masks_sam[idx]
            data_dict["mask_sam"] = mask_sam
        if self.cfg.load_mask_gsam:
            mask_gsam = self.masks_gsam[idx]
            data_dict["mask_gsam"] = mask_gsam
        return data_dict


def main():
    from crc.engine.defaults import setup
    from crc.engine.defaults import default_argument_parser
    from crc.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/rb_solve/baxter_real/0.yaml'
    cfg = setup(args, do_archive=False)
    dl = make_data_loader(cfg, is_train=True)
    ds = dl.dataset
    vis3d = Vis3D(xyz_pattern=('x', '-y', '-z'), out_folder="dbg", sequence_name="baxter_real_loader")
    for i in range(len(ds)):
        dd = ds[i]
        rgb = dd['rgb']
        if "mask_anno" in dd:
            mask_anno = dd['mask_anno']
            tmp = plt_utils.vis_mask(rgb, mask_anno.numpy().astype(np.uint8))
            plt.imshow(tmp)
            plt.show()
        # depth = dd["depth"]
        # K = dd["K"].numpy()
        # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
        # vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3))
        vis3d.add_baxter([0] * 8 + dd['qpos'].tolist(), dd['Tc_c2b'].numpy())
        vis3d.increase_scene_id()
        print()


if __name__ == '__main__':
    main()
