import glob
import io
import os
import os.path as osp

import cv2
import imageio
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import transforms3d
import trimesh
from PIL import Image
from dl_ext.timer import EvalTime
from pytorch3d.transforms import se3_log_map, se3_exp_map
from skimage.io import imread
from tensorboardX import SummaryWriter
from torch import nn

from crc.structures.nvdiffrast_renderer import NVDiffrastRenderer
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import utils_3d, render_api, plt_utils
from crc.utils.pn_utils import to_array
from crc.utils.vis3d_ext import Vis3D

sk = SAPIENKinematicsModelStandalone("data/xarm7_textured.urdf")


class Model(nn.Module):
    def __init__(self, mesh_dir, links, K, init_op, refs, frame_ids, data_dir,
                 dbg, neg_iou_loss=False):
        super(Model, self).__init__()
        self.dbg = dbg
        self.neg_iou_loss = neg_iou_loss
        for link in links:
            mesh = trimesh.load(osp.join(mesh_dir, f"link{link}.STL"))
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f'vertices_{link}', vertices)
            self.register_buffer(f'faces_{link}', faces)
        self.links = links
        self.K = torch.as_tensor(K).float().cuda()
        self.frame_ids = frame_ids
        for k, v in refs.items():
            self.register_buffer(f'image_ref_{k}', v)

        # camera parameters
        init_dof = se3_log_map(torch.as_tensor(init_op, dtype=torch.float32)[None].permute(0, 2, 1))[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        # setup renderer
        self.H, self.W = refs[list(refs.keys())[0]].shape[:2]
        # self.orig_size = max(refs[list(refs.keys())[0]].shape)
        self.renderer = NVDiffrastRenderer([self.H, self.W])

        self.pose_ebs = {}
        for frame_id in frame_ids:
            pose_eb_qpos = np.loadtxt(osp.join(data_dir, f"pose_eb_{frame_id:06d}.txt"))
            for link in self.links:
                pq = sk.compute_forward_kinematics(pose_eb_qpos, link + 1)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.matrix_3x4_to_4x4(np.concatenate([R, t[:, None]], axis=-1))
                self.pose_ebs[f"frame_{frame_id}_link_{link}"] = torch.from_numpy(pose_eb).cuda().float()

    def forward(self):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="nvdrender_based_solver",
                      auto_increase=True, enable=False)
        evaltime = EvalTime(disable=True)
        evaltime('')
        pose_bc = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        losses = []
        all_frame_all_link_si = []
        for frame_id in self.frame_ids:
            all_link_si = []
            for link in self.links:
                pose_ec = pose_bc @ self.pose_ebs[f"frame_{frame_id}_link_{link}"]
                # vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, pose_ec), mesh.faces)
                verts, faces = getattr(self, f"vertices_{link}"), getattr(self, f"faces_{link}")
                vis3d.add_mesh(utils_3d.transform_points(verts, pose_ec), faces, name=f"link{link}")
                si = self.renderer.render_mask(verts, faces, K=self.K, object_pose=pose_ec)
                # si = interpolate(si[None, None], size=self.orig_size)[0, 0]
                # si = si[:self.H, :self.W]
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            if self.neg_iou_loss:
                predict = all_link_si
                target = getattr(self, f"image_ref_{frame_id}")
                # dims = tuple(range(predict.ndimension())[1:])
                intersect = (predict * target).sum()
                union = (predict + target - predict * target).sum() + 1e-6
                loss = 1. - (intersect / union)

                # loss = neg_iou_loss(all_link_si, )
            else:
                loss = torch.sum((all_link_si - getattr(self, f"image_ref_{frame_id}")) ** 2)
            losses.append(loss)
        evaltime('3')
        loss = torch.stack(losses).mean()
        return loss, all_frame_all_link_si


class ModelEyeOnHand(nn.Module):
    def __init__(self, mesh_dir, links, K, init_op, refs, frame_ids, data_dir, dbg):
        """

        @param mesh_dir:
        @param links:
        @param K:
        @param init_op: init transformation: TC_CtoE
        @param refs:
        @param frame_ids:
        @param data_dir:
        """
        super(ModelEyeOnHand, self).__init__()
        self.dbg = dbg
        for link in links:
            mesh = trimesh.load(osp.join(mesh_dir, f"link{link}.STL"))
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f'vertices_{link}', vertices)
            self.register_buffer(f'faces_{link}', faces)
        self.links = links
        self.K = torch.as_tensor(K).float().cuda()
        self.frame_ids = frame_ids
        for k, v in refs.items():
            self.register_buffer(f'image_ref_{k}', v)

        # camera parameters
        init_dof = se3_log_map(torch.as_tensor(init_op, dtype=torch.float32)[None].permute(0, 2, 1))[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        # setup renderer
        self.H, self.W = refs[list(refs.keys())[0]].shape[:2]
        self.orig_size = max(refs[list(refs.keys())[0]].shape)
        # renderer = nr.Renderer(image_size=render_size, orig_size=self.orig_size)  # todo: 64,128,256
        # self.renderer = renderer
        self.renderer = NVDiffrastRenderer([self.H, self.W])

        self.Tb_b2l = {}
        for frame_id in frame_ids:
            pose_eb_qpos = np.loadtxt(osp.join(data_dir, f"pose_eb_{frame_id:06d}.txt"))
            for link in self.links:
                pq = sk.compute_forward_kinematics(pose_eb_qpos, link + 1)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.matrix_3x4_to_4x4(np.concatenate([R, t[:, None]], axis=-1))
                self.Tb_b2l[f"frame_{frame_id}_link_{link}"] = torch.from_numpy(pose_eb).cuda().float()

    def forward(self):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="nvdrender_based_solver",
                      auto_increase=True, enable=self.dbg)
        evaltime = EvalTime(disable=True)
        evaltime('')
        Tc_c2e = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        losses = []
        all_frame_all_link_si = []
        for frame_id in self.frame_ids:
            all_link_si = []
            for link in self.links:
                Tb_b2e = self.Tb_b2l[f'frame_{frame_id}_link_{self.links[-1]}']
                Tc_c2l = Tc_c2e @ Tb_b2e.inverse() @ self.Tb_b2l[f"frame_{frame_id}_link_{link}"]
                # R = Tc_c2l[:3, :3]
                # t = Tc_c2l[:3, 3]
                verts, faces = getattr(self, f"vertices_{link}"), getattr(self, f"faces_{link}")
                vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link}")
                si = self.renderer.render_mask(verts, faces, self.K, Tc_c2l)
                if self.dbg:
                    pyrender_mask = render_api.pyrender_render_mesh_api(
                        trimesh.Trimesh(to_array(verts), to_array(faces)),
                        Tc_c2l, self.H, self.W, self.K.cpu().numpy())
                    print(f'link {link}, {np.unique(pyrender_mask)}')
                    vis3d.add_image(pyrender_mask, name=f'pyrender_mask{link}')
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            loss = torch.sum((all_link_si - getattr(self, f"image_ref_{frame_id}")) ** 2)
            losses.append(loss)
        evaltime('3')
        loss = torch.stack(losses).mean()
        return loss, all_frame_all_link_si


class NVDiffrastRenderBasedSolver:
    def __init__(self, cfg):
        # cfg = cfg.rb_solve
        self.cfg = cfg.rb_solve
        self.data_dir = self.cfg.data_dir
        self.eye_on_hand = self.cfg.eye_on_hand
        self.init_Tbc = self.cfg.init_Tbc
        self.init_Tc_c2e = self.cfg.init_Tc_c2e

        self.masks = self.load_mask()
        if self.cfg.nimgs > 0:
            new_keys = list(self.masks.keys())[:self.cfg.nimgs]
            self.masks = {k: self.masks[k] for k in new_keys}
        loguru.logger.info(f"using {len(self.masks)} masks!")
        self.K = np.loadtxt(osp.join(self.data_dir, "K.txt"))
        self.mesh_dir = "data/xarm_description/meshes/xarm7/visual"
        self.links = self.cfg.links
        self.test_links = [0, 1, 2, 3, 4, 5, 6, 7]
        self.solved_op = None
        self.output_dir = cfg.output_dir
        self._tb_writer = None
        self.history_ops = []

    def solve(self):
        steps = self.cfg.steps
        lrs = self.cfg.lrs
        assert len(lrs) == len(steps)
        tb_writer = self.tb_writer
        for step, lr in zip(steps, lrs):
            if not self.eye_on_hand:
                init_Tbc = self.init_Tbc
                model = Model(self.mesh_dir, self.links, self.K, init_Tbc, self.masks,
                              list(self.masks.keys()), self.data_dir, dbg=self.cfg.dbg,
                              neg_iou_loss=self.cfg.neg_iou_loss)
            else:
                init_Tc_c2e = self.init_Tc_c2e
                model = ModelEyeOnHand(self.mesh_dir, self.links, self.K, init_Tc_c2e, self.masks,
                                       list(self.masks.keys()), self.data_dir, self.cfg.dbg)
            model.cuda()
            if self.cfg.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif self.cfg.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                raise NotImplementedError()
            loop = tqdm.tqdm(range(step))
            plt_utils.image_grid(list(self.masks.values()), show=False)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            tb_writer.add_image("gt", np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1))
            plt.close("all")
            for i in loop:
                optimizer.zero_grad()
                loss, si = model()
                self.history_ops.append(model.dof.detach().cpu().numpy())
                tb_writer.add_scalar(f"loss", loss.item(), i)
                loss.backward()
                optimizer.step()
                # image = si.detach().cpu().numpy()
                if i % self.cfg.log_interval == 0:
                    plt_utils.image_grid(torch.stack(si), show=False)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format="png")
                    tb_writer.add_image(f"optim",
                                        np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), i)
                    plt.close("all")
                    # visualize mask error
                    err_map = (torch.stack(list(self.masks.values())).cuda() - torch.stack(si)).abs()
                    plt_utils.image_grid(err_map, show=False)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format="png")
                    tb_writer.add_image(f"error_map",
                                        np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), i)
                    plt.close("all")

                    op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
                    self.solved_op = op
                    loguru.logger.info(f"step {i} {np.array2string(to_array(op), separator=',')}")
                    np.savetxt(osp.join(self.output_dir, "result.txt"), to_array(op))
                    self.evaluate(i)
                    np.savetxt(osp.join(self.output_dir, "history_dof6.txt"), np.stack(self.history_ops))
                loop.set_description('Optimizing (loss %.4f)' % loss.data)
            op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
            self.solved_op = op
            np.savetxt(osp.join(self.output_dir, "history_dof6.txt"), np.stack(self.history_ops))
            np.savetxt(osp.join(self.output_dir, "result.txt"), to_array(op))
            self.evaluate(i)
        return to_array(self.solved_op)

    def load_mask(self):
        if self.cfg.use_gt_mask:
            loguru.logger.info("USING GT MASK!")
            mask_paths = sorted(glob.glob(osp.join(self.data_dir, "gt_mask*png")))
        else:
            mask_paths = sorted(glob.glob(osp.join(self.data_dir, "mask*png")))
        image_refs = {}
        for mask_path in mask_paths:
            mask = imread(mask_path)[:, :, 0] > 0
            # mask = imread(osp.join(self.data_dir, "mask_000001.png"))
            # mask = mask[:, :, 0] > 0
            image_ref = torch.from_numpy(mask.astype(np.float32))
            frame_id = int(mask_path.rstrip(".png")[-6:])
            image_refs[frame_id] = image_ref
        return image_refs

    def validate(self, data_dir=""):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="render_based_solver_validate",
                      auto_increase=True, enable=True)
        if data_dir == "":
            data_dir = self.data_dir
        rgb_paths = sorted(glob.glob(osp.join(data_dir, "color/*.png")))
        depth_paths = sorted(glob.glob(osp.join(data_dir, "depth/*.png")))
        pose_eb_paths = sorted(glob.glob(osp.join(data_dir, "qpos/*.txt")))
        if self.cfg.nimgs > 0:
            rgb_paths = rgb_paths[:self.cfg.nimgs]
            depth_paths = depth_paths[:self.cfg.nimgs]
            pose_eb_paths = pose_eb_paths[:self.cfg.nimgs]
        K = np.loadtxt(osp.join(data_dir, "K.txt"))
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        test_link_meshes = {}
        for link in self.test_links:
            mesh = trimesh.load(osp.join(self.mesh_dir, f"link{link}.STL"))
            test_link_meshes[link] = mesh
        for i in tqdm.trange(len(rgb_paths)):
            vis3d.set_scene_id(i)
            rgb = imageio.imread_v2(rgb_paths[i])
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            H, W = rgb.shape[:2]
            depth = cv2.imread(depth_paths[i], 2).astype(np.float32) / 1000.0
            pose_eb_qpos = np.loadtxt(pose_eb_paths[i])

            pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
            vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=20)
            if not self.eye_on_hand:
                Tc_c2b = self.solved_op
            else:
                Tc_c2e = self.solved_op
            all_rendered_mask = []
            for link in self.test_links:
                pq = sk.compute_forward_kinematics(pose_eb_qpos, link + 1)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                mesh = test_link_meshes[link]
                if not self.eye_on_hand:
                    Tb_b2l = utils_3d.Rt_to_pose(R, t)
                    Tb_b2l = torch.from_numpy(Tb_b2l).cuda().float()
                    Tc_c2l = to_array(Tc_c2b @ Tb_b2l)
                else:
                    Tb_b2l = utils_3d.Rt_to_pose(R, t)
                    pq = sk.compute_forward_kinematics(pose_eb_qpos, self.test_links[-1] + 1)
                    R = transforms3d.quaternions.quat2mat(pq.q)
                    t = pq.p
                    Tb_b2e = utils_3d.Rt_to_pose(R, t)
                    Tc_c2l = to_array(Tc_c2e) @ np.linalg.inv(Tb_b2e) @ Tb_b2l
                rendered_mask = render_api.pyrender_render_mesh_api(mesh, Tc_c2l, H, W, K)
                vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, Tc_c2l), mesh.faces, name=f"link{link}")
                all_rendered_mask.append(rendered_mask)
            all_rendered_mask = np.clip(np.sum(all_rendered_mask, axis=0), a_min=0, a_max=1).astype(bool)
            tmp = rgb.copy() / 255.0
            tmp[all_rendered_mask] *= np.array([0.5, 1, 1])
            tmp[all_rendered_mask] += np.array([0.5, 0, 0])
            vis3d.add_image(tmp, name='optim')
            if data_dir == self.data_dir:
                anno_mask = to_array(self.masks[i]).astype(bool)
                tmp = rgb.copy() / 255.0
                tmp[anno_mask] *= np.array([0.5, 1, 1])
                tmp[anno_mask] += np.array([0.5, 0, 0])
                vis3d.add_image(tmp, name='anno')
            print()
        # print()

    def load_result(self):
        self.solved_op = torch.from_numpy(np.loadtxt(osp.join(self.output_dir, "result.txt"))).cuda().float()

    def evaluate(self, step=0):
        if osp.exists(osp.join(self.data_dir, "campose.txt")):
            gt = np.loadtxt(osp.join(self.data_dir, "campose.txt"))
            gt = torch.from_numpy(gt).float()
            gt_dof6 = utils_3d.se3_log_map(gt[None].permute(0, 2, 1), backend='opencv')[0]
            solved_dof6 = utils_3d.se3_log_map(self.solved_op[None].permute(0, 2, 1).cpu(), backend='opencv')[0]
            trans_err = ((gt_dof6[:3] - solved_dof6[:3]) * 100).abs().tolist()
            rot_err = (gt_dof6[3:] - solved_dof6[3:]).abs().max() / np.pi * 180
            # loguru.logger.info(f"trans error {trans_err} cm, ")
            loguru.logger.info(f"norm {np.linalg.norm(trans_err):.4f} cm")
            loguru.logger.info(f"rot error {rot_err:.4f} degree")
            self.tb_writer.add_scalar("eval/trans_x", trans_err[0], step)
            self.tb_writer.add_scalar("eval/trans_y", trans_err[1], step)
            self.tb_writer.add_scalar("eval/trans_z", trans_err[2], step)
            self.tb_writer.add_scalar("eval/rot", rot_err, step)
            self.tb_writer.add_scalar("eval/trans", np.linalg.norm(trans_err), step)
        else:
            loguru.logger.info("no gt found.")

    @property
    def tb_writer(self):
        if self._tb_writer is None:
            self._tb_writer = SummaryWriter(self.output_dir, flush_secs=20)
        return self._tb_writer


def make_gif(filename):
    with imageio.get_writer(filename, mode='I', fps=10) as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()
