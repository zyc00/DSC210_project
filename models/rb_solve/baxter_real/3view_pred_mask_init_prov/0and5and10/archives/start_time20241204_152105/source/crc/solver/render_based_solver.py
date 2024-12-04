import glob
import io
import os

import cv2
import imageio
import loguru
import matplotlib.pyplot as plt
import os.path as osp

import numpy as np
import torch
import tqdm
import transforms3d
import trimesh
from PIL import Image
from dl_ext.timer import EvalTime
from pytorch3d.transforms import se3_log_map, se3_exp_map
from skimage.io import imread, imsave
from tensorboardX import SummaryWriter
from torch import nn
# import neural_renderer as nr

from crc.modeling.misc import interpolate
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import utils_3d, render_api, plt_utils
from crc.utils.pn_utils import to_array
from crc.utils.vis3d_ext import Vis3D

sk = SAPIENKinematicsModelStandalone("data/xarm7_textured.urdf")


class Model(nn.Module):
    def __init__(self, mesh_dir, links, K, init_op, refs, render_size, frame_ids, data_dir, dbg):
        super(Model, self).__init__()
        self.dbg = dbg
        for link in links:
            mesh = trimesh.load(osp.join(mesh_dir, f"link{link}.STL"))
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).long()
            self.register_buffer(f'vertices_{link}', vertices[None, :, :])
            self.register_buffer(f'faces_{link}', faces[None, :, :])
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
        renderer = nr.Renderer(image_size=render_size, orig_size=self.orig_size)  # todo: 64,128,256
        self.renderer = renderer

        # all_pose_eb_qpos=[]
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
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="render_based_solver",
                      auto_increase=True, enable=self.dbg)
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
                R = pose_ec[:3, :3]
                t = pose_ec[:3, 3]
                verts, faces = getattr(self, f"vertices_{link}"), getattr(self, f"faces_{link}")
                vis3d.add_mesh(utils_3d.transform_points(verts[0], pose_ec), faces[0], name=f"link{link}")
                si = self.renderer(verts, faces, mode='silhouettes', K=self.K[None], R=R[None], t=t[None])[0]
                si = interpolate(si[None, None], size=self.orig_size)[0, 0]
                si = si[:self.H, :self.W]
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            loss = torch.sum((all_link_si - getattr(self, f"image_ref_{frame_id}")) ** 2)
            losses.append(loss)
        evaltime('3')
        loss = torch.stack(losses).mean()
        return loss, all_frame_all_link_si


class ModelEyeOnHand(nn.Module):
    def __init__(self, mesh_dir, links, K, init_op, refs, render_size, frame_ids, data_dir, dbg):
        """

        @param mesh_dir:
        @param links:
        @param K:
        @param init_op: init transformation: TC_CtoE
        @param refs:
        @param render_size:
        @param frame_ids:
        @param data_dir:
        """
        super(ModelEyeOnHand, self).__init__()
        self.dbg = dbg
        for link in links:
            mesh = trimesh.load(osp.join(mesh_dir, f"link{link}.STL"))
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).long()
            self.register_buffer(f'vertices_{link}', vertices[None, :, :])
            self.register_buffer(f'faces_{link}', faces[None, :, :])
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
        renderer = nr.Renderer(image_size=render_size, orig_size=self.orig_size)  # todo: 64,128,256
        self.renderer = renderer

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
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="render_based_solver",
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
                R = Tc_c2l[:3, :3]
                t = Tc_c2l[:3, 3]
                verts, faces = getattr(self, f"vertices_{link}"), getattr(self, f"faces_{link}")
                vis3d.add_mesh(utils_3d.transform_points(verts[0], Tc_c2l), faces[0], name=f"link{link}")
                pyrender_mask = render_api.pyrender_render_mesh_api(
                    trimesh.Trimesh(to_array(verts[0]), to_array(faces[0])),
                    Tc_c2l, self.H, self.W, self.K.cpu().numpy())
                print(f'link {link}, {np.unique(pyrender_mask)}')
                vis3d.add_image(pyrender_mask, name=f'pyrender_mask{link}')
                si = self.renderer(verts, faces, mode='silhouettes', K=self.K[None], R=R[None], t=t[None])[0]
                si = interpolate(si[None, None], size=self.orig_size)[0, 0]
                si = si[:self.H, :self.W]
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            loss = torch.sum((all_link_si - getattr(self, f"image_ref_{frame_id}")) ** 2)
            losses.append(loss)
        evaltime('3')
        loss = torch.stack(losses).mean()
        return loss, all_frame_all_link_si


class RenderBasedSolver:
    def __init__(self, cfg):
        self.cfg = cfg.rb_solve
        self.data_dir = self.cfg.data_dir
        self.eye_on_hand = self.cfg.eye_on_hand
        self.init_Tbc = self.cfg.init_Tbc
        self.init_Tc_c2e = self.cfg.init_Tc_c2e

        self.masks = self.load_mask()
        self.K = np.loadtxt(osp.join(self.data_dir, "K.txt"))
        self.mesh_dir = "data/xarm_description/meshes/xarm7/visual"
        self.links = self.cfg.links
        self.test_links = [0, 1, 2, 3, 4, 5, 6, 7]
        self.solved_op = None
        self.output_dir = cfg.output_dir

    def solve(self):
        render_sizes = self.cfg.render_sizes
        steps = self.cfg.steps
        lrs = self.cfg.lrs
        assert len(render_sizes) == len(steps)
        tb_writer = SummaryWriter(self.output_dir, flush_secs=20)
        for render_size, step, lr in zip(render_sizes, steps, lrs):
            if not self.eye_on_hand:
                init_Tbc = self.init_Tbc
                model = Model(self.mesh_dir, self.links, self.K, init_Tbc, self.masks, render_size,
                              list(self.masks.keys()), self.data_dir, dbg=self.cfg.dbg)
            else:
                init_Tc_c2e = self.init_Tc_c2e
                model = ModelEyeOnHand(self.mesh_dir, self.links, self.K, init_Tc_c2e, self.masks, render_size,
                                       list(self.masks.keys()), self.data_dir, self.cfg.dbg)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loop = tqdm.tqdm(range(step))
            plt_utils.image_grid(list(self.masks.values()), show=False)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            tb_writer.add_image("gt", np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1))
            for i in loop:
                optimizer.zero_grad()
                loss, si = model()
                tb_writer.add_scalar(f"loss_render_size_{render_size}", loss.item(), i)
                loss.backward()
                optimizer.step()
                if i % self.cfg.log_interval == 0:
                    plt_utils.image_grid(torch.stack(si), show=False)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format="png")
                    tb_writer.add_image(f"optim_{render_size}",
                                        np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), i)
                    op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
                    loguru.logger.info(f"step {i} {np.array2string(to_array(op), separator=',')}")
                    np.savetxt(osp.join(self.output_dir, "result.txt"), to_array(op))
                loop.set_description('Optimizing (loss %.4f)' % loss.data)
                if loss.item() < 500:
                    break
            op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
            if not self.eye_on_hand:
                init_Tbc = op
            else:
                init_Tc_c2e = op
            self.solved_op = op
        return to_array(self.solved_op)

    def load_mask(self):
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
        rgb_paths = sorted(glob.glob(osp.join(data_dir, "rgb*.png")))
        depth_paths = sorted(glob.glob(osp.join(data_dir, "depth*.png")))
        pose_eb_paths = sorted(glob.glob(osp.join(data_dir, "pose_eb*.txt")))
        K = np.loadtxt(osp.join(data_dir, "K.txt"))
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        test_link_meshes = {}
        for link in self.test_links:
            mesh = trimesh.load(osp.join(self.mesh_dir, f"link{link}.STL"))
            test_link_meshes[link] = mesh
        for i in tqdm.trange(len(rgb_paths)):
            vis3d.set_scene_id(i)
            rgb = imageio.imread(rgb_paths[i])
            H, W = rgb.shape[:2]
            depth = cv2.imread(depth_paths[i], 2).astype(np.float32) / 1000.0
            pose_eb_qpos = np.loadtxt(pose_eb_paths[i])

            pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
            vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=2)
            pose_bc = self.solved_op
            all_rendered_mask = []
            for link in self.test_links:
                pq = sk.compute_forward_kinematics(pose_eb_qpos, link + 1)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.matrix_3x4_to_4x4(np.concatenate([R, t[:, None]], axis=-1))
                pose_eb = torch.from_numpy(pose_eb).cuda().float()
                pose_ec = to_array(pose_bc @ pose_eb)
                mesh = test_link_meshes[link]
                # verts, faces = getattr(self.model, f"vertices_{link}"), getattr(self.model, f"faces_{link}")
                vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, pose_ec), mesh.faces, name=f"link{link}")
                rendered_mask = render_api.pyrender_render_mesh_api(mesh, pose_ec, H, W, K)
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

    def evaluate(self):
        if osp.exists(osp.join(self.data_dir, "campose.txt")):
            gt = np.loadtxt(osp.join(self.data_dir, "campose.txt"))
            print("error", gt - to_array(self.solved_op))
        else:
            loguru.logger.info("no gt found.")


def make_gif(filename):
    with imageio.get_writer(filename, mode='I', fps=10) as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()
