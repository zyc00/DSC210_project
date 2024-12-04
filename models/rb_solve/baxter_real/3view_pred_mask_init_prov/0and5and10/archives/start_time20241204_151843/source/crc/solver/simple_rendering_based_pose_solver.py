import imageio
import os.path as osp

import cv2
import numpy as np
import io

import matplotlib.pyplot as plt
import torch
import tqdm
import trimesh
from PIL import Image
from dl_ext.timer import EvalTime
from pytorch3d.transforms import se3_log_map, se3_exp_map
from tensorboardX import SummaryWriter
from torch import nn
import neural_renderer as nr

from crc.modeling.misc import interpolate
from crc.utils import utils_3d, render_api, plt_utils
from crc.utils.pn_utils import to_array
from crc.utils.vis3d_ext import Vis3D


class Model(nn.Module):
    def __init__(self, mesh, K, init_object_pose, mask_gt, render_size):
        super().__init__()
        # mesh = trimesh.load_mesh(mesh_path)
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        self.register_buffer(f'vertices', vertices[None, :, :])
        self.register_buffer(f'faces', faces[None, :, :])
        self.K = torch.as_tensor(K).float().cuda()
        init_dof = se3_log_map(torch.as_tensor(init_object_pose, dtype=torch.float32)[None].permute(0, 2, 1))[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        self.H, self.W = mask_gt.shape
        self.mask_gt = torch.as_tensor(mask_gt).cuda().float()
        self.orig_size = max(self.H, self.W)
        renderer = nr.Renderer(image_size=render_size, orig_size=self.orig_size)  # todo: 64,128,256
        self.renderer = renderer

    def forward(self):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="simple_render_based_solver",
                      auto_increase=True, enable=False)
        evaltime = EvalTime(disable=True)
        evaltime('')
        op = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        # losses = []
        R = op[:3, :3]
        t = op[:3, 3]
        verts, faces = self.vertices, self.faces
        # vis3d.add_mesh(utils_3d.transform_points(verts[0], pose_ec), faces[0], name=f"link{link}")
        si = self.renderer(verts, faces, mode='silhouettes', K=self.K[None], R=R[None], t=t[None])[0]
        si = interpolate(si[None, None], size=self.orig_size)[0, 0]
        si = si[:self.H, :self.W]
        # all_link_si.append(si)
        #     all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
        #     all_frame_all_link_si.append(all_link_si)
        loss = torch.sum((si - self.mask_gt) ** 2)
        # losses.append(loss)
        # evaltime('3')
        # loss = torch.stack(losses).mean()
        return loss, si


class RenderBasedSolver:
    def __init__(self, mesh, mask_gt, init_op, K, log_dir,
                 render_sizes, iterations, lrs):
        # cfg = cfg.rb_solve
        self.render_sizes = render_sizes
        self.iterations = iterations
        self.lrs = lrs
        self.output_dir = log_dir
        self.init_op = torch.as_tensor(init_op)
        self.solved_op = None
        self.mesh = mesh
        self.mask_gt = mask_gt
        self.K = K

    def solve(self):
        render_sizes = self.render_sizes
        steps = self.iterations
        lrs = self.lrs
        assert len(render_sizes) == len(steps) == len(lrs)
        tb_writer = SummaryWriter(self.output_dir, flush_secs=20)
        init_op = self.init_op
        for render_size, step, lr in zip(render_sizes, steps, lrs):
            model = Model(self.mesh, self.K, self.init_op, self.mask_gt, render_size)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loop = tqdm.tqdm(range(step))
            plt.imshow(self.mask_gt)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            tb_writer.add_image("gt", np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1))
            for i in loop:
                optimizer.zero_grad()
                loss, si = model()
                tb_writer.add_scalar(f"loss_render_size_{render_size}", loss.item(), i)
                loss.backward()
                optimizer.step()
                # image = si.detach().cpu().numpy()
                if i % 20 == 0:
                    plt.imshow(to_array(si))
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format="png")
                    tb_writer.add_image(f"optim_{render_size}",
                                        np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), i)
                    op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
                    print(f"step {i} {np.array2string(to_array(op), separator=',')}")
                    np.savetxt(osp.join(self.output_dir, "result.txt"), to_array(op))
                loop.set_description('Optimizing (loss %.4f)' % loss.data)
                if loss.item() < 500:
                    break
            op = se3_exp_map(model.dof[None]).permute(0, 2, 1)[0]
            self.solved_op = op
        return to_array(self.solved_op)

    def validate(self, rgb):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg",
                      sequence_name="simple_render_based_solver_validate", auto_increase=True, enable=True)
        K = self.K
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        mesh = self.mesh
        H, W = rgb.shape[:2]
        op = to_array(self.solved_op)
        vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, op), mesh.faces)
        rendered_mask = render_api.pyrender_render_mesh_api(mesh, op, H, W, K)
        tmp = plt_utils.vis_mask(rgb, rendered_mask.astype(np.uint8), [255, 0, 0])
        vis3d.add_image(tmp, name='optim')


def main():
    mesh = trimesh.primitives.Box(extents=[0.0635, 0.127, 0.04445])
    mask_gt = imageio.imread("data/realsense/20230205_161717/mask_000000.png")[:, :, 0] > 0
    init_op = np.eye(4)
    init_op[:3, 3] = np.array([0, 0, 0.4])
    K = np.loadtxt("data/realsense/20230205_161717/K.txt")

    solver = RenderBasedSolver(mesh, mask_gt, init_op, K, "models/20230205_161717",
                               [128], [500], [0.03])
    solver.solve()
    solver.validate(imageio.imread("data/realsense/20230205_161717/rgb_000000.png"))


if __name__ == '__main__':
    main()
