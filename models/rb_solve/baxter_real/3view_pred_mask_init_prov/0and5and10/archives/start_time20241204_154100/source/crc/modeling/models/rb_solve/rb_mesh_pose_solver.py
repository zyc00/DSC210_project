import os.path as osp

import loguru
import numpy as np
import torch
import torch.nn as nn
import trimesh

from crc.structures.nvdiffrast_renderer import NVDiffrastRenderer
from crc.utils import utils_3d
from crc.utils.utils_3d import se3_log_map, se3_exp_map
from crc.utils.vis3d_ext import Vis3D


class RBMeshPoseSolver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.rbsolver

        self.dbg = self.total_cfg.dbg
        mesh_paths = self.cfg.mesh_paths
        self.mesh_vertices = {}
        for link_idx, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(osp.expanduser(mesh_path))
            vertices: torch.Tensor = torch.from_numpy(mesh.vertices).float()
            # vertices.requires_grad = True
            faces = torch.from_numpy(mesh.faces).int()
            # self.register_buffer(f'vertices_{link_idx}', vertices)
            # vertices = nn.Parameter(vertices, requires_grad=True)
            # self.mesh_vertices[f'vertices_{link_idx}'] = vertices
            setattr(self, f'vertices_{link_idx}', nn.Parameter(vertices, requires_grad=True))
            self.register_buffer(f'faces_{link_idx}', faces)
        self.nlinks = len(mesh_paths)
        # camera parameters
        if not self.cfg.eye_in_hand:
            init_Tc_c2b = self.cfg.init_Tc_c2b
            init_dof = se3_log_map(torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                                   backend="opencv")[0]
        else:
            init_Tc_c2e = self.cfg.init_Tc_c2e
            init_dof = se3_log_map(torch.as_tensor(init_Tc_c2e, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                                   backend="opencv")[0]
        self.dof = nn.Parameter(init_dof, requires_grad=False)
        # setup renderer
        self.H, self.W = self.cfg.H, self.cfg.W
        self.renderer = NVDiffrastRenderer([self.H, self.W])

        self.register_buffer(f'history_ops', torch.zeros(10000, 6))
        self.register_buffer(f'history_losses', torch.full((10000,), fill_value=1000000, dtype=torch.float))

    def forward(self, dps):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="rbmeshposesolver_forward",
                      auto_increase=True, enable=self.dbg)
        assert dps['global_step'] == 0
        put_id = (self.history_ops == 0).all(dim=1).nonzero()[0, 0].item()
        self.history_ops[put_id] = self.dof.detach()
        all_frame_all_link_si = []
        if self.cfg.use_mask == "gt":
            loguru.logger.info("using gt mask!")
            masks_ref = dps['mask']
        elif self.cfg.use_mask == "pred":
            masks_ref = dps['mask_pred']
        elif self.cfg.use_mask == "anno":
            masks_ref = dps['mask_anno']
        elif self.cfg.use_mask == 'sam':
            masks_ref = dps['mask_sam']
        link_poses = dps['link_poses']
        # assert len(self.links) == link_poses.shape[1]
        K = dps['K'][0]

        batch_size = masks_ref.shape[0]
        losses = []
        if not self.cfg.eye_in_hand:
            Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
            for bid in range(batch_size):
                all_link_si = []
                for link_idx in range(self.nlinks):
                    Tc_c2l = Tc_c2b @ link_poses[bid, link_idx]
                    verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                    vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                    si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                    all_link_si.append(si)
                all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
                all_frame_all_link_si.append(all_link_si)
                if self.cfg.loss_type == 'mse':
                    loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
                elif self.cfg.loss_type == 'huber':
                    loss = torch.nn.functional.huber_loss(all_link_si, masks_ref[bid].float(), reduction='sum')
                else:
                    raise NotImplementedError()
                losses.append(loss)
        else:
            Tc_c2e = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
            for bid in range(batch_size):
                all_link_si = []
                for link_idx in range(self.nlinks):
                    Tc_c2l = Tc_c2e @ link_poses[bid, 7].inverse() @ link_poses[bid, link_idx]
                    # todo assert end effector is the last link, and camera is attached to the end effector
                    verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                    vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                    si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                    all_link_si.append(si)
                all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
                all_frame_all_link_si.append(all_link_si)
                if self.cfg.loss_type == 'mse':
                    loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
                elif self.cfg.loss_type == 'huber':
                    loss = torch.nn.functional.huber_loss(all_link_si, masks_ref[bid].float(), reduction='sum')
                else:
                    raise NotImplementedError()
                losses.append(loss)
        if self.cfg.weighted_loss:
            loss = torch.stack(losses)
            loss_weight = masks_ref.sum(1).sum(1) > 0
            loss = (loss * loss_weight.float()).sum() / (loss_weight.float().sum() + 1e-5)
        else:
            loss = torch.stack(losses).mean()
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)
        output = {"rendered_masks": all_frame_all_link_si,
                  "ref_masks": masks_ref,
                  "error_maps": (all_frame_all_link_si - masks_ref.float()).abs(),
                  }
        # metrics
        loss_dict = {"mask_loss": loss}
        self.history_losses[put_id] = loss.detach()

        if not self.cfg.eye_in_hand:
            gt_Tc_c2b = dps['Tc_c2b'][0]
            if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
                gt_dof6 = utils_3d.se3_log_map(gt_Tc_c2b[None].permute(0, 2, 1), backend='opencv')[0]
                if self.cfg.use_last_as_result:
                    trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                    rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180
                    R_err, t_err = utils_3d.pose_distance(se3_exp_map(self.dof[None]).permute(0, 2, 1)[0], gt_Tc_c2b)
                    R_err = R_err / np.pi * 180
                    t_err = t_err * 100
                else:
                    min_loss_index = self.history_losses.argmin().item()
                    dof = self.history_ops[min_loss_index]
                    trans_err = ((gt_dof6[:3] - dof[:3]) * 100).abs()
                    rot_err = (gt_dof6[3:] - dof[3:]).abs().max() / np.pi * 180
            else:
                trans_err = torch.zeros(3)
                rot_err = torch.zeros(1)
                R_err = torch.zeros(1)
                t_err = torch.zeros(1)
        else:
            gt_Tc_c2e = dps['Tc_c2e'][0]
            if not torch.allclose(gt_Tc_c2e, torch.eye(4).to(gt_Tc_c2e.device)):
                gt_dof6 = utils_3d.se3_log_map(gt_Tc_c2e[None].permute(0, 2, 1), backend='opencv')[0]
                if self.cfg.use_last_as_result:
                    trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                    rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180
                    R_err, t_err = utils_3d.pose_distance(se3_exp_map(self.dof[None]).permute(0, 2, 1)[0], gt_Tc_c2e)
                    R_err = R_err / np.pi * 180
                    t_err = t_err * 100
                else:
                    min_loss_index = self.history_losses.argmin().item()
                    dof = self.history_ops[min_loss_index]
                    trans_err = ((gt_dof6[:3] - dof[:3]) * 100).abs()
                    rot_err = (gt_dof6[3:] - dof[3:]).abs().max() / np.pi * 180
                    # R_err, t_err = utils_3d.pose_distance(gt_Tc_c2e[None].permute(0, 2, 1), se3_exp_map(self.dof[None]).permute(0, 2, 1))
            else:
                trans_err = torch.zeros(3)
                rot_err = torch.zeros(1)
                R_err = torch.zeros(1)
                t_err = torch.zeros(1)
        metrics = {
            "err_x": trans_err[0],
            "err_y": trans_err[1],
            "err_z": trans_err[2],
            "err_trans": trans_err.norm(),
            "err_rot": rot_err,
            'err_R': R_err,
            'err_t': t_err,
        }
        output["metrics"] = metrics

        return output, loss_dict
