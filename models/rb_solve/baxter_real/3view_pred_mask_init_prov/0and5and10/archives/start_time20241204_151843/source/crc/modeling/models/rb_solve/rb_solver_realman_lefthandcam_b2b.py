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


class RBSolverRealmanLeftHandCamB2B(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.rbsolver

        self.dbg = self.total_cfg.dbg
        mesh_paths = self.cfg.mesh_paths
        for link_idx, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(osp.expanduser(mesh_path))
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f'vertices_{link_idx}', vertices)
            self.register_buffer(f'faces_{link_idx}', faces)
            if self.cfg.optim_mesh_scale is True:
                setattr(self, f'mesh_scale_{link_idx}',
                        nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True))
        self.nlinks = len(mesh_paths)
        # camera parameters
        init_Tc_c2b = self.cfg.init_Tc_c2b
        init_dof = se3_log_map(torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                               backend="opencv")[0]
        if self.cfg.optim_trans_only is False:
            self.dof = nn.Parameter(init_dof, requires_grad=True)
        else:
            self.dof = nn.Parameter(init_dof[:3], requires_grad=True)
            self.register_buffer('fix_dof', init_dof[3:])
        # version 1 delta modelling
        # self.b2b = nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32), requires_grad=True)
        # version 2 Trb_rb2lb modelling
        # init = torch.Tensor([[1, 0, 0, 0],
        #                      [0, 1, 0, 0],
        #                      [0, 0, 1, -0.183],
        #                      [0, 0, 0, 1]])
        # b2b = se3_log_map(torch.as_tensor(init, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
        #                   backend="opencv")[0]
        # optim only z
        # b2b = torch.tensor(-0.183, dtype=torch.float32)
        # optim xyz
        # b2b = torch.tensor([0, 0, -0.183], dtype=torch.float32)
        # self.b2b = nn.Parameter(b2b, requires_grad=True)
        # optim xy
        b2b = torch.tensor([0, 0], dtype=torch.float32)
        self.b2b = nn.Parameter(b2b, requires_grad=True)
        # setup renderer
        self.H, self.W = self.cfg.H, self.cfg.W
        self.renderer = NVDiffrastRenderer([self.H, self.W], render_scale=self.cfg.render_scale)

        self.register_buffer(f'history_ops', torch.zeros(10000, 6))
        self.register_buffer(f'history_losses', torch.full((10000,), fill_value=1000000, dtype=torch.float))

    def forward(self, dps):
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="rbsolver_realman_head_forward",
                      auto_increase=True, enable=self.dbg)
        assert dps['global_step'] == 0
        put_id = (self.history_ops == 0).all(dim=1).nonzero()[0, 0].item()
        if self.cfg.optim_trans_only:
            dof = torch.cat([self.dof.detach(), self.fix_dof], dim=0)
            self.history_ops[put_id] = dof
        else:
            self.history_ops[put_id] = self.dof.detach()
        all_frame_all_link_si = []
        if self.cfg.use_mask == "gt":
            loguru.logger.info("using gt mask!")
            masks_ref = dps['mask']
        elif self.cfg.use_mask == "pred":
            masks_ref = dps['mask_pred']
        elif self.cfg.use_mask == "anno":
            masks_ref = dps['mask_anno']
        elif self.cfg.use_mask == 'mask_gsam':
            masks_ref = dps['mask_gsam']
        # Tb_b2ls = dps['link_poses']
        qposes = dps['qpos']
        K = dps['K'][0]

        batch_size = masks_ref.shape[0]
        losses = []
        Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        # optim matrix
        # B2B = se3_exp_map(self.b2b[None]).permute(0, 2, 1)[0]
        # optim z
        # B2B = torch.eye(4, dtype=torch.float32, device='cuda')
        # B2B[2, 3] += self.b2b
        # optim xyz
        # B2B = torch.eye(4, dtype=torch.float32, device='cuda')
        # B2B[:3, 3] += self.b2b
        # optim xy
        B2B = torch.eye(4, dtype=torch.float32, device='cuda')
        B2B[:2, 3] += self.b2b
        B2B[2, 3] = -0.183
        #     lb:3, rb:4, le:16, re:17
        for bid in range(batch_size):
            all_link_si = []
            qpos = qposes[bid]
            Tbase_base2le = torch.from_numpy(vis3d.realman_sk.compute_forward_kinematics(
                qpos.cpu().numpy(), 16).to_transformation_matrix()).float().cuda()
            Tbase_base2rb = torch.from_numpy(vis3d.realman_sk.compute_forward_kinematics(
                qpos.cpu().numpy(), 4).to_transformation_matrix()).float().cuda()
            Tbase_base2lb = torch.from_numpy(vis3d.realman_sk.compute_forward_kinematics(
                qpos.cpu().numpy(), 3).to_transformation_matrix()).float().cuda()

            for link_idx in range(self.nlinks):
                link_id = self.total_cfg.dataset.realman.use_links[link_idx]
                Tbase_base2l = torch.from_numpy(vis3d.realman_sk.compute_forward_kinematics(
                    qpos.cpu().numpy(), link_id).to_transformation_matrix()).float().cuda()
                # version1
                # Tle_le2l = Tbase_base2le.inverse() @ Tbase_base2rb @ B2B @ Tbase_base2rb.inverse() @ Tbase_base2l
                # version2
                Tle_le2l = Tbase_base2le.inverse() @ Tbase_base2rb @ B2B @ Tbase_base2lb.inverse() @ Tbase_base2l
                Tc_c2l = Tc_c2b @ Tle_le2l  # Tc_c2b is actually Tc_c2le
                verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                all_link_si.append(si)
            # vis3d.add_camera_pose(Tc_c2b.inverse(), name='cam_pose')
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            if self.cfg.loss_type == 'mse':
                if self.cfg.last2link_loss_weights == 1:
                    loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
                else:
                    x1, y1, x2, y2 = dps['link78regions'][0, bid]  #
                    unreduced_loss = (all_link_si - masks_ref[bid].float()) ** 2
                    unreduced_loss[y1:y2, x1:x2] = unreduced_loss[y1:y2, x1:x2] * self.cfg.last2link_loss_weights
                    loss = unreduced_loss.sum()
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

        gt_Tc_c2b = dps['Tc_c2b'][0]
        if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
            gt_dof6 = utils_3d.se3_log_map(gt_Tc_c2b[None].permute(0, 2, 1), backend='opencv')[0]
            if self.cfg.use_last_as_result:
                if self.cfg.optim_trans_only is False:
                    trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                    rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180
                    R_err, t_err = utils_3d.pose_distance(se3_exp_map(self.dof[None]).permute(0, 2, 1)[0],
                                                          gt_Tc_c2b)
                else:
                    trans_err = ((gt_dof6[:3] - self.dof) * 100).abs()
                    rot_err = (gt_dof6[3:] - self.fix_dof).abs().max() / np.pi * 180
                    dof = torch.cat([self.dof, self.fix_dof], dim=0)
                    R_err, t_err = utils_3d.pose_distance(se3_exp_map(dof[None]).permute(0, 2, 1)[0], gt_Tc_c2b)
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
