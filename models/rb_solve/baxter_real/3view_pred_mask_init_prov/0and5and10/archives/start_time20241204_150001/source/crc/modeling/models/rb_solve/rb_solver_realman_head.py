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


class RBSolverRealmanHead(nn.Module):
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
        if not self.cfg.eye_in_hand:
            init_Tc_c2b = self.cfg.init_Tc_c2b
            init_dof = se3_log_map(torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                                   backend="opencv")[0]
        else:
            init_Tc_c2e = self.cfg.init_Tc_c2e
            init_dof = se3_log_map(torch.as_tensor(init_Tc_c2e, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                                   backend="opencv")[0]
        if self.cfg.optim_inverse is True:
            tmp = se3_exp_map(init_dof[None]).permute(0, 2, 1)[0]
            tmp = tmp.inverse()
            init_dof = se3_log_map(tmp[None].permute(0, 2, 1), eps=1e-5, backend='opencv')[0]
        if self.cfg.optim_trans_only is False:
            self.dof = nn.Parameter(init_dof, requires_grad=True)
        else:
            self.dof = nn.Parameter(init_dof[:3], requires_grad=True)
            self.register_buffer('fix_dof', init_dof[3:])
        if self.cfg.optim_eep:
            if self.cfg.optim_eep_mask != []:
                if self.cfg.eep_init != []:
                    self.eep = nn.Parameter(torch.tensor(self.cfg.eep_init, dtype=torch.float32), requires_grad=True)
                else:
                    noptim = len(self.cfg.optim_eep_mask)
                    self.eep = nn.Parameter(torch.zeros(noptim, dtype=torch.float32), requires_grad=True)
            else:
                raise DeprecationWarning()
                if self.cfg.optim_eep_xy:
                    self.eep = nn.Parameter(torch.tensor([0, 0, 0.005, 0, 0, 0], dtype=torch.float32),
                                            requires_grad=True)
                else:
                    if self.cfg.optim_eep_rot:
                        self.eep = nn.Parameter(torch.tensor([0.005, 0, 0, 0], dtype=torch.float32), requires_grad=True)
                    else:
                        self.eep = nn.Parameter(torch.tensor([0.005], dtype=torch.float32), requires_grad=True)
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
        Tb_b2ls = dps['link_poses']
        K = dps['K'][0]

        batch_size = masks_ref.shape[0]
        losses = []
        if not self.cfg.eye_in_hand:
            if self.cfg.optim_inverse is False:
                if self.cfg.optim_trans_only is False:
                    Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
                else:
                    dof = torch.cat([self.dof, self.fix_dof], dim=0)
                    Tc_c2b = se3_exp_map(dof[None]).permute(0, 2, 1)[0]
            else:
                Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0].inverse()
            for bid in range(batch_size):
                all_link_si = []
                # relative_to_link_index=self.cfg.relative_to_link_index
                for link_idx in range(self.nlinks):
                    Tc_c2l = Tc_c2b @ Tb_b2ls[bid, link_idx]
                    verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                    if self.cfg.optim_mesh_scale is True:
                        verts = verts * getattr(self, f'mesh_scale_{link_idx}')[None]
                    # vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                    vis3d.add_mesh(utils_3d.transform_points(verts, Tb_b2ls[bid, link_idx]), faces,
                                   name=f"link{link_idx}")
                    si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                    all_link_si.append(si)
                vis3d.add_camera_pose(Tc_c2b.inverse(), name='cam_pose')
                vis3d.add_realman(dps['qpos'][bid])
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
        else:
            assert self.cfg.optim_inverse is False
            Tc_c2e = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
            # else:
            #     raise NotImplementedError()
            for bid in range(batch_size):
                all_link_si = []
                for link_idx in range(self.nlinks):
                    Tc_c2l = Tc_c2e @ Tb_b2ls[bid, 7].inverse() @ Tb_b2ls[bid, link_idx]
                    # todo assert end effector is the last link, and camera is attached to the end effector
                    verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                    if self.cfg.optim_eep and link_idx > 7:  # magic number for xarm7
                        Tb_b2e = Tb_b2ls[bid, 7]
                        Te_e2l = Tb_b2e.inverse() @ Tb_b2ls[bid, link_idx]
                        if self.cfg.optim_eep_mask != []:
                            eep = torch.zeros(6, device=self.eep.device, dtype=torch.float)
                            eep[self.cfg.optim_eep_mask] = self.eep
                            Te_e2ep = se3_exp_map(eep[None]).permute(0, 2, 1)[0]
                            if self.cfg.eep_rotz != 0:
                                R = utils_3d.Rt_to_pose(utils_3d.rotz_np(np.deg2rad(self.cfg.eep_rotz))[0])
                                R = torch.from_numpy(R).float().cuda()
                                Te_e2ep[:3, :3] = R[:3, :3]
                            if self.cfg.eep_tz != 0:
                                Te_e2ep[2, 3] = self.cfg.eep_tz
                        else:
                            raise DeprecationWarning()
                            if self.cfg.optim_eep_xy:
                                Te_e2ep = se3_exp_map(self.eep[None]).permute(0, 2, 1)[0]
                            else:
                                if self.cfg.optim_eep_rot:
                                    eep = torch.cat([torch.zeros(2, device=self.eep.device, dtype=torch.float),
                                                     self.eep], dim=0)
                                    Te_e2ep = se3_exp_map(eep[None]).permute(0, 2, 1)[0]
                                else:
                                    eep = torch.cat([torch.zeros(2, device=self.eep.device, dtype=torch.float),
                                                     self.eep,
                                                     torch.zeros(3, device=self.eep.device, dtype=torch.float), ],
                                                    dim=0)
                                    Te_e2ep = se3_exp_map(eep[None]).permute(0, 2, 1)[0]
                        Tb_b2l = Tb_b2e @ Te_e2ep @ Te_e2l
                        Tc_c2l = Tc_c2e @ Tb_b2ls[bid, 7].inverse() @ Tb_b2l
                    vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                    si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                    all_link_si.append(si)
                all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
                all_frame_all_link_si.append(all_link_si)
                if self.cfg.loss_type == 'mse':
                    if self.cfg.ignore_gripper_mask is False:
                        loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
                    else:
                        loss_map = all_link_si - masks_ref[bid].float()
                        loss_map = loss_map * (1 - dps['mask_gripper'][bid].float())
                        loss = torch.sum(loss_map ** 2)
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
        if self.cfg.eye_in_hand is True and self.cfg.ignore_gripper_mask is True:
            loss_map = all_frame_all_link_si - masks_ref.float()
            loss_map = loss_map * (1 - dps['mask_gripper'].float())
            output['error_maps'] = loss_map.abs()
        # metrics
        loss_dict = {"mask_loss": loss}
        if self.cfg.optim_eep:
            if self.cfg.reg_loss_weight > 0:
                if self.cfg.optim_eep_xy:
                    reg_loss = self.cfg.reg_loss_weight * (
                            self.eep - torch.tensor([0, 0, 0.005, 0, 0, 0], device='cuda')).norm()
                else:
                    reg_loss = self.cfg.reg_loss_weight * (
                            self.eep - torch.tensor([0.005, 0, 0, 0], device='cuda')).norm()
                loss_dict['reg_loss'] = reg_loss
        self.history_losses[put_id] = loss.detach()

        if not self.cfg.eye_in_hand:
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
        tsfm = utils_3d.se3_exp_map(self.dof[None].detach().cpu()).permute(0, 2, 1)[0]
        output['tsfm'] = tsfm

        return output, loss_dict
