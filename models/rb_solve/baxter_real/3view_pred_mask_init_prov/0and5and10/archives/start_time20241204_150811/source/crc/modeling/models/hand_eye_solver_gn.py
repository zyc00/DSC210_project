import glob
import io
import os.path as osp

import cv2
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import transforms3d
import trimesh
from PIL import Image
from torch import nn

from crc.utils import utils_3d, render_api, plt_utils
from crc.utils.utils_3d import Rt_to_pose, pose_distance
from crc.utils.vis3d_ext import Vis3D

VELOCITY_THRESHOLD = 0.02


class HandEyeSolverGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.hand_eye_solver_gn
        self.dbg = cfg.dbg is True
        self.cnt = 0
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, dps):
        K = dps['K'][0]
        Tc_c2ms = self.calculate_marker_pose(dps)
        if self.cfg.use_gt_marker_pose:
            loguru.logger.warning("Use GT marker pose!!!")
            Tc_c2ms = dps['Tc_c2m'].cpu().numpy()
        ref_Tc_c2e = dps['Tc_c2e'][0].cpu().numpy()
        Te_e2c, data_num = self.calculate_and_show(Tc_c2ms, dps, ref_Tc_c2e)
        self.vis_base_origin_in_cam(Te_e2c, dps)
        output = {'Te_e2c': torch.from_numpy(Te_e2c).cuda().float(),
                  'data_num': torch.tensor(data_num).cuda().float()}
        return output, {}

    def calculate_marker_pose(self, dps):
        imgs = dps['rgb']  # T,H,W,3
        # rgb_paths = sorted(glob.glob(osp.join(data_dir, "color/*.png")))
        # depth_paths = sorted(glob.glob(osp.join(data_dir, "depth/*.png")))
        # assert int(self.cfg.icp) + int(self.cfg.cpa) < 2
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="hesolve_cal_marker_pose",
                      auto_increase=True, enable=self.dbg)
        mesh = trimesh.primitives.Box(extents=np.array([0.04, 0.05, 0.01]))
        mesh = trimesh.Trimesh(mesh.vertices + np.array([[0.01, 0.015, 0]]), mesh.faces)
        model = {'pts': mesh.vertices * 1000.0, 'faces': mesh.faces}

        Tc_c2ms = []
        # reproj_errors = []
        manual_set_zero = self.cfg.manual_set_zero

        for i in tqdm.trange(imgs.shape[0]):
            if i in manual_set_zero:
                Tc_c2ms.append(np.zeros((4, 4)))
                continue
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((3 * 4, 3), np.float32)
            # 4.9cm grid
            # objp[:, :2] = np.mgrid[0:3, 0:4].T.reshape(-1, 2) / 100.0 * 4.9
            objp[:, :2] = np.mgrid[0:3, 0:4].T.reshape(-1, 2) / 100.0 * 5
            axis = np.float32([[4, 0, 0], [0, 3, 0], [0, 0, 0]]).reshape(-1, 3) / 100.0
            # img = cv2.imread(fname)info
            img = imgs[i].cpu().numpy()
            # cv2.imwrite("image.png", img)
            H, W, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCornersSB(gray, (3, 4), None)

            K = dps['K'][0].cpu().numpy()
            if ret is False:
                Tc_c2ms.append(np.zeros((4, 4)))
                loguru.logger.warning(f"Cannot find chessboard in {i}")
                continue
            if self.cfg.subpixel:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                # print(corners)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners, K, cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvecs)
            Tc_c2m = utils_3d.Rt_to_pose(R, tvecs.reshape(3))
            rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh, Tc_c2m, H, W, K)
            vis3d.add_image(plt_utils.vis_mask(img, rendered_mask.astype(np.uint8), [255, 0, 0]), name='chessboard')
            pts_cam = utils_3d.transform_points(axis, Tc_c2m)

            plt.figure(dpi=200)
            plt.imshow(img)
            plt.scatter(corners[:, 0, 0], corners[:, 0, 1], marker='.')
            for tmpi, c in enumerate(corners):
                plt.text(corners[tmpi, 0, 0], corners[tmpi, 0, 1], str(tmpi), fontsize=10, fontdict={'color': 'red'})
            axis_pt_img = utils_3d.rect_to_img(K, utils_3d.transform_points(axis, Tc_c2m))
            plt.plot(axis_pt_img[[0, 2], 0], axis_pt_img[[0, 2], 1], 'r-')
            plt.plot(axis_pt_img[[1, 2], 0], axis_pt_img[[1, 2], 1], 'b-')
            # img_buf = io.BytesIO()
            plt.savefig(osp.join(self.total_cfg.sim_hec_eye_in_hand.outdir, f"{self.cnt}_{i}.png"), format="png")
            # vis3d.add_image(Image.open(img_buf), name="corners")
            plt.close("all")

            depth_image = dps['depth'][i].cpu().numpy()

            pts_rect = utils_3d.depth_to_rect(K[0, 0], K[1, 1], K[0, 2], K[1, 2], depth_image)
            vis3d.add_point_cloud(pts_rect, img.reshape(-1, 3), max_z=2)
            vis3d.add_spheres(pts_cam, 0.01, name='axis')
            vis3d.add_mesh(utils_3d.transform_points(model['pts'] / 1000.0, Tc_c2m), model['faces'], name="box")

            vis3d.increase_scene_id()
            # outfile = fname.replace("rgb_", "pose_cm_").replace(".png", ".txt")
            # np.savetxt(outfile, pose)
            Tc_c2ms.append(Tc_c2m)
        self.cnt += 1
        Tc_c2ms = np.stack(Tc_c2ms)
        # reproj_errors = np.array(reproj_errors)
        # loguru.logger.info(f"Reprojection Error {reproj_errors}")
        return Tc_c2ms

    def calculate_and_show(self, Tc_c2ms, dps, ref_Tc_c2e=None):
        Tb_b2es = dps['link_poses'].cpu().numpy()
        keep = (Tc_c2ms != 0).sum(1).sum(1) > 0
        Tb_b2es = Tb_b2es[keep]
        Tc_c2ms = Tc_c2ms[keep]
        data_num = Tc_c2ms.shape[0]
        loguru.logger.info("Data Number: {}".format(data_num))
        if data_num < 4:
            return np.eye(4), data_num

        ee_pose = Tb_b2es.transpose([1, 2, 0])
        marker_pose = Tc_c2ms.transpose([1, 2, 0])
        Te_e2c, res = self.__calculate(ee_pose, marker_pose)

        loguru.logger.info("*" * 10 + " Result " + "*" * 10)
        loguru.logger.info("The transformation Tc_c2e:")
        loguru.logger.info(np.array2string(np.linalg.inv(Te_e2c), separator=', '))
        loguru.logger.info("Residual Term")
        loguru.logger.info(str(res))
        ######
        Tc_c2e = np.linalg.inv(Te_e2c)
        Tc_c2e_gt = ref_Tc_c2e
        dof6_gt = utils_3d.se3_log_map(torch.from_numpy(Tc_c2e_gt).float()[None].permute(0, 2, 1), backend="opencv")[0]
        dof6 = utils_3d.se3_log_map(torch.from_numpy(Tc_c2e).float()[None].permute(0, 2, 1), backend="opencv")[0]
        loguru.logger.info(f"error {dof6_gt - dof6}")
        R_err, t_err = pose_distance(torch.from_numpy(Tc_c2e).float(), torch.from_numpy(Tc_c2e_gt).float())
        R_err = np.rad2deg(R_err.item())
        t_err = t_err.item() * 100

        loguru.logger.info(f"R_err {R_err:.4f} degree, t_err {t_err:.4f} cm")
        # loguru.logger.info(f"pose distance {pose_distance(torch.from_numpy(Tc_c2e).float(), torch.from_numpy(Tc_c2e_gt).float())}")
        log_file = loguru.logger.add(osp.join(self.total_cfg.sim_hec_eye_in_hand.outdir, "log.txt"), rotation="1 day", level="DEBUG")
        loguru.logger.debug(f"R_err {R_err:.4f} degree, t_err {t_err:.4f} cm")
        loguru.logger.remove(log_file)

        return Te_e2c, data_num

    def load_poses(self):
        raise NotImplementedError()

    @staticmethod
    def skew_matrix(v):
        v = v.flatten()
        if len(v) != 3:
            raise RuntimeError("Skew matrix should take a 3-d vector as input")
        a1, a2, a3 = v
        return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    def __calculate(self, Hee2base, Hmarker2cam):
        """
        :param Hee2base: Hee2base
        :param Hmarker2cam: Hmarker2cam
        :return:
        """
        # This is the eye-to-hand version, not eye-in-hand
        n = Hee2base.shape[2]
        assert n == Hmarker2cam.shape[2], "Input matrix should have same number"

        Hgij_list = []
        Hcij_list = []
        A = np.zeros([3 * n * (n - 1) // 2, 3])
        b = np.zeros([3 * n * (n - 1) // 2])

        # for i in range(n - 1):
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # tmp1 = Hee2base[:, :, j] @ np.linalg.inv(Hee2base[:, :, i])
                tmp1 = np.linalg.inv(Hee2base[:, :, j]) @ Hee2base[:, :, i]
                tmp2 = Hmarker2cam[:, :, j] @ np.linalg.inv(Hmarker2cam[:, :, i])
                Hgij_list.append(tmp1)
                Hcij_list.append(tmp2)

                rgij = cv2.Rodrigues(tmp1[:3, :3])[0].reshape(3, )
                rcij = cv2.Rodrigues(tmp2[:3, :3])[0].reshape(3, )
                tgij = np.linalg.norm(rgij)
                tcij = np.linalg.norm(rcij)
                if tgij > 1e-2 and tcij > 1e-2:
                    rgij /= tgij
                    rcij /= tcij

                    # Turn it into modified rodrigues in Tsai
                    Pgij = 2 * np.sin(tgij / 2) * rgij
                    Pcij = 2 * np.sin(tcij / 2) * rcij

                    # Solve equation: skew(Pgij+Pcij)*x = Pcij-Pgij
                    # A = skew(Pgij+Pcij) b = Pcij-Pgij
                    A[3 * idx:3 * idx + 3, 0:3] = self.skew_matrix(Pgij + Pcij)
                    b[3 * idx:3 * idx + 3] = Pcij - Pgij
                    idx += 1
        # for fragment in self.fragments:
        #     start, end = fragment
        #     for i in range(start, end - 1):
        A = A[:idx * 3, :]
        b = b[:idx * 3]
        x = np.dot(np.linalg.pinv(A), b.reshape(-1, 1))

        # Compute residue
        err = np.dot(A, x) - b.reshape(-1, 1)
        res_rotation = np.sqrt(sum((err * err)) / A.shape[0] // 3)
        Pcg = 2 * x / (np.sqrt(1 + np.linalg.norm(x) ** 2))
        Rcg = (1 - np.linalg.norm(Pcg) ** 2 / 2) * np.eye(3) + 0.5 * (
                np.dot(Pcg.reshape((3, 1)), Pcg.reshape((1, 3))) + np.sqrt(
            4 - np.linalg.norm(Pcg) ** 2) * self.skew_matrix(Pcg.reshape(3, )))

        # Compute translation from A*Tcg = b
        # (rgij-I)*Tcg=rcg*tcij-tgij
        for i in range(A.shape[0] // 3):
            A[3 * i:3 * i + 3, :] = (Hgij_list[i][:3, :3] - np.eye(3))
            b[3 * i:3 * i + 3] = np.dot(Rcg, Hcij_list[i][:3, 3]) - Hgij_list[i][:3, 3]

        Tcg = np.dot(np.linalg.pinv(A), b)
        err = np.dot(A, Tcg) - b
        res_translation = np.sqrt(sum(err ** 2) / (A.shape[0] // 3))
        Hc2g = np.hstack((Rcg, Tcg.reshape(3, 1)))
        Hc2g = np.vstack((Hc2g, np.array([[0, 0, 0, 1]])))
        error = np.hstack((res_translation, res_rotation))

        return Hc2g, error

    def test(self):
        Hcam2base = np.eye(4)
        Hcam2base[:3, :3] = transforms3d.euler.euler2mat(1, 0.5, 0.2)
        Hcam2base[:3, 3] = np.random.rand(3)
        Hmarker2ee = np.eye(4)
        Hmarker2ee[:3, :3] = transforms3d.euler.euler2mat(1, 0.5, 0.2)
        Hmarker2ee[:3, 3] = np.random.rand(3)

        n = 6
        input1 = np.ones([4, 4, n])
        input2 = np.ones([4, 4, n])
        for i in range(n):
            Hee2base = np.eye(4)
            random_quat = np.random.rand(4)
            random_quat /= np.linalg.norm(random_quat)
            Hee2base[:3, :3] = transforms3d.quaternions.quat2mat(random_quat)
            Hee2base[:3, 3] = np.random.rand(3)
            Hmarker2cam = np.linalg.inv(Hcam2base) @ Hee2base @ Hmarker2ee

            input1[:, :, i] = Hee2base
            input2[:, :, i] = Hmarker2cam

        Hcam2base_est, res = self.__calculate(input1, input2)

        print("*" * 10, "Result", "*" * 10)
        print("The transformation from:")
        print(Hcam2base_est)
        print("Ground Truth")
        print(Hcam2base)
        print("Residual Term")
        print(res)

    def vis_base_origin_in_cam(self, Tb_b2c, dps):
        Tc_c2b = np.linalg.inv(Tb_b2c)
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name=f"hesolver_vis_base_origin_in_cam",
                      auto_increase=True, enable=self.dbg)
        depth = dps['depth'].cpu().numpy()[0]
        rgb = dps['rgb'].cpu().numpy()[0]
        # depth_file = sorted(glob.glob(osp.join(self.folder, "depth/*.png")))[0]
        # color_file = sorted(glob.glob(osp.join(self.folder, "color/*.png")))[0]
        # depth = cv2.imread(depth_file, 2).astype(np.float32) / 1000.0
        # rgb = imageio.imread(color_file)
        H, W, _ = rgb.shape
        # K = np.loadtxt(osp.join(self.folder, "K.txt"))
        K = dps['K'].cpu().numpy()[0]
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
        vis3d.add_image(rgb)
        vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=2)
        base_in_cam = utils_3d.transform_points(np.array([0, 0, 0]), Tc_c2b)
        vis3d.add_spheres(base_in_cam, 0.05)
        # vis3d.add_xarm(np.zeros(9), Tw_w2B=pose_cam_to_base)
        mesh_base = trimesh.load_mesh(f"data/xarm_description/meshes/xarm7/visual/link_base.STL")
        vis3d.add_mesh(utils_3d.transform_points(mesh_base.vertices, Tc_c2b), mesh_base.faces)
        rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh_base, Tc_c2b, H, W, K)
        rgb = rgb / 255.0
        rgb[rendered_mask] *= 0.5
        rgb[rendered_mask] += 0.5
        vis3d.add_image(rgb, name='rendered_mask')
        print()
        # fake_b2c = self.poses_cam_marker[0] @ np.linalg.inv(self.poses_ef_base[0])
        # base_in_cam = utils_3d.transform_points(np.array([0, 0, 0]), fake_b2c)
        # vis3d.add_spheres(base_in_cam, 0.05, name='fake')
        # vis3d.add_mesh(utils_3d.transform_points(mesh_base.vertices, fake_b2c), mesh_base.faces, name='fake')

    # def vis_on_testset(self, Tb_b2c):
    #     Tc_c2b = np.linalg.inv(Tb_b2c)
    #
    #     vis3d = Vis3D(
    #         xyz_pattern=("x", "-y", "-z"),
    #         out_folder="dbg",
    #         sequence=f"vis_on_testset_icp_{self.cfg.icp}_cpa_{self.cfg.cpa}",
    #         auto_increase=True,
    #         enable=True,
    #     )
    #     LINK = 7
    #     mesh_path = f"data/xarm_description/meshes/xarm7/visual/link{LINK}.STL"
    #     mesh = trimesh.load(mesh_path)
    #     K = np.loadtxt(osp.join(data_dir, "K.txt"))
    #     fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    #     rgb_files = sorted(glob.glob(osp.join(data_dir, "color/*png")))
    #     depth_files = sorted(glob.glob(osp.join(data_dir, "depth/*png")))
    #     pose_eb_files = sorted(glob.glob(osp.join(data_dir, "qpos/*txt")))
    #     for i in trange(len(rgb_files)):
    #         vis3d.set_scene_id(i)
    #         rgb = imageio.imread(rgb_files[i])
    #         H, W = rgb.shape[:2]
    #         depth = cv2.imread(depth_files[i], 2).astype(np.float32) / 1000.0
    #         pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
    #         vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=2)
    #         qpos = np.loadtxt(pose_eb_files[i])
    #         pq = self.sk.compute_forward_kinematics(qpos, LINK + 1)
    #         R = transforms3d.quaternions.quat2mat(pq.q)
    #         t = pq.p
    #         Tb_b2e = utils_3d.matrix_3x4_to_4x4(np.concatenate([R, t[:, None]], axis=-1))
    #         Tc_c2e = Tc_c2b @ Tb_b2e
    #
    #         vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, Tc_c2e), mesh.faces)
    #         rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh, Tc_c2e, H, W, K)
    #         rgb = rgb / 255.0
    #         rgb[rendered_mask] *= 0.5
    #         rgb[rendered_mask] += 0.5
    #         vis3d.add_image(rgb, name='rendered_mask')
    #         # print()

    # def debug(self, solution, other_solution):
    #     rot_solution = np.rad2deg(transforms3d.euler.mat2euler(solution[:3, :3]))
    #     rot_other_solution = np.rad2deg(transforms3d.euler.mat2euler(other_solution[:3, :3]))
    #     print("solution", solution[:3, 3], rot_solution)
    #     print("other_solution", other_solution[:3, 3], rot_other_solution)
    #     errs, errs_solution = [], []
    #     for index0 in range(len(self.Tb_b2es) - 1):
    #         index1 = index0 + 1
    #         err = self.Tb_b2es[index1] @ np.linalg.inv(self.Tb_b2es[index0]) @ other_solution - other_solution @ \
    #               self.Tc_c2ms[index1] @ np.linalg.inv(self.Tc_c2ms[index0])
    #
    #         err_solution = self.Tb_b2es[index1] @ np.linalg.inv(
    #             self.Tb_b2es[index0]) @ solution - solution @ \
    #                        self.Tc_c2ms[index1] @ np.linalg.inv(self.Tc_c2ms[index0])
    #         errs.append(np.linalg.norm(err))
    #         errs_solution.append(np.linalg.norm(err_solution))
    #     print("err other norm", np.mean(errs))
    #     print("err solution norm", np.mean(errs_solution))
    #
    #     print()

    def load_gt_marker_pose(self):
        pose_cm_files = sorted(glob.glob(osp.join(self.folder, "Tc_c2m/*.txt")))
        Tc_c2ms = np.stack([np.loadtxt(pcf) for pcf in pose_cm_files])
        return Tc_c2ms
