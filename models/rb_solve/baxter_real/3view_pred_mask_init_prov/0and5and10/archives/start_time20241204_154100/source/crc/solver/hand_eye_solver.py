import tqdm
import torch
import io

import matplotlib.pyplot as plt
import imageio
import os.path as osp
import glob
import json
import os

import cv2
import loguru
import numpy as np
import transforms3d
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import trange

from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import utils_3d, render_api, plt_utils

from crc.utils.utils_3d import Rt_to_pose, reproj_error, cpa_pytorch3d_api
from crc.utils.vis3d_ext import Vis3D

VELOCITY_THRESHOLD = 0.02


class HandEyeSolver:
    def __init__(self, cfg):
        self.folder = cfg.data_dir
        self.cfg = cfg
        # self.fragments = cfg.fragments
        # self.n = cfg.n
        assert os.path.exists(self.folder)
        self.sk = SAPIENKinematicsModelStandalone("data/xarm7_textured.urdf")
        self.Tc_c2ms_gt = self.load_gt_marker_pose()
        if self.cfg.use_gt_marker_pose:
            loguru.logger.info("Use GT marker pose!!!")
            self.Tc_c2ms = self.Tc_c2ms_gt
        else:
            if not self.cfg.icp:
                self.Tc_c2ms = self.calculate_marker_pose()
            else:
                raise NotImplementedError()
                self.Tc_c2ms = self.load_marker_pose()
        self.Tb_b2es = self.load_poses()
        keep = (self.Tc_c2ms != 0).sum(1).sum(1) > 0
        self.Tb_b2es = self.Tb_b2es[keep]
        self.Tc_c2ms = self.Tc_c2ms[keep]
        self.data_num = self.Tc_c2ms.shape[0]

    def calculate_marker_pose(self):
        data_dir = self.folder
        rgb_paths = sorted(glob.glob(osp.join(data_dir, "color/*.png")))
        depth_paths = sorted(glob.glob(osp.join(data_dir, "depth/*.png")))
        assert int(self.cfg.icp) + int(self.cfg.cpa) < 2
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg", sequence_name="check_marker_to_cam",
                      auto_increase=True, enable=True)
        mesh = trimesh.primitives.Box(extents=np.array([0.04, 0.05, 0.01]))
        mesh = trimesh.Trimesh(mesh.vertices + np.array([[0.01, 0.015, 0]]), mesh.faces)
        model = {'pts': mesh.vertices * 1000.0, 'faces': mesh.faces}
        if self.cfg.icp:
            from crc.utils.icp_utils import ICPRefiner
            icp_refiner = ICPRefiner(model, (640, 480))
        Tc_c2ms = []
        reproj_errors = []
        for i, fname in enumerate(tqdm.tqdm(rgb_paths)):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((3 * 4, 3), np.float32)
            objp[:, :2] = np.mgrid[0:3, 0:4].T.reshape(-1, 2) / 100.0
            axis = np.float32([[4, 0, 0], [0, 3, 0], [0, 0, 0]]).reshape(-1, 3) / 100.0
            img = cv2.imread(fname)
            H, W, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (3, 4), None)

            K = np.loadtxt(osp.join(data_dir, "K.txt"))
            if ret is False:
                Tc_c2ms.append(np.zeros((4, 4)))
                loguru.logger.warning(f"Cannot find chessboard in {fname}")
                continue
            if self.cfg.subpixel:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners, K, cv2.SOLVEPNP_ITERATIVE)
            # tvecs[2, 0] = tvecs[2, 0] + 0.01
            # tvecs[[0, 1], 0] = tvecs[[0, 1], 0] + np.random.rand(2) * 0.005
            R, _ = cv2.Rodrigues(rvecs)
            Tc_c2m = utils_3d.Rt_to_pose(R, tvecs.reshape(3))
            rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh, Tc_c2m, H, W, K)
            vis3d.add_image(plt_utils.vis_mask(img, rendered_mask.astype(np.uint8), [255, 0, 0]), name='chessboard')
            pts_cam = utils_3d.transform_points(axis, Tc_c2m)

            plt.figure(dpi=200)
            plt.imshow(img)
            plt.scatter(corners[:, 0, 0], corners[:, 0, 1], marker='.')
            for tmpi, c in enumerate(corners):
                plt.text(corners[tmpi, 0, 0], corners[tmpi, 0, 1], str(tmpi), fontsize=4, fontdict={'color': 'red'})
            axis_pt_img = utils_3d.rect_to_img(K, utils_3d.transform_points(axis, Tc_c2m))
            plt.plot(axis_pt_img[[0, 2], 0], axis_pt_img[[0, 2], 1], 'r-')
            plt.plot(axis_pt_img[[1, 2], 0], axis_pt_img[[1, 2], 1], 'b-')
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            vis3d.add_image(Image.open(img_buf), name="corners")
            plt.close("all")

            # print('reproj_err', reproj_error(K, Tc_c2m, corners[:, 0], objp))
            reproj_errors.append(reproj_error(K, Tc_c2m, corners[:, 0], objp))
            depth_image = cv2.imread(depth_paths[i], 2).astype(np.float32) / 1000.0

            pts_rect = utils_3d.depth_to_rect(K[0, 0], K[1, 1], K[0, 2], K[1, 2], depth_image)
            vis3d.add_point_cloud(pts_rect, img.reshape(-1, 3), max_z=2)
            vis3d.add_spheres(pts_cam, 0.01, name='axis')
            vis3d.add_mesh(utils_3d.transform_points(model['pts'] / 1000.0, Tc_c2m), model['faces'], name="box")
            if self.cfg.icp:
                R_refined, t_refined = icp_refiner.refine(depth_image * 1000.0, Tc_c2m[:3, :3], Tc_c2m[:3, 3] * 1000.0,
                                                          K.copy(),
                                                          depth_only=True, max_mean_dist_factor=2.0)
                R_refined, _ = icp_refiner.refine(depth_image, R_refined, t_refined, K.copy(), no_depth=True)
                Tc_c2m = utils_3d.matrix_3x4_to_4x4(np.hstack([R_refined, t_refined.reshape(3, 1) / 1000.0]))
                print('reproj_err after icp', reproj_error(K, Tc_c2m, corners[:, 0], objp))
                vis3d.add_mesh(utils_3d.transform_points(model['pts'] / 1000.0, Tc_c2m), model['faces'], name="box_icp")
                # plt.subplot(2, 1, 2)
                # pts_cam = utils_3d.transform_points(axis, Tc_c2m)
                # pts_img = utils_3d.rect_to_img(K[0, 0], K[1, 1], K[0, 2], K[1, 2], pts_cam)
                # plt.imshow(img)
                # plt.scatter(pts_img[:, 0], pts_img[:, 1], marker=".")
            if self.cfg.cpa:
                pts0 = objp
                # rows = corners2[:, 0, 1].astype(int)
                # cols = corners2[:, 0, 0].astype(int)
                rows = np.round(corners[:, 0, 1]).astype(int)
                cols = np.round(corners[:, 0, 0]).astype(int)
                pts1 = pts_rect.reshape(H, W, 3)[rows, cols]
                keep = pts1[:, 2] > 0
                if keep.sum() > 6:
                    pts0 = pts0[keep]
                    pts1 = pts1[keep]
                    vis3d.add_point_cloud(pts0, name='cpa_pts0')
                    vis3d.add_point_cloud(pts1, name='cpa_pts1')
                    pose_cpa = cpa_pytorch3d_api(pts0, pts1)
                    vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, pose_cpa), mesh.faces, name='mesh_cpa')
                    rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh, pose_cpa, H, W, K)
                    tmp = img / 255.0
                    tmp[rendered_mask] *= 0.5
                    tmp[rendered_mask] += 0.5
                    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    pts_img = utils_3d.rect_to_img(fu, fv, cu, cv, utils_3d.transform_points(objp, pose_cpa))
                    pts_img = pts_img.astype(int)
                    tmp[pts_img[:, 1], pts_img[:, 0]] = np.array([1, 0, 0])
                    vis3d.add_image(tmp, name='rendered_mask')
                    reproj_err = reproj_error(K, pose_cpa, corners[:, 0], objp)
                    print('reproj_err after cpa', reproj_err)
                    print("tsfm err", np.linalg.norm(utils_3d.transform_points(pts0, pose_cpa) - pts1, axis=1).mean(0))
                    if reproj_err < 5:
                        pose = pose_cpa
                    else:
                        print()
            # plt.show()
            # vis3d.add_image(img)
            vis3d.increase_scene_id()
            # outfile = fname.replace("rgb_", "pose_cm_").replace(".png", ".txt")
            # np.savetxt(outfile, pose)
            Tc_c2ms.append(Tc_c2m)
        Tc_c2ms = np.stack(Tc_c2ms)
        reproj_errors = np.array(reproj_errors)
        loguru.logger.info(f"Reprojection Error {reproj_errors}")
        return Tc_c2ms

    def calculate_and_show(self, ref_Tc_c2b=None):
        vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="calcuate_and_show")
        ee_pose = self.Tb_b2es.transpose([1, 2, 0])
        marker_pose = self.Tc_c2ms.transpose([1, 2, 0])
        Tb_b2c, res = self.__calculate(ee_pose, marker_pose)

        loguru.logger.info("*" * 10 + " Result " + "*" * 10)
        loguru.logger.info("The transformation Tc_c2b:")
        loguru.logger.info(np.array2string(np.linalg.inv(Tb_b2c), separator=', '))
        loguru.logger.info("Residual Term")
        loguru.logger.info(str(res))
        ######
        # fake_Tc_c2b = self.Tc_c2ms[0] @ self.Tb_b2es[0]
        Tc_c2b = np.linalg.inv(Tb_b2c)
        # loguru.logger.info(f"fake Tc_c2b {fake_Tc_c2b}, computed Tc_c2b {Tc_c2b}")
        for i in range(len(self.Tc_c2ms)):
            Tm_m2e = np.linalg.inv(self.Tc_c2ms[i]) @ Tc_c2b @ self.Tb_b2es[i]
            Te_e2m = np.linalg.inv(Tm_m2e)
            loguru.logger.info(f"computed Te_e2m_{i} {np.array2string(Te_e2m, separator=',')}")
            rot = np.rad2deg(transforms3d.euler.mat2euler(Te_e2m[:3, :3]))
            loguru.logger.info(f"computed Te_e2m rot {i} {np.array2string(rot, separator=',')}")
        if ref_Tc_c2b is not None:
            for i in range(len(self.Tc_c2ms)):
                ref_Tm_m2e = np.linalg.inv(self.Tc_c2ms[i]) @ ref_Tc_c2b @ self.Tb_b2es[i]
                ref_Te_e2m = np.linalg.inv(ref_Tm_m2e)
                loguru.logger.info(f"ref Te_e2m {i} {np.array2string(ref_Te_e2m, separator=',')}")
                rot = np.rad2deg(transforms3d.euler.mat2euler(ref_Te_e2m[:3, :3]))
                loguru.logger.info(f"ref Te_e2m rot {i} {np.array2string(rot, separator=',')}")

        if osp.exists(osp.join(self.folder, "Tc_c2b.txt")):
            Tc_c2b_gt = np.loadtxt(osp.join(self.folder, "Tc_c2b.txt"))
            dof6_gt = utils_3d.se3_log_map(torch.from_numpy(Tc_c2b_gt).float()[None].permute(0, 2, 1))[0]
            dof6 = utils_3d.se3_log_map(torch.from_numpy(Tc_c2b).float()[None].permute(0, 2, 1))[0]
            loguru.logger.info(f"error {dof6_gt - dof6}")
        return Tb_b2c

    def load_poses(self):
        # pose_cm_files = sorted(glob.glob(osp.join(self.folder, "pose_cm*.txt")))
        pose_eb_files = sorted(glob.glob(osp.join(self.folder, "qpos/*.txt")))
        # poses_cam_marker = np.stack([np.loadtxt(pcf) for pcf in pose_cm_files])
        poses_ef_base_radians = np.stack([np.loadtxt(pef) for pef in pose_eb_files])
        Tb_b2es = []
        for peb in poses_ef_base_radians:
            pq = self.sk.compute_forward_kinematics(peb, 8)
            Tb_b2e = Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
            Tb_b2es.append(Tb_b2e)
        Tb_b2es = np.stack(Tb_b2es)
        return Tb_b2es

    def load_marker_pose(self):
        pose_cm_files = sorted(glob.glob(osp.join(self.folder, "pose_cm*.txt")))
        Tc_c2ms = np.stack([np.loadtxt(pcf) for pcf in pose_cm_files])
        return Tc_c2ms

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
                tmp1 = Hee2base[:, :, j] @ np.linalg.inv(Hee2base[:, :, i])
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

    def vis_base_origin_in_cam(self, Tb_b2c):
        Tc_c2b = np.linalg.inv(Tb_b2c)
        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg",
                      sequence_name=f"vis_base_origin_in_cam_icp_{self.cfg.icp}_cpa_{self.cfg.cpa}", auto_increase=True,
                      enable=True)
        depth_file = sorted(glob.glob(osp.join(self.folder, "depth/*.png")))[0]
        color_file = sorted(glob.glob(osp.join(self.folder, "color/*.png")))[0]
        depth = cv2.imread(depth_file, 2).astype(np.float32) / 1000.0
        rgb = imageio.imread(color_file)
        H, W, _ = rgb.shape
        K = np.loadtxt(osp.join(self.folder, "K.txt"))
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

    def vis_on_testset(self, Tb_b2c, data_dir):
        Tc_c2b = np.linalg.inv(Tb_b2c)

        vis3d = Vis3D(xyz_pattern=("x", "-y", "-z"), out_folder="dbg",
                      sequence_name=f"vis_on_testset_icp_{self.cfg.icp}_cpa_{self.cfg.cpa}", auto_increase=True,
                      enable=True)
        LINK = 7
        mesh_path = f"data/xarm_description/meshes/xarm7/visual/link{LINK}.STL"
        mesh = trimesh.load(mesh_path)
        K = np.loadtxt(osp.join(data_dir, "K.txt"))
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rgb_files = sorted(glob.glob(osp.join(data_dir, "color/*png")))
        depth_files = sorted(glob.glob(osp.join(data_dir, "depth/*png")))
        pose_eb_files = sorted(glob.glob(osp.join(data_dir, "qpos/*txt")))
        for i in trange(len(rgb_files)):
            vis3d.set_scene_id(i)
            rgb = imageio.imread(rgb_files[i])
            H, W = rgb.shape[:2]
            depth = cv2.imread(depth_files[i], 2).astype(np.float32) / 1000.0
            pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
            vis3d.add_point_cloud(pts_rect, rgb.reshape(-1, 3), max_z=2)
            qpos = np.loadtxt(pose_eb_files[i])
            pq = self.sk.compute_forward_kinematics(qpos, LINK + 1)
            R = transforms3d.quaternions.quat2mat(pq.q)
            t = pq.p
            Tb_b2e = utils_3d.matrix_3x4_to_4x4(np.concatenate([R, t[:, None]], axis=-1))
            Tc_c2e = Tc_c2b @ Tb_b2e

            vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, Tc_c2e), mesh.faces)
            rendered_mask = render_api.nvdiffrast_render_mesh_api(mesh, Tc_c2e, H, W, K)
            rgb = rgb / 255.0
            rgb[rendered_mask] *= 0.5
            rgb[rendered_mask] += 0.5
            vis3d.add_image(rgb, name='rendered_mask')
            # print()

    def debug(self, solution, other_solution):
        rot_solution = np.rad2deg(transforms3d.euler.mat2euler(solution[:3, :3]))
        rot_other_solution = np.rad2deg(transforms3d.euler.mat2euler(other_solution[:3, :3]))
        print("solution", solution[:3, 3], rot_solution)
        print("other_solution", other_solution[:3, 3], rot_other_solution)
        errs, errs_solution = [], []
        for index0 in range(len(self.Tb_b2es) - 1):
            index1 = index0 + 1
            err = self.Tb_b2es[index1] @ np.linalg.inv(self.Tb_b2es[index0]) @ other_solution - other_solution @ \
                  self.Tc_c2ms[index1] @ np.linalg.inv(self.Tc_c2ms[index0])

            err_solution = self.Tb_b2es[index1] @ np.linalg.inv(
                self.Tb_b2es[index0]) @ solution - solution @ \
                           self.Tc_c2ms[index1] @ np.linalg.inv(self.Tc_c2ms[index0])
            errs.append(np.linalg.norm(err))
            errs_solution.append(np.linalg.norm(err_solution))
        print("err other norm", np.mean(errs))
        print("err solution norm", np.mean(errs_solution))

        print()

    def load_gt_marker_pose(self):
        pose_cm_files = sorted(glob.glob(osp.join(self.folder, "Tc_c2m/*.txt")))
        poses = [np.loadtxt(pcf) for pcf in pose_cm_files]
        if len(poses) > 0:
            Tc_c2ms = np.stack(poses)
        else:
            Tc_c2ms = np.empty([0, 4, 4])
        return Tc_c2ms
