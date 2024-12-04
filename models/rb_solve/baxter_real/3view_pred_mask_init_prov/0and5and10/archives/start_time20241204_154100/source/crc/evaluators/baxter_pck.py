import os
import os.path as osp

import cv2
import loguru
import numpy as np

from crc.registry import EVALUATORS
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import comm, utils_3d


@EVALUATORS.register('baxter_ee_pck')
def build(cfg):
    def f(x, trainer):
        if comm.get_rank() == 0:
            dof = trainer.model.dof
            # history_ops = trainer.model.history_ops
            Tc_c2b = utils_3d.se3_exp_map(dof[None]).permute(0, 2, 1)[0].detach().cpu().numpy()
            # Tc_c2b = np.loadtxt("/home/linghao/Datasets/baxter_real/baxter-real-dataset/cleaned/Tc_c2b.txt")
            # loguru.logger.info("using provided Tc_c2b!")
            # load gt
            data_dir = trainer.train_dl.dataset.data_dir
            K = np.loadtxt(osp.join(data_dir, "../../cleaned/K.txt"))
            dist = np.loadtxt(osp.join(data_dir, "../../cleaned/dist.txt"))
            ee_2d = np.loadtxt(osp.join(data_dir, "../../cleaned/ee_2d.txt"))
            ee_3d = np.loadtxt(osp.join(data_dir, "../../cleaned/ee_3d.txt"))
            urdf_file = "./baxter_common/baxter.urdf"
            sk = SAPIENKinematicsModelStandalone(urdf_file)
            qposes = np.loadtxt(osp.join(data_dir, "../../cleaned/joints.txt"))
            Tl_l2e = np.array([[1, 0, 0, 0.363],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            ee_in_base = []
            for qpos in qposes:
                pose38 = sk.compute_forward_kinematics([0] * 8 + qpos.tolist(), 38)
                pose38 = pose38.to_transformation_matrix()
                ee_pose = pose38 @ Tl_l2e
                ee_in_base.append(ee_pose[:3, 3])
            ee_in_base = np.array(ee_in_base)
            # 2D
            err_2ds = []
            for i in range(20):  # todo: use all?
                pts_img = cv2.projectPoints(utils_3d.transform_points(ee_in_base[i], Tc_c2b),
                                            rvec=(0, 0, 0), tvec=(0, 0, 0),
                                            cameraMatrix=K, distCoeffs=dist)[0][:, 0, :]
                err_2d = np.linalg.norm(ee_2d[i] - pts_img)
                err_2ds.append(err_2d)
            err_2ds = np.array(err_2ds)
            print("PCK2D")
            loguru.logger.info("err_2ds: {}".format(np.array2string(err_2ds, precision=2, separator=', ')))
            eval_dump_path = osp.join(trainer.output_dir, "evaluation/err2d.txt")
            os.makedirs(osp.dirname(eval_dump_path), exist_ok=True)
            np.savetxt(eval_dump_path, err_2ds)
            for thresh in [5, 10, 20, 30, 40, 50, 100, 150, 200]:
                # compute ratio of error < thresh
                loguru.logger.info("ratio of error < {}: {}".format(thresh, np.mean(err_2ds < thresh)))
            # 3D
            cpa_err = np.linalg.norm(utils_3d.transform_points(ee_in_base, Tc_c2b) - ee_3d, axis=-1)
            loguru.logger.info("PCK3D")
            loguru.logger.info("err_3ds: {}".format(np.array2string(cpa_err, precision=2, separator=', ')))
            eval_dump_path = osp.join(trainer.output_dir, "evaluation/err3d.txt")
            os.makedirs(osp.dirname(eval_dump_path), exist_ok=True)
            np.savetxt(eval_dump_path, cpa_err)
            for thresh in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
                # compute ratio of error < thresh
                loguru.logger.info("ratio of error < {}cm: {}".format(thresh, np.mean(cpa_err < thresh)))

    return f
