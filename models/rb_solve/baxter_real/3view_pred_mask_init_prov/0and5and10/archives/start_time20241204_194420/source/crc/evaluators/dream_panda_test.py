import os
import os.path as osp

import cv2
import loguru
import numpy as np

from crc.registry import EVALUATORS
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils import comm, utils_3d


@EVALUATORS.register('dream_panda_test')
def build(cfg):
    def f(x, trainer):
        if comm.get_rank() == 0:
            dof = trainer.model.dof
            Tc_c2b_pred = utils_3d.se3_exp_map(dof[None]).permute(0, 2, 1)[0].detach().cpu().numpy()
            data_dir = trainer.train_dl.dataset.data_dir
            # Tc_c2b_gt = np.loadtxt(osp.join(data_dir, "Tc_c2b.txt"))
            K = np.loadtxt(osp.join(data_dir, "../K.txt"))
            kpt_3d_cam = np.loadtxt(osp.join(data_dir, "chosen_keypoint_3dlocations.txt"))
            kpt_2d = np.loadtxt(osp.join(data_dir, "chosen_keypoint_locations.txt"))
            kpt_3d_base = np.loadtxt(osp.join(data_dir, "chosen_keypoint_3dlocations_in_base.txt"))
            kpt_3d_cam_pred = utils_3d.transform_points(kpt_3d_base, Tc_c2b_pred)
            fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            kpt_2d_pred = utils_3d.rect_to_img(fu, fv, cu, cv, kpt_3d_cam_pred)
            inference_dir = osp.join(trainer.output_dir, "inference")
            os.makedirs(inference_dir, exist_ok=True)
            np.savetxt(osp.join(inference_dir, "Tc_c2b_pred.txt"), Tc_c2b_pred)
            np.savetxt(osp.join(inference_dir, "kpt_3d_cam_pred.txt"), kpt_3d_cam_pred)
            np.savetxt(osp.join(inference_dir, "kpt_2d_pred.txt"), kpt_2d_pred)
            np.savetxt(osp.join(inference_dir, "kpt_3d_cam.txt"), kpt_3d_cam)
            np.savetxt(osp.join(inference_dir, "kpt_2d.txt"), kpt_2d)

    return f


def keypoint_metrics(
        keypoints_detected, keypoints_gt, image_resolution, auc_pixel_threshold=20.0
):
    """
    Compute keypoint metrics for a batch of samples.
    @param keypoints_detected: array of shape (n_samples, n_keypoints, 2) 5998,7,2
    @param keypoints_gt: array of shape (n_samples, n_keypoints, 2) 5998,7,2
    @param image_resolution: tuple of (width, height) 640,480
    """

    # TBD: input argument handling
    num_gt_outframe = 0
    num_gt_inframe = 0
    num_missing_gt_outframe = 0
    num_found_gt_outframe = 0
    num_found_gt_inframe = 0
    num_missing_gt_inframe = 0

    kp_errors = []
    for kp_proj_detect, kp_proj_gt in zip(keypoints_detected, keypoints_gt):

        if (
                kp_proj_gt[0] < 0.0
                or kp_proj_gt[0] > image_resolution[0]
                or kp_proj_gt[1] < 0.0
                or kp_proj_gt[1] > image_resolution[1]
        ):
            # GT keypoint is out of frame
            num_gt_outframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (correct)
                num_missing_gt_outframe += 1
            else:
                # Found a keypoint (wrong)
                num_found_gt_outframe += 1

        else:
            # GT keypoint is in frame
            num_gt_inframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (wrong)
                num_missing_gt_inframe += 1
            else:
                # Found a keypoint (correct)
                num_found_gt_inframe += 1

                kp_errors.append((kp_proj_detect - kp_proj_gt).tolist())

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)

        # compute the auc
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
                np.trapz(y_values, dx=delta_pixel)
                / float(auc_pixel_threshold)
                / float(num_gt_inframe)
        )

    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None

    metrics = {
        "num_gt_outframe": num_gt_outframe,
        "num_missing_gt_outframe": num_missing_gt_outframe,
        "num_found_gt_outframe": num_found_gt_outframe,
        "num_gt_inframe": num_gt_inframe,
        "num_found_gt_inframe": num_found_gt_inframe,
        "num_missing_gt_inframe": num_missing_gt_inframe,
        "l2_error_mean_px": kp_l2_error_mean,
        "l2_error_median_px": kp_l2_error_median,
        "l2_error_std_px": kp_l2_error_std,
        "l2_error_auc": kp_auc,
        "l2_error_auc_thresh_px": auc_pixel_threshold,
        "pck_values": pck_values,
        "y_values": np.array(y_values) / float(num_gt_inframe)
    }
    return metrics


def auc_metrics(
        kpts_3d_pred,
        kpts_3d_gt,
        # num_inframe_projs_gt,
        # num_min_inframe_projs_gt_for_pnp=4,
        add_auc_threshold=0.1,
        # pnp_magic_number=-999.0,
):
    add = np.linalg.norm(kpts_3d_pred - kpts_3d_gt, axis=1)
    # pnp_add = np.array(pnp_add)
    # num_inframe_projs_gt = np.array(num_inframe_projs_gt)

    # idx_pnp_found = np.where(pnp_add > pnp_magic_number)[0]
    # add_pnp_found = pnp_add[idx_pnp_found]
    num_pnp_found = len(kpts_3d_pred)

    mean_add = np.mean(add)
    median_add = np.median(add)
    std_add = np.std(add)

    # num_pnp_possible = len(
    #     np.where(num_inframe_projs_gt >= num_min_inframe_projs_gt_for_pnp)[0]
    # )
    # num_pnp_not_found = num_pnp_possible - num_pnp_found

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = (add <= value).mean()
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    metrics = {
        # "num_pnp_found": num_pnp_found,
        # "num_pnp_not_found": num_pnp_not_found,
        # "num_pnp_possible": num_pnp_possible,
        # "num_min_inframe_projs_gt_for_pnp": num_min_inframe_projs_gt_for_pnp,
        # "pnp_magic_number": pnp_magic_number,
        "add_mean": mean_add,
        "add_median": median_add,
        "add_std": std_add,
        "add_auc": auc,
        "add_auc_thresh": add_auc_threshold,
        "xs": add_threshold_values,
        "ys": counts,
    }
    return metrics
