import matplotlib.pyplot as plt
import warnings

import numpy as np
import cv2
import os
import os.path as osp
import imageio
from copy import deepcopy

import loguru
import torch
from crc.modeling.models.loftr import LoFTR, default_cfg
import matplotlib.cm as cm

from crc.utils.plotting import make_matching_figure
from crc.utils.timer import EvalTime


class LoFTRHelper:
    _feature_matcher = None

    @classmethod
    def get_feature_matcher(cls, thresh=0.2):
        if cls._feature_matcher is None:
            loguru.logger.info("Loading feature matcher...")
            _default_cfg = deepcopy(default_cfg)
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            matcher = LoFTR(config=_default_cfg)
            ckpt_path = "weights/indoor_ds_new.ckpt"
            if not osp.exists(ckpt_path):
                loguru.logger.info("Downloading feature matcher...")
                os.makedirs("weights", exist_ok=True)
                import gdown
                gdown.cached_download(url="https://drive.google.com/uc?id=19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS",
                                      path=ckpt_path)
            matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
            matcher = matcher.eval().cuda()
            cls._feature_matcher = matcher
        cls._feature_matcher.coarse_matching.thr = thresh
        return cls._feature_matcher


def get_feature_matching(img_path0, img_path1, thresh=0.2, dbg=False):
    evaltime = EvalTime(disable=True)
    evaltime("")
    matcher = LoFTRHelper.get_feature_matcher(thresh=thresh)
    evaltime("get_feature_matcher")
    img0_raw = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    evaltime("io")
    original_shape = img0_raw.shape
    img0_raw_resized = img0_raw
    img1_raw_resized = img1_raw
    evaltime("resize")

    img0 = torch.from_numpy(img0_raw_resized)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw_resized)[None][None].cuda() / 255.
    evaltime("to cuda")
    batch = {'image0': img0, 'image1': img1}

    # if mask0 is not None:
    #     mask0 = cv2.resize(mask0.astype(np.uint8), (mask0.shape[0] // 8, mask0.shape[1] // 8))
    #     mask0 = torch.from_numpy(mask0)[None].cuda().bool()
    #     batch['mask0'] = mask0
    # if mask1 is not None:
    #     mask1 = cv2.resize(mask1.astype(np.uint8), (mask1.shape[0] // 8, mask1.shape[1] // 8))
    #     mask1 = torch.from_numpy(mask1)[None].cuda().bool()
    #     batch['mask1'] = mask1

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    evaltime("forward")
    # mkpts0[:, 0] = mkpts0[:, 0] * original_shape[1] / 480
    # mkpts0[:, 1] = mkpts0[:, 1] * original_shape[0] / 480
    # mkpts1[:, 0] = mkpts1[:, 0] * original_shape[1] / 480
    # mkpts1[:, 1] = mkpts1[:, 1] * original_shape[0] / 480
    if dbg:
        # Draw visualization
        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        fig.show()
    evaltime("return")
    return mkpts0, mkpts1, mconf
