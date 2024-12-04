import PIL.Image as Image
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
from dkm import DKMv3_indoor

import matplotlib.cm as cm

from crc.utils.plotting import make_matching_figure


class DKMHelper:
    _feature_matcher = None

    @classmethod
    def get_feature_matcher(cls):
        if cls._feature_matcher is None:
            loguru.logger.info("Loading feature matcher...")
            matcher = DKMv3_indoor()  # creates an indoor trained model
            cls._feature_matcher = matcher
        return cls._feature_matcher

    @staticmethod
    def reset():
        del DKMHelper._feature_matcher
        DKMHelper._feature_matcher = None
        torch.cuda.empty_cache()


@torch.no_grad()
def get_feature_matching(img_path0, img_path1, mask0=None, mask1=None, dbg=False,
                         resize_factor=1, num=10000):
    matcher = DKMHelper.get_feature_matcher()
    W, H = Image.open(img_path1).size
    warp, certainty = matcher.match(Image.open(img_path0).convert("RGB").resize((int(W * resize_factor), int(H * resize_factor))),
                                    Image.open(img_path1).convert("RGB").resize((int(W * resize_factor), int(H * resize_factor))))  # produces a warp of shape [B,H,W,4] and certainty of shape [B,H,W]
    matches, certainty = matcher.sample(warp, certainty, num=num)  # samples from the warp using the certainty
    mkpts0, mkpts1 = matcher.to_pixel_coordinates(matches, int(H * resize_factor), int(W * resize_factor),
                                                  int(H * resize_factor), int(W * resize_factor))  # convenience function to convert
    mkpts0 = mkpts0.detach().cpu().numpy()
    mkpts1 = mkpts1.detach().cpu().numpy()
    certainty = certainty.detach().cpu().numpy()

    mkpts0 = mkpts0 / resize_factor
    mkpts1 = mkpts1 / resize_factor

    if mask0 is not None:
        keep = mask0[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        certainty = certainty[keep]
    if mask1 is not None:
        keep = mask1[mkpts1[:, 1].astype(int), mkpts1[:, 0].astype(int)]
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        certainty = certainty[keep]

    mconf = certainty
    if dbg:
        # Draw visualization
        color = cm.jet(mconf)
        text = [
            'DKM',
            'Matches: {}'.format(len(mkpts0)),
        ]
        img0_raw = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        fig.show()
    return mkpts0, mkpts1, mconf
