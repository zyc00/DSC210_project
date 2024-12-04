import pickle

import tqdm
import numpy as np
import imageio
import cv2
import os.path as osp
import glob

from detectron2.structures import BoxMode


def get_xarm_dicts(data_dir):
    img_files = sorted(glob.glob(osp.join(data_dir, "color/*.png")))
    # mask_files = sorted(glob.glob(osp.join(data_dir, "gt_mask/*png")))
    polys = pickle.load(open(osp.join(data_dir, "polys.pkl"), "rb"))
    no_hole_ids = np.loadtxt(osp.join(data_dir, "no_holes.txt"), dtype=np.int32).tolist()
    dataset_dicts = []
    for idx in tqdm.tqdm(no_hole_ids):
        record = {}
        filename = img_files[idx]
        height, width = 720, 1280

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        bbox, poly = polys[idx]
        # mask = imageio.imread_v2(mask_files[idx])[:, :, 0] > 0
        # contours = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        #     -2
        # ]
        # rows, cols = np.nonzero(mask)
        # poly = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]

        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": poly,
            # "segmentation": mask,
            "category_id": 0,
        }
        # objs.append(obj)
        record["annotations"] = [obj]
        dataset_dicts.append(record)
    return dataset_dicts
