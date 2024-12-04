import io
import os
import os.path as osp
import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch
from PIL import Image

from crc.utils import plt_utils
from crc.utils.pn_utils import random_choice


class SAMAPI:
    predictor = None

    @staticmethod
    def get_instance(sam_checkpoint=None):
        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = "third_party/segment_anything/sam_vit_h_4b8939.pth"
            if not osp.exists(sam_checkpoint):
                os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P third_party/segment_anything')
            device = "cuda"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def destory():
        del SAMAPI.predictor
        SAMAPI.predictor = None
        torch.cuda.empty_cache()

    @staticmethod
    def segment_api(rgb, masks=None, bbox=None, point_center=False, random_points=False,
                    sam_checkpoint=None, box_scale=1.0, dbg=False):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        masks: np.ndarray h,w bool
        dbg

        Returns
        -------

        """
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if masks is None and bbox is None:
            box_input = None
            point_coords = None
            point_labels = None
        else:
            # mask to bbox
            if masks.ndim == 2:
                masks = masks[None]
            mask = masks.sum(0).clip(0, 1).astype(np.uint8)
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
                if dbg:
                    print("box xyxy", x1, y1, x2, y2)
            else:
                x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            x1 -= w * (box_scale - 1) / 2
            x2 += w * (box_scale - 1) / 2
            y1 -= h * (box_scale - 1) / 2
            y2 += h * (box_scale - 1) / 2
            box_input = np.array([[x1, y1, x2, y2]])
            box_input[0][0] = np.clip(box_input[0][0], 0, rgb.shape[1] - 1)
            box_input[0][1] = np.clip(box_input[0][1], 0, rgb.shape[0] - 1)
            box_input[0][2] = np.clip(box_input[0][2], 0, rgb.shape[1] - 1)
            box_input[0][3] = np.clip(box_input[0][3], 0, rgb.shape[0] - 1)
            if point_center is True and masks is not None:
                point_coords = []
                point_labels = []
                for m in masks:
                    yy1, yy2, xx1, xx2 = np.nonzero(m)[0].min(), np.nonzero(m)[0].max(), np.nonzero(m)[1].min(), \
                                         np.nonzero(m)[1].max()
                    pc = np.array([[(xx1 + xx2) / 2, (yy1 + yy2) / 2]])
                    pl = np.array([m[int((yy1 + yy2) / 2), int((xx1 + xx2) / 2)]])
                    if dbg:
                        print("point & label", pc, pl)
                    point_coords.append(pc)
                    point_labels.append(pl)
                point_coords = np.concatenate(point_coords, axis=0)
                point_labels = np.concatenate(point_labels, axis=0)
            elif random_points is True and masks is not None:
                ys, xs = mask.nonzero()
                _, idxs = random_choice(xs, size=200, dim=0, replace=False)
                point_coords = np.stack([xs[idxs], ys[idxs]], axis=1)
                point_labels = np.ones((point_coords.shape[0],), dtype=np.int64)
            else:
                point_coords = None
                point_labels = None
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            # mask_input=None,
            multimask_output=True,
            return_logits=False,
        )
        maxidx = np.argmax(scores)
        mask = masks[maxidx]
        if dbg:
            plt.subplot(1, 2, 1)
            plt.imshow(rgb)
            if box_input is not None:
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2))
            if point_coords is not None:
                plt.gca().scatter(point_coords[:, 0], point_coords[:, 1], c='r', s=10)
            plt.subplot(1, 2, 2)
            plt.title("sam output")
            plt.imshow(plt_utils.vis_mask(rgb, mask.astype(np.uint8), [0, 255, 0]))
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            plt.close("all")
            return mask, Image.open(img_buf)
        return mask


def main():
    alpha_paths = sorted(
        glob.glob("/home/linghao/Datasets/oppo/post_process_egg_colmap_rename_dbg/output/alphas/*.png"))
    fg_img_paths = sorted(glob.glob(
        "/home/linghao/Datasets/oppo/post_process_egg_colmap_rename_dbg/segmentation/images_undistorted/*.jpg"))
    for alpha_path, fg_img_path in zip(alpha_paths, fg_img_paths):
        alpha = imageio.imread_v2(alpha_path)
        fg_img = imageio.imread_v2(fg_img_path)
        mask = SAMAPI.segment_api(fg_img, alpha > 0, dbg=True)
    print()


if __name__ == '__main__':
    main()
