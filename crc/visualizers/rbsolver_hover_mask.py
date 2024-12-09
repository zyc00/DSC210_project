import imageio
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from crc.registry import VISUALIZERS
from crc.utils import plt_utils
from crc.utils.plt_utils import image_grid
from crc.utils.vis3d_ext import Vis3D


@VISUALIZERS.register('rbsolver_hover_mask')
def rbsolver_hover_mask(cfg):
    def f(*args, **kwargs):
        vis_dir = osp.join(cfg.output_dir, 'visualization', cfg.datasets.test)
        os.makedirs(vis_dir, exist_ok=True)
        outputs = args[0]
        # print('tsfm', np.array2string(outputs[0]['tsfm'].numpy(), separator=',', suppress_small=True))
        if "rendered_masks" in outputs[0]:
            rendered_masks = outputs[0]['rendered_masks'].numpy()
            trainer = args[1]
            dl = trainer.valid_dl
            ds = dl.dataset
            tmps = []
            for i in range(len(ds)):
                data_dict = ds[i]
                image = data_dict['rgb']
                rendered_mask = rendered_masks[i] > 0
                tmp = plt_utils.vis_mask(image, rendered_mask.astype(np.uint8), [255, 0, 0])
                imageio.imwrite(f'{i:06d}_rendered_mask.png', tmp)
                tmps.append(tmp)
            image_grid(tmps, show=False)
            plt.savefig('grid_rendered_masks.png')

    return f


@VISUALIZERS.register('vis_realman_head_to_base')
def vis_realman_head_to_base(cfg):
    def f(*args, **kwargs):
        vis_dir = osp.join(cfg.output_dir, 'visualization', cfg.datasets.test)
        os.makedirs(vis_dir, exist_ok=True)
        outputs = args[0]
        Tc_c2b = outputs[0]['tsfm'].numpy()
        print('tsfm', np.array2string(Tc_c2b, separator=',', suppress_small=True))
        wis3d = Vis3D(sequence_name="vis_realman_head_to_base")
        wis3d.add_realman()
        wis3d.add_camera_pose(np.linalg.inv(Tc_c2b))

    return f
