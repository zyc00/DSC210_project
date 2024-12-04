from ..registry import VISUALIZERS
from .rbsolver_hover_mask import rbsolver_hover_mask
from .rbsolver_hover_mask import vis_realman_head_to_base


def build_visualizer(cfg):
    print(VISUALIZERS[cfg.test.visualizer](cfg))
    return VISUALIZERS[cfg.test.visualizer](cfg)
