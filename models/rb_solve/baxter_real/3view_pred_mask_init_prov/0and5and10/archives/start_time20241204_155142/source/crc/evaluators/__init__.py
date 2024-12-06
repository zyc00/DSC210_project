from ..registry import EVALUATORS
from .build import *
from .baxter_pck import *
from .baxter_iter_ee_pck import *
from .dream_panda_test import *


def build_evaluators(cfg):
    evaluators = []
    for e in cfg.test.evaluators:
        evaluators.append(EVALUATORS[e](cfg))
    return evaluators
