import torch
import os
import io
# import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from loguru import logger
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

from crc.trainer.base import BaseTrainer
from crc.trainer.utils import *
from crc.utils import plt_utils


class HandEyeSolverTrainer(BaseTrainer):

    def save(self, epoch):
        if self.save_mode == 'epoch':
            name = os.path.join(self.output_dir, 'model_epoch_%06d.pth' % epoch)
        else:
            name = os.path.join(self.output_dir, 'model_iteration_%06d.pth' % self.global_steps)
        net_sd = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        if self.cfg.solver.pop_verts_faces:
            net_sd = {k: v for k, v in net_sd.items() if 'vert' not in k and 'faces' not in k}
        if self.cfg.solver.compress_history_ops and "history_ops" in net_sd:
            keep = (net_sd["history_ops"] != 0).any(dim=1)
            net_sd["history_ops"] = net_sd["history_ops"][keep]
        d = {'model': net_sd,
             'epoch': epoch,
             'best_val_loss': self.best_val_loss,
             'global_steps': self.global_steps}
        if self.cfg.solver.save_optimizer:
            d['optimizer'] = self.optimizer.state_dict()
        if self.cfg.solver.save_scheduler:
            d['scheduler'] = self.scheduler.state_dict()
        torch.save(d, name)
