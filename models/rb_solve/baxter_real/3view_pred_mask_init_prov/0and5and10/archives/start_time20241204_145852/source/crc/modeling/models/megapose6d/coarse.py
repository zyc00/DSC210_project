from torch import nn

from crc.modeling.models.megapose6d.torchvision_resnet import resnet34


class MegaPose6D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.megapose6d
        self.backbone = resnet34(num_classes=512, n_input_channels=6)
        self.views_logits_head = nn.Linear(512, 1, bias=True)

    def forward(self, dps):
        print()
