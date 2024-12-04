import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import crc.data.transforms.dream_transforms as dream_image_proc


class ResnetSimple(nn.Module):
    def __init__(
            self, n_keypoints=7, pretrained=True, full=False,
    ):
        super(ResnetSimple, self).__init__()
        net = torchvision.models.resnet101(pretrained=pretrained)
        self.full = full
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # upconvolution and final layer
        BN_MOMENTUM = 0.1
        if not full:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            # This brings it up from 208x208 to 416x416
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)

        if self.full:
            x = self.upsample2(x)

        return [x]


class DreamNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.dream.resnet
        self.model = ResnetSimple(
            n_keypoints=self.cfg.n_keypoints,
            pretrained=self.cfg.pretrained,
            full=self.cfg.full,
        )  # expect 330 params

        self.criterion = torch.nn.MSELoss()

    def forward(self, dps):
        image_rgb_input = dps['image_rgb_input']
        network_output_heads = self.model(image_rgb_input)
        target = dps['belief_maps']
        loss = self.criterion(network_output_heads[0], target)
        loss_dict = {"belief_map_loss": loss}
        if self.training:
            outputs = {"network_output_heads": network_output_heads}
        else:
            belief_maps_batch = network_output_heads[0]
            kpts_det = []
            for belief_maps in belief_maps_batch:
                if self.cfg.old_process:
                    peaks = dream_image_proc.peaks_from_belief_maps(
                        belief_maps, offset_due_to_upsampling=0.4395
                    )
                    detected_kp_projs = []
                    for peak in peaks:
                        if len(peak) == 1:
                            detected_kp_projs.append([peak[0][0], peak[0][1]])
                        else:
                            if len(peak) > 1:
                                # Try to use the belief map scores
                                peak_sorted_by_score = sorted(
                                    peak, key=lambda x: x[2], reverse=True
                                )
                                belief_peak_next_best_score = 0.25
                                if (
                                        peak_sorted_by_score[0][2] - peak_sorted_by_score[1][2]
                                        >= belief_peak_next_best_score
                                ):
                                    # Keep the best score
                                    detected_kp_projs.append(
                                        [
                                            peak_sorted_by_score[0][0],
                                            peak_sorted_by_score[0][1],
                                        ]
                                    )
                                else:
                                    # Can't determine -- return no detection
                                    # Can't use None because we need to return it as a tensor
                                    detected_kp_projs.append([-999.999, -999.999])

                            else:
                                # Can't determine -- return no detection
                                # Can't use None because we need to return it as a tensor
                                detected_kp_projs.append([-999.999, -999.999])
                    detected_kp_projs = torch.tensor(detected_kp_projs)
                else:
                    pts2d, conf = hrnet_get_max_preds(belief_maps[None].cpu().numpy())
                    pts2d = pts2d[0]
                    conf = conf[0]
                    pts2d[pts2d < 0.0001] = -999.999
                    detected_kp_projs = torch.tensor(pts2d)
                network_output_res_inf = (208, 208)
                network_input_res_inf = (400, 400)
                image_raw_resolution = (1280, 720)
                image_preprocessing = "resize"
                this_detected_kps_netin = dream_image_proc.convert_keypoints_to_netin_from_netout(
                    detected_kp_projs,
                    network_output_res_inf,
                    network_input_res_inf,
                )
                this_detected_kps_raw = dream_image_proc.convert_keypoints_to_raw_from_netin(
                    this_detected_kps_netin,
                    network_input_res_inf,
                    image_raw_resolution,
                    image_preprocessing,
                )
                kpts_det.append(this_detected_kps_raw)
            outputs = {"detected_kps": kpts_det}
            # loss_dict = {}
            if self.total_cfg.dbg is True:
                raw_img = image_rgb_input[0].cpu().numpy().transpose(1, 2, 0)
                raw_img = raw_img * 0.5 + 0.5
                raw_img = (raw_img * 255).astype(np.uint8)
                raw_img = cv2.resize(raw_img, (208, 208))
                raw_img = raw_img / 255
                gt_map = dps['belief_maps'][0].sum(0).clamp(max=1).cpu().numpy()[..., None].repeat(3, axis=2)
                plt.title("gt map")
                plt.imshow((raw_img + gt_map) / 2)
                plt.show()
                plt.title("pred kpts")
                plt.imshow(dps['original_image'][0].cpu())
                plt.scatter(kpts_det[0][:, 0], kpts_det[0][:, 1])
                plt.show()
                print()
        return outputs, loss_dict


def hrnet_get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
