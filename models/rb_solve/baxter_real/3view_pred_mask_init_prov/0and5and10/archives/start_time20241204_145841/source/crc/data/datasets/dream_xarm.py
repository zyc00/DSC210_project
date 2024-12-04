import cv2
import glob
import json
import os.path as osp
import matplotlib.pyplot as plt

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from dl_ext.timer import EvalTime
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
import crc.data.transforms.dream_transforms as dream_image_proc


class DreamXArmDataset(TorchDataset):
    def __init__(self, data_dir, split, cfg, transforms=None, ds_len=-1):
        self.data_dir = osp.join(data_dir, split)
        self.network_input_resolution = (400, 400)
        self.network_output_resolution = (208, 208)
        # self.augment_data = split == "train"
        self.augment_data = False  # todo:
        print("augment_data", self.augment_data)
        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )

        self.tensor_from_image_tform = TVTransforms.Compose(
            [
                TVTransforms.ToTensor(),
                TVTransforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                ),
            ]
        )

        # self.image_preprocessing = 'shrink-and-crop' if split == 'train' else 'resize'
        self.image_preprocessing = 'resize'  # todo debug
        print("image_preprocessing", self.image_preprocessing)
        self.color_files = sorted(glob.glob(osp.join(self.data_dir, "color", "*.png")))
        self.joint_position = json.load(open(osp.join(self.data_dir, "joint_position.json"), "r"))
        if ds_len > 0:
            self.color_files = self.color_files[:ds_len]
        # print()

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, index):
        evaltime = EvalTime(disable=True)
        evaltime("", False)
        image_rgb_path = self.color_files[index]
        image_rgb_raw = PILImage.open(image_rgb_path)
        evaltime("open", False)
        # image_rgb_raw = image_rgb_raw.convert("RGB")
        # evaltime("convert RGB", False)
        imgid = image_rgb_path.split("/")[-1].replace(".png", "")
        # Extract keypoints from the json file
        keypoints = self.joint_position[int(imgid)]
        evaltime("joint position", False)

        # plt.imshow(image_rgb_raw)
        # proj = np.array(keypoints['pts_in_img'])
        # plt.scatter(proj[:, 0], proj[:, 1])
        # plt.show()
        # print()
        image_raw_resolution = image_rgb_raw.size

        # Do image preprocessing, including keypoint conversion
        image_rgb_before_aug = dream_image_proc.preprocess_image(
            image_rgb_raw, self.network_input_resolution, self.image_preprocessing
        )
        kp_projs_before_aug = dream_image_proc.convert_keypoints_to_netin_from_raw(
            keypoints["pts_in_img"],
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing,
        )
        evaltime("preprocess", False)

        # Handle data augmentation
        if self.augment_data:
            augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            data_to_aug = {
                "image": np.array(image_rgb_before_aug),
                "keypoints": kp_projs_before_aug,
            }
            augmented_data = augmentation(**data_to_aug)
            image_rgb_net_input = PILImage.fromarray(augmented_data["image"])
            kp_projs_net_input = augmented_data["keypoints"]
        else:
            image_rgb_net_input = image_rgb_before_aug
            kp_projs_net_input = kp_projs_before_aug
        evaltime("augment", False)
        assert (
                image_rgb_net_input.size == self.network_input_resolution
        ), "Expected resolution for image_rgb_net_input to be equal to specified network input resolution, but they are different."

        # Now convert keypoints at network input to network output for use as the trained label
        kp_projs_net_output = dream_image_proc.convert_keypoints_to_netout_from_netin(
            kp_projs_net_input,
            self.network_input_resolution,
            self.network_output_resolution,
        )

        # Convert to tensor for output handling
        # This one goes through image normalization (used for inference)
        image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
            image_rgb_net_input
        )

        # This one is not (used for net input overlay visualizations - hence "viz")
        image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
            image_rgb_net_input
        )

        # Convert keypoint data to tensors - use float32 size
        keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
            np.array(keypoints["pts_in_cam"])
        ).float()
        kp_projs_net_output_as_tensor = torch.from_numpy(
            np.array(kp_projs_net_output)
        ).float()

        # Construct output sample
        sample = {
            "original_image": np.array(image_rgb_raw),
            "image_rgb_input": image_rgb_net_input_as_tensor,
            "keypoint_projections_output": kp_projs_net_output_as_tensor,
            "keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
            # "config": datum,
        }

        # Generate the belief maps directly
        belief_maps = dream_image_proc.create_belief_map(
            self.network_output_resolution, kp_projs_net_output_as_tensor
        )
        evaltime("create_belief_map", False)
        belief_maps_as_tensor = torch.tensor(belief_maps).float()
        sample["belief_maps"] = belief_maps_as_tensor
        evaltime("io", False)
        return sample


def main():
    from crc.engine.defaults import setup
    from crc.engine.defaults import default_argument_parser
    from crc.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/dream/resnet_h.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg, is_train=True).dataset
    sample = ds[0]
    rgb = sample['image_rgb_input']
    raw_rgb = rgb.permute(1, 2, 0) * 0.5 + 0.5
    raw_rgb = raw_rgb.numpy()
    raw_rgb = (raw_rgb * 255).astype(np.uint8)
    rgb = cv2.resize(raw_rgb, (208, 208))
    rgb = rgb / 255.0
    belief_maps = sample['belief_maps'].sum(0).clamp(max=1).numpy()[..., None].repeat(3, -1)
    plt.imshow((rgb + belief_maps) / 2)
    plt.show()


if __name__ == '__main__':
    main()
