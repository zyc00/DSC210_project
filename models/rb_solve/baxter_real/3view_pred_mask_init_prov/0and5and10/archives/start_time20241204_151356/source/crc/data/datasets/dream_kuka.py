import tqdm
import cv2
import os.path as osp
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as TVTransforms
import crc.data.transforms.dream_transforms as dream_image_proc

import albumentations as albu


class DreamKukaDataset(Dataset):
    def __init__(self, data_dir, split, cfg, transforms=None, ds_len=-1):
        self.data_dir = data_dir
        self.network_input_resolution = (400, 400)
        self.network_output_resolution = (208, 208)
        self.augment_data = split == "train"
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

        self.image_preprocessing = 'shrink-and-crop' if split == 'train' else 'resize'
        # self.image_preprocessing = 'resize'  # todo debug
        # print("image_preprocessing", self.image_preprocessing)
        self.color_files = sorted(glob.glob(osp.join(self.data_dir, "*.jpg")))
        self.anno_files = sorted(glob.glob(osp.join(self.data_dir, "*.json")))
        # self.joint_position = json.load(open(osp.join(self.data_dir, "joint_position.json"), "r"))
        if ds_len > 0:
            self.color_files = self.color_files[:ds_len]
        print()

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, index):
        from crc.data.datasets import dream_utilities
        image_rgb_path = self.color_files[index]

        # Extract keypoints from the json file
        data_path = self.anno_files[index]
        keypoint_names = ['iiwa7_link_0', 'iiwa7_link_1', 'iiwa7_link_2', 'iiwa7_link_3', 'iiwa7_link_4', 'iiwa7_link_5', 'iiwa7_link_6', 'iiwa7_link_7']
        keypoints = dream_utilities.load_keypoints(
            data_path, "kuka", keypoint_names
        )

        # Load image and transform to network input resolution -- pre augmentation
        image_rgb_raw = Image.open(image_rgb_path).convert("RGB")
        # assert image_rgb_raw.mode == "RGB", "Expected RGB image"
        image_raw_resolution = image_rgb_raw.size

        # Do image preprocessing, including keypoint conversion
        image_rgb_before_aug = dream_image_proc.preprocess_image(
            image_rgb_raw, self.network_input_resolution, self.image_preprocessing
        )
        kp_projs_before_aug = dream_image_proc.convert_keypoints_to_netin_from_raw(
            keypoints["projections"],
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing,
        )

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
            image_rgb_net_input = Image.fromarray(augmented_data["image"])
            kp_projs_net_input = augmented_data["keypoints"]
        else:
            image_rgb_net_input = image_rgb_before_aug
            kp_projs_net_input = kp_projs_before_aug

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
            np.array(keypoints["positions_wrt_cam"])
        ).float()
        kp_projs_net_output_as_tensor = torch.from_numpy(
            np.array(kp_projs_net_output)
        ).float()

        # Construct output sample
        sample = {
            "image_rgb_input": image_rgb_net_input_as_tensor,
            "keypoint_projections_output": kp_projs_net_output_as_tensor,
            "keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
        }

        # Generate the belief maps directly
        belief_maps = dream_image_proc.create_belief_map(
            self.network_output_resolution, kp_projs_net_output_as_tensor
        )
        belief_maps_as_tensor = torch.tensor(belief_maps).float()
        sample["belief_maps"] = belief_maps_as_tensor

        return sample


def main():
    from crc.engine.defaults import setup
    from crc.engine.defaults import default_argument_parser
    from crc.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/dream/resnet_h_kuka.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg, is_train=True).dataset
    for d in tqdm.tqdm(ds):
        pass
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
