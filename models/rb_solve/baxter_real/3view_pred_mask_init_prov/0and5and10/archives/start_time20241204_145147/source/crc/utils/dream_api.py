import PIL.Image as Image
import sys

import numpy as np
import torch
import os

from ruamel.yaml import YAML

sys.path.append("./DREAM")

import dream


class DreamApiHelper:
    _network = None

    @staticmethod
    def get_model(ckpt_path):
        if DreamApiHelper._network is None:
            input_config_path = os.path.splitext(ckpt_path)[0] + ".yaml"
            data_parser = YAML(typ="safe")

            with open(input_config_path, "r") as f:
                network_config = data_parser.load(f)

            network_config["training"]["platform"]["gpu_ids"] = None

            dream_network = dream.create_network_from_config_data(network_config)

            dream_network.model.load_state_dict(torch.load(ckpt_path))
            dream_network.enable_evaluation()
            DreamApiHelper._network = dream_network
        return DreamApiHelper._network


def dream_api(image_path, ckpt_path):
    dream_network = DreamApiHelper.get_model(ckpt_path)

    image_rgb_OrigInput_asPilImage = Image.open(image_path).convert("RGB")

    image_preprocessing = dream_network.image_preprocessing()

    detection_result = dream_network.keypoints_from_image(image_rgb_OrigInput_asPilImage,
                                                          image_preprocessing_override=image_preprocessing,
                                                          debug=True, )
    kpt2d = detection_result["detected_keypoints"]
    return kpt2d


def main():
    img_path = "data/sim_for_hec/version2_no_marker_predinit_se/000006/color/000000.png"
    dream_api(img_path, "../DREAM/models/xarm/best_network.pth")


if __name__ == '__main__':
    main()
