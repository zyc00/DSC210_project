import matplotlib.pyplot as plt
import json
import os.path as osp
from pathlib import Path
from typing import List

import imageio
import pandas as pd
import torch
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
)
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection

from crc.utils import render_api


def load_object_data(data_path: str) -> List[ObjectData]:
    object_data = [ObjectData.from_json(d) for d in json.load(open(data_path))]
    return object_data


def load_detections(
        label,
        bbox,
) -> DetectionsType:
    """
    Load detections from a directory containing the detections file.
    params: example_dir: str, label: str, bbox: list
    bbox: [x1, y1, x2, y2]
    """
    infos = pd.DataFrame(dict(
        label=[label],
        batch_im_id=0,
        instance_id=0))
    bboxes = torch.tensor(bbox)[None].float()
    return PandasTensorCollection(infos=infos, bboxes=bboxes)


def make_object_dataset(mesh_path: str, label) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    rigid_objects.append(RigidObject(label=label, mesh_path=Path(mesh_path), mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def run_inference(
        model_name: str,
        mesh_path: str,
        rgb,
        K,
        bbox
) -> None:
    observation = ObservationTensor.from_numpy(rgb, K=K).cuda()
    model_info = NAMED_MODELS[model_name]

    detections = load_detections("barbecue-sauce", bbox, ).cuda()
    object_dataset = make_object_dataset(mesh_path, "barbecue-sauce")

    pose_estimator = load_named_model(model_name, object_dataset,
                                      model_dir="./megapose6d/local_data").cuda()

    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"])
    # print(output)
    print(output.poses_input)
    # pose = output.poses_input.to("cpu").numpy()
    return pose


def main():
    data_dir = "./megapose6d/local_data/examples/barbecue-sauce"
    mesh_path = osp.join(data_dir, "meshes/barbecue-sauce/hope_000002.ply")
    rgb = imageio.imread_v2(osp.join(data_dir, "image_rgb.png"))
    K = json.load(open(osp.join(data_dir, "camera_data.json")))["K"]
    pose = run_inference("megapose-1.0-RGB-multi-hypothesis", mesh_path, rgb, K,
                         bbox=[384, 234, 522, 455])
    # print(pose.shape)
    # mask = render_api.nvdiffrast_render_mesh_api(trimesh.load_mesh(mesh_path), pose, 480, 640, K)
    # print(mask.shape)
    # rgb = rgb / 255.0
    # rgb[mask > 0] *= 0.5
    # rgb[mask > 0] += 0.5
    # plt.imshow(rgb)
    # plt.show()


if __name__ == "__main__":
    main()
