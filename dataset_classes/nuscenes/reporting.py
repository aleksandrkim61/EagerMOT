import os
import ujson as json
import time
from typing import IO, Any, Dict, Iterable

import numpy as np
from pyquaternion import Quaternion

from objects.fused_instance import FusedInstance
from dataset_classes.nuscenes.classes import name_from_id
from transform.nuscenes import convert_kitti_bbox_coordinates_to_nu


def build_results_dict(instance: FusedInstance, frame_token: str) -> Dict[str, Any]:
    assert instance.report_mot
    bbox3d_coords = instance.coordinates_3d  # [h, w, l, x, y, z, theta]
    assert bbox3d_coords is not None
    center, wlh, rotation = convert_kitti_bbox_coordinates_to_nu(bbox3d_coords)

    track_dict: Dict[str, Any] = {"sample_token": frame_token}
    track_dict["translation"] = center.tolist()
    track_dict["size"] = wlh.tolist()
    track_dict["rotation"] = rotation.elements.tolist()
    velocity = instance.bbox3d.velocity
    track_dict["velocity"] = list(velocity) if velocity is not None else [1.0, 1.0]
    track_dict["tracking_id"] = str(instance.track_id)
    track_dict["tracking_name"] = name_from_id(instance.class_id)
    track_dict["tracking_score"] = instance.bbox3d.confidence
    track_dict["yaw"] = bbox3d_coords[6]
    return track_dict


def add_results_to_submit(submission: Dict[str, Dict[str, Any]], frame_token: str,
                      predicted_instances: Iterable[FusedInstance]) -> None:
    assert frame_token not in submission["results"], submission["results"][frame_token]
    submission["results"][frame_token] = []

    for instance in predicted_instances:
        if instance.report_mot:
            submission["results"][frame_token].append(build_results_dict(instance, frame_token))

    if len(submission["results"][frame_token]) == 0:
        print(f"Nothing tracked for {frame_token}")


def save_to_json_file(submission: Dict[str, Dict[str, Any]],
                             folder_name: str, version: str) -> None:
    print(f"Frames tracked: {len(submission['results'].keys())}")
    results_file = os.path.join(folder_name, (version + "_tracking.json"))
    with open(results_file, 'w') as f:
        json.dump(submission, f, indent=4)
