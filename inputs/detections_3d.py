from __future__ import annotations
import os
import ujson as json
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np

import inputs.utils_io_ab3dmot as ab3dmot_io
from inputs.bbox import Bbox2d, Bbox3d
import inputs.detections_2d as detections_2d
import inputs.utils as utils
import dataset_classes.nuscenes.classes as nu_classes
import dataset_classes.kitti.classes as kitti_classes
from configs.local_variables import KITTI_DATA_DIR, SPLIT, MOUNT_PATH

CLASS_STR_TO_SEG_CLASS = {'Car': 1, 'Pedestrian': 2, 'Cyclist': -1}


# Adapted from AB3DMOT https://github.com/xinshuoweng/AB3DMOT
def _load_detections_ab3dmot(target_seq_name):
    detections_per_frame = []
    for seq_file_list in utils.det_pointrcnn_files_lists_all():
        target_seq_file = [seq_file for seq_file in seq_file_list if ab3dmot_io.fileparts(seq_file)[
            1] == target_seq_name]
        assert len(target_seq_file) == 1
        seq_dets = np.loadtxt(target_seq_file[0], delimiter=',')  # load detections
        for frame_int in range(int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max()) + 1):
            utils.pad_lists_if_necessary(frame_int, [detections_per_frame])
            frame_det = seq_dets[seq_dets[:, 0] == frame_int]
            if len(frame_det) == 0:
                continue

            detections_per_frame[frame_int].extend(parse_pointrcnn_dets_for_frame(frame_det))
    return detections_per_frame


def _load_detections_pointgnn(target_seq_dirs):
    detections_per_frame = []
    for class_dir in target_seq_dirs:
        for file_name in sorted(os.listdir(class_dir)):
            frame_int = int(file_name.split('.')[0])
            utils.pad_lists_if_necessary(frame_int, [detections_per_frame])

            with open(os.path.join(class_dir, file_name)) as f:
                lines = [line.strip() for line in f.readlines()]
            if lines:
                detections_per_frame[frame_int].extend(
                    [parse_pointgnn_det(line.split(' ')) for line in lines if line])
    return detections_per_frame


def _load_detections_motsfison_rrc(target_seq_name, scores, boxes):
    motsfusion_rrc_detections_dir = utils.det_motsfusion_rrc_dir(target_seq_name)
    for file_name in sorted(os.listdir(motsfusion_rrc_detections_dir)):
        scores.append([])
        boxes.append([])
        with open(os.path.join(motsfusion_rrc_detections_dir, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                scores[-1].append(float(line[4]))
                boxes[-1].append((int(round(float(line[0]))), int(round(float(line[1]))),
                                  int(round(float(line[2]))), int(round(float(line[3])))))


def parse_pointrcnn_dets_for_frame(dets_for_frame):
    dets_coordinates = dets_for_frame[:, 7:14]
    reorder = [3, 4, 5, 6, 2, 1, 0]  # from [h, w, l, x, y, z, theta] to [x,y,z,theta,l,w,h]
    dets_coordinates = dets_coordinates[:, reorder]

    alpha_orientation = dets_for_frame[:, -1].reshape((-1, 1))
    other_array = dets_for_frame[:, 1:7]  # det type, x1, y1, x2, y2, score
    additional_info = np.concatenate((alpha_orientation, other_array), axis=1)
    return [Bbox3d.from_pointrcnn(det_coordinates, info=info, det_to_track_seg_class=detections_2d.DET_TO_TRACK_SEG_CLASS)
            for det_coordinates, info in zip(dets_coordinates[:], additional_info[:])]


def parse_pointgnn_det(det_values):
    confidence = float(det_values[15])
    seg_class_id = CLASS_STR_TO_SEG_CLASS[det_values[0]]
    bbox_2d_coordinates = (int(round(float(det_values[4]))), int(round(float(det_values[5]))),
                           int(round(float(det_values[6]))), int(round(float(det_values[7]))))
    bbox_3d_coordinates = np.array(det_values[8:15], dtype=np.float)
    reorder = [3, 4, 5, 6, 2, 1, 0]  # from [h, w, l, x, y, z, theta] to [x,y,z,theta,l,w,h]
    bbox_3d_coordinates = bbox_3d_coordinates[reorder]
    info = np.array([-1, -1, *bbox_2d_coordinates, confidence], dtype=np.float)

    return Bbox3d.from_pointgnn(bbox_3d_coordinates, confidence=confidence, seg_class_id=seg_class_id,
                                bbox_2d_in_cam={"image_02": bbox_2d_coordinates}, info=info)


def confidence_distribution_for_detections(detections_per_frame):
    return [[detection.confidence for detections_list in detections_per_frame
             for detection in detections_list
             if detection.seg_class_id == target_class]
            for target_class in [detections_2d.CAR_CLASS, detections_2d.PED_CLASS]]


def load_detections_centerpoint() -> Dict[str, List[Bbox3d]]:
    filepath = os.path.join(utils.DETECTIONS_CENTER_POINT_NUSCENES, "detections.json")
    print(f"Parsing {filepath}")
    with open(filepath, 'r') as f:
        full_results_json = json.load(f)

    all_detections = full_results_json["results"]
    all_frames_to_bboxes: Dict[str, List[Bbox3d]] = {}
    for frame_token, frame_dets in all_detections.items():
        assert frame_token not in all_frames_to_bboxes
        all_frames_to_bboxes[frame_token] = [Bbox3d.from_nu_det(det) for det in frame_dets
                                             if det["detection_name"] in nu_classes.ALL_NUSCENES_CLASS_NAMES]
    return all_frames_to_bboxes


def load_annotations_kitti(seq_name: str) -> Dict[str, List[Bbox3d]]:
    filepath = os.path.join(KITTI_DATA_DIR, SPLIT, "label_02", f"{seq_name}.txt")
    print(f"Parsing {filepath}")
    with open(filepath) as f:
        content = f.readlines()

    all_frames_to_bboxes: Dict[str, List[Bbox3d]] = defaultdict(list)
    for line in content:
        entries = line.split(" ")
        class_str = entries[2]
        if class_str != "Car" and class_str != "Pedestrian":
            continue

        class_id = kitti_classes.id_from_name(class_str.lower())
        frame_int = int(entries[0])
        track_id = int(entries[1])
        occlusion = int(entries[4])
        bbox_2d_coords = (float(entries[6]), float(entries[7]), float(entries[8]), float(entries[9]))
        bbox_2d_in_cam: Dict[str, Optional[Bbox2d]] = {"image_02": Bbox2d(*bbox_2d_coords)}
        dims_hwl = (float(entries[10]), float(entries[11]), float(entries[12]))
        dims_lwh = (dims_hwl[2], dims_hwl[1], dims_hwl[0])
        center_xyz = (float(entries[13]), float(entries[14]), float(entries[15]))
        rotation_around_y = float(entries[16])

        # 7 elements: (x y z rotation-around-y l w h)
        bbox_3d = Bbox3d(np.array([*center_xyz, rotation_around_y, *dims_lwh]),
                         track_id, confidence=1.0, obs_angle=occlusion, seg_class_id=class_id,
                         bbox_2d_in_cam=bbox_2d_in_cam)
        all_frames_to_bboxes[str(frame_int).zfill(6)].append(bbox_3d)

    return all_frames_to_bboxes


def load_detections_3dop(seq_name: str) -> Dict[str, List[Bbox3d]]:
    filepath = os.path.join(utils.DETECTIONS_3DOP_PATH, seq_name)

    all_frames_to_bboxes: Dict[str, List[Bbox3d]] = defaultdict(list)
    for file_name in sorted(os.listdir(filepath)):
        frame_str = int(file_name.split('.')[0])

        lines = []
        with open(os.path.join(filepath, file_name)) as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            entries = line.split(" ")
            class_str = entries[0]
            if class_str == "car":
                class_id = 1
            elif class_str == "pedestrian":
                class_id = 2
            else:
                continue

            bbox_2d_coords = (float(entries[4]), float(entries[5]), float(entries[6]), float(entries[7]))
            bbox_2d_in_cam = {"image_02": Bbox2d(*bbox_2d_coords)}
            dims_hwl = (float(entries[8]), float(entries[9]), float(entries[10]))
            dims_lwh = (dims_hwl[2], dims_hwl[1], dims_hwl[0])
            center_xyz = (float(entries[11]), float(entries[12]), float(entries[13]))
            rotation_around_y = float(entries[14])
            confidence = float(entries[15])

            # 7 elements: (x y z rotation-around-y l w h)
            bbox_3d = Bbox3d(np.array([*center_xyz, rotation_around_y, *dims_lwh]),
                             obs_angle=-1, confidence=confidence, seg_class_id=class_id,
                             bbox_2d_in_cam=bbox_2d_in_cam)
            all_frames_to_bboxes[str(frame_str).zfill(6)].append(bbox_3d)
    return all_frames_to_bboxes
