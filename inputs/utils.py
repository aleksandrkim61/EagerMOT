import os
import ujson as json
from typing import Optional, Mapping, Dict

import numpy as np

import inputs.utils_io_ab3dmot as ab3dmot_io
from configs.local_variables import MOUNT_PATH, SPLIT
from dataset_classes.nuscenes.classes import id_from_name as id_nuscenes_from_name
from dataset_classes.kitti.classes import id_from_name as id_kitti_from_name

AB3DMOT = 'ab3dmot'
POINTGNN_T3 = 'pointgnn_t3'
POINTGNN_T2 = 'pointgnn_t2'
_SPLITS_POINTGNN_T3 = ['results_tracking_car_auto_t3_trainval', 'results_tracking_ped_cyl_auto_trainval']
_SPLITS_POINTGNN_T2 = ['results_tracking_car_auto_t2_train', 'results_tracking_ped_cyl_auto_trainval']
CENTER_POINT = 'center_point'
EFFICIENT_DET = 'efficient_det'
STEREO_3DOP = '3dop'

TRACKRCNN = 'trackrcnn'
MOTSFUSION_RRC = 'motsfusion_rrc'
MOTSFUSION_TRACKRCNN = 'motsfusion_trackrcnn'
MOTSFUSION_BEST = 'motsfusion_best'
TRACKING_BEST = 'rrc_trackrcnn'
MMDETECTION_CASCADE_NUIMAGES = 'mmdetection_cascade_nuimages'

# Edit paths to point to where you store 3D and 2D detections / segmentations on your machine
POINTGNN_DETS_DIR = MOUNT_PATH + "/storage/pointgnn/" + SPLIT
DETECTIONS_AB3DMOT = MOUNT_PATH + "/storage/ab3dmot/" + SPLIT
DETECTIONS_EFFICIENT_DET_NUSCENES = MOUNT_PATH + "/storage/efficientdet/" + SPLIT
DETECTIONS_CENTER_POINT_NUSCENES = MOUNT_PATH + "/storage/centerpoint/" + SPLIT

SEGMENTATIONS_TRACKRCNN_DIR = MOUNT_PATH + "/storage/trackrcnn/" + SPLIT
MOTSFUSION_RRC_DIR = MOUNT_PATH + "/storage/detections_segmentations_RRC_BB2SegNet/" + SPLIT
MOTSFUSION_TRACKRCNN_DIR = MOUNT_PATH + "/storage/detections_segmentations_trackrcnn_BB2SegNet/" + SPLIT
DETECTIONS_MMDETECTION_CASCADE_NUIMAGES_NUSCENES = MOUNT_PATH + \
    "/storage/mmdetection_cascade_x101/" + SPLIT
DETECTIONS_MMDETECTION_CASCADE_NUIMAGES_KITTI = MOUNT_PATH + \
    "/storage/mmdetection_cascade_x101_kitti/" + SPLIT

########################################3333

_SPLITS_AB3DMOT = ['car_3d_det', 'ped_3d_det']
DETECTIONS_MOTSFUSION_RRC_DIR = MOTSFUSION_RRC_DIR + "/detections"
SEGMENTATIONS_MOTSFUSION_RRC_DIR = MOTSFUSION_RRC_DIR + "/segmentations"
DETECTIONS_MOTSFUSION_TRACKRCNN_DIR = MOTSFUSION_TRACKRCNN_DIR + "/detections_segmentations_trackrcnn"
SEGMENTATIONS_MOTSFUSION_TRACKRCNN_DIR = MOTSFUSION_TRACKRCNN_DIR + "/segmentations_trackrcnn"

DETECTIONS_3DOP_PATH = MOUNT_PATH + "/storage/3dop/" + SPLIT


def parse_and_add_seg_for_frame(frame, current_seg, classes_to_load, parse_func,
                                classes, scores, masks, boxes, reids):
    cur_class, score, mask, box, reid = parse_func(current_seg)
    if classes_to_load is None or cur_class in classes_to_load:
        classes[frame].append(cur_class)
        scores[frame].append(score)
        masks[frame].append(mask)
        boxes[frame].append(box)
        reids[frame].append(reid)


def pad_lists_if_necessary(frame_num, collection_of_lists):
    while len(collection_of_lists[0]) < frame_num + 1:
        for lst in collection_of_lists:
            lst.append([])


def convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids):
    for t in range(len(classes)):
        classes[t] = np.array(classes[t])
        scores[t] = np.array(scores[t])
        masks[t] = np.array(masks[t])
        boxes[t] = np.vstack(boxes[t]) if len(classes[t]) > 0 else np.array([])
        reids[t] = np.array(reids[t])
    return boxes, scores, reids, classes, masks


def seg_trackrcnn_dir(target_seq_name):
    return os.path.join(SEGMENTATIONS_TRACKRCNN_DIR, '%s.txt' % target_seq_name)


def det_motsfusion_rrc_dir(target_seq_name):
    return os.path.join(DETECTIONS_MOTSFUSION_RRC_DIR, target_seq_name)


def seg_motsfusion_rrc_dir(target_seq_name):
    return os.path.join(SEGMENTATIONS_MOTSFUSION_RRC_DIR, target_seq_name)


def det_motsfusion_trackrcnn_dir(target_seq_name):
    return os.path.join(DETECTIONS_MOTSFUSION_TRACKRCNN_DIR, '%s.txt' % target_seq_name)


def seg_motsfusion_trackrcnn_dir(target_seq_name):
    return os.path.join(SEGMENTATIONS_MOTSFUSION_TRACKRCNN_DIR, target_seq_name)


def det_pointgnn_t3_dirs(target_seq_name):
    return [os.path.join(POINTGNN_DETS_DIR, split, target_seq_name, 'data') for split in _SPLITS_POINTGNN_T3]


def det_pointgnn_t2_dirs(target_seq_name):
    return [os.path.join(POINTGNN_DETS_DIR, split, target_seq_name, 'data') for split in _SPLITS_POINTGNN_T2]


def det_pointrcnn_files_lists_all():
    return [ab3dmot_io.load_list_from_folder(os.path.join(DETECTIONS_AB3DMOT, split_file))[0] for split_file in _SPLITS_AB3DMOT]


coco_class_id_mapping: Mapping[int, int] = {
    1: id_nuscenes_from_name("pedestrian"),
    2: id_nuscenes_from_name("bicycle"),
    3: id_nuscenes_from_name("car"),
    4: id_nuscenes_from_name("motorcycle"),
    6: id_nuscenes_from_name("bus"),
    8: id_nuscenes_from_name("truck"),
}


mmdetection_nuimages_class_mapping_nuscenes: Mapping[str, int] = {
    "0": id_nuscenes_from_name("pedestrian"),
    "3": id_nuscenes_from_name("bicycle"),
    "4": id_nuscenes_from_name("bus"),
    "5": id_nuscenes_from_name("car"),
    "7": id_nuscenes_from_name("motorcycle"),
    "8": id_nuscenes_from_name("trailer"),
    "9": id_nuscenes_from_name("truck"),
}


mmdetection_nuimages_class_mapping_kitti: Mapping[str, int] = {
    "0": id_kitti_from_name("pedestrian"),
    "5": id_kitti_from_name("car"),
}


def load_json_for_sequence(folder_path: str, target_seq_name: str) -> Dict:
    folder_dir = os.path.join(folder_path)
    assert os.path.isdir(folder_dir)

    # Parse sequences
    filepath = None
    for scene_json in os.listdir(folder_dir):
        if scene_json.startswith(target_seq_name):
            filepath = os.path.join(folder_dir, scene_json)
            break
    else:
        raise NotADirectoryError(f"No detections for {target_seq_name}")

    print(f"Parsing {filepath}")
    with open(filepath, 'r') as f:
        all_detections = json.load(f)
    return all_detections