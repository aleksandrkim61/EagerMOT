from __future__ import annotations
from typing import Dict, List

import numpy as np

import inputs.utils_io_ab3dmot as ab3dmot_io
import inputs.utils as utils
import inputs.detections_2d as detections_2d
import inputs.detections_3d as detections_3d
from inputs.bbox import Bbox3d
from inputs.detection_2d import Detection2D


def load_segmentations_trackrcnn(target_seq_name, classes_to_load=None):
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_trackrcnn(
        target_seq_name, classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_rrc(target_seq_name, classes_to_load=None):
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(utils.seg_motsfusion_rrc_dir(target_seq_name), target_seq_name,
                                                 classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_trackrcnn(target_seq_name, classes_to_load=None):
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(utils.seg_motsfusion_trackrcnn_dir(target_seq_name), target_seq_name,
                                                 classes, scores, masks, boxes, reids, classes_to_load)
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_motsfusion_best(target_seq_name, classes_to_load=None):
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(utils.seg_motsfusion_rrc_dir(target_seq_name), target_seq_name,
                                                 classes, scores, masks, boxes, reids, [detections_2d.CAR_CLASS])
    # print('loaded MOTSFusion', len(classes), len(classes[0]))
    detections_2d._load_segmentations_motsfusion(utils.seg_motsfusion_trackrcnn_dir(target_seq_name), target_seq_name,
                                                 classes, scores, masks, boxes, reids, [detections_2d.PED_CLASS])
    # print('loaded Both', len(classes), len(classes[0]), len(masks), len(masks[0]))
    # print(masks[0])

    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_segmentations_tracking_best(target_seq_name, classes_to_load=None):
    """ Load 2D detections for "car" given by MOTSFusion and for "ped" given by TrackRCNN """
    classes, scores, masks, boxes, reids = ([] for _ in range(5))
    detections_2d._load_segmentations_motsfusion(utils.seg_motsfusion_rrc_dir(target_seq_name), target_seq_name,
                                                 classes, scores, masks, boxes, reids, [detections_2d.CAR_CLASS])
    detections_2d._load_segmentations_trackrcnn(
        target_seq_name, classes, scores, masks, boxes, reids, [detections_2d.PED_CLASS])
    return utils.convert_nested_lists_to_numpy(classes, scores, masks, boxes, reids)


def load_detections_3d(dets_3d_source: str, seq_name: str) -> Dict[str, List[Bbox3d]]:
    if dets_3d_source == utils.AB3DMOT:
        return detections_3d._load_detections_ab3dmot(seq_name)
    if dets_3d_source == utils.POINTGNN_T3:
        return detections_3d._load_detections_pointgnn(utils.det_pointgnn_t3_dirs(seq_name))
    if dets_3d_source == utils.POINTGNN_T2:
        return detections_3d._load_detections_pointgnn(utils.det_pointgnn_t2_dirs(seq_name))
    if dets_3d_source == utils.STEREO_3DOP:
        return detections_3d.load_detections_3dop(seq_name)
    if dets_3d_source == utils.CENTER_POINT:
        return detections_3d.load_detections_centerpoint()
    raise NotImplementedError


def load_detections_2d_kitti(dets_2d_source: str, seq_name: str):
    if dets_2d_source == utils.TRACKRCNN:
        return load_segmentations_trackrcnn(seq_name)
    elif dets_2d_source == utils.MOTSFUSION_RRC:
        return load_segmentations_motsfusion_rrc(seq_name)
    elif dets_2d_source == utils.MOTSFUSION_TRACKRCNN:
        return load_segmentations_motsfusion_trackrcnn(seq_name)
    elif dets_2d_source == utils.MOTSFUSION_BEST:
        return load_segmentations_motsfusion_best(seq_name)
    elif dets_2d_source == utils.TRACKING_BEST:
        return load_segmentations_tracking_best(seq_name)
    raise NotImplementedError


def load_detections_2d_kitti_new(dets_2d_source: str, seq_name: str) -> Dict[str, Dict[str, List[Detection2D]]]:
    """ Should return a dict mapping frame to each camera with its detections """
    if dets_2d_source == utils.MMDETECTION_CASCADE_NUIMAGES:
        return detections_2d.load_detections_2d_mmdetection_kitti(seq_name)
    raise NotImplementedError


def load_detections_2d_nuscenes(dets_2d_source: str, seq_name: str) -> Dict[str, Dict[str, List[Detection2D]]]:
    """ Should return a dict mapping frame to each camera with its detections """
    if dets_2d_source == utils.EFFICIENT_DET:
        return detections_2d.load_detections_2d_efficient_det(seq_name)
    if dets_2d_source == utils.MMDETECTION_CASCADE_NUIMAGES:
        return detections_2d.load_detections_2d_mmdetection_nuscenes(seq_name)
    raise NotImplementedError


def load_annotations_kitti(seq_name: str) -> Dict[str, List[Bbox3d]]:
    return detections_3d.load_annotations_kitti(seq_name)
