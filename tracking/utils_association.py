import numpy as np

from utils.utils_geometry import (iou_3d_from_corners, box_2d_overlap_union,
                                  tracking_center_distance_2d, tracking_distance_2d_dims, tracking_distance_2d_full)


def iou_bbox_3d_matrix(detections, predictions, detections_dims, predictions_dims):
    return generic_similarity_matrix_two_args(detections, predictions,
                                              detections_dims, predictions_dims, iou_3d_from_corners)


def distance_2d_matrix(centers_0, centers_1):
    return generic_similarity_matrix(centers_0, centers_1, tracking_center_distance_2d)


def distance_2d_dims_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_dims)


def distance_2d_full_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_full)


def iou_bbox_2d_matrix(det_bboxes, seg_bboxes):
    return generic_similarity_matrix(det_bboxes, seg_bboxes, box_2d_overlap_union)


def generic_similarity_matrix(list_0, list_1, similarity_function):
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1)
    return matrix


def generic_similarity_matrix_two_args(list_0, list_1, attrs_0, attrs_1, similarity_function):
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1, attrs_0[i], attrs_1[j])
    return matrix
