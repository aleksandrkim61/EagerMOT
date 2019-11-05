import time
from typing import List, Iterable, Tuple, Sequence, Mapping, Dict, Set, Any
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from inputs.bbox import Bbox2d, Bbox3d, ProjectsToCam
from inputs.detection_2d import Detection2D
from utils.utils_geometry import convert_bbox_coordinates_to_corners, box_2d_area
from tracking.utils_association import (iou_bbox_2d_matrix, iou_bbox_3d_matrix,
                                        distance_2d_matrix, distance_2d_dims_matrix, distance_2d_full_matrix)
from tracking.tracks import Track
from objects.fused_instance import FusedInstance


CamDetectionIndices = Tuple[str, int]


def associate_instances_to_tracks_3d_iou(detected_instances, tracks, params: Mapping):
    """
    Assigns detected_objects to tracked objects
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(tracks) == 0:  # original association
        return np.empty((0, 2), dtype=int), list(range(len(detected_instances))), []
    if len(detected_instances) == 0:  # nothing detected in the current frame
        return np.empty((0, 2), dtype=int), [], list(range(len(tracks)))

    track_coordinates = [track.predict_motion()[:7] for track in tracks]
    track_classes = [track.class_id for track in tracks]

    if params['first_matching_method'] == 'iou_3d':
        detected_corners = [instance.bbox3d.corners_3d for instance in detected_instances]
        tracks_corners = [convert_bbox_coordinates_to_corners(state) for state in track_coordinates]
        detections_dims = [instance.bbox3d.kf_coordinates[4:7] for instance in detected_instances]
        tracks_dims = [state[4:7] for state in track_coordinates]
        matrix_3d_sim = iou_bbox_3d_matrix(detected_corners, tracks_corners, detections_dims, tracks_dims)
    elif params['first_matching_method'] == "dist_2d":
        detected_centers = [instance.bbox3d.kf_coordinates[:3] for instance in detected_instances]
        track_centers = [state[:3] for state in track_coordinates]
        matrix_3d_sim = distance_2d_matrix(detected_centers, track_centers)
        matrix_3d_sim *= -1
    elif params['first_matching_method'] == "dist_2d_dims":
        detected_coordinates = [instance.bbox3d.kf_coordinates for instance in detected_instances]
        matrix_3d_sim = distance_2d_dims_matrix(detected_coordinates, track_coordinates)
        matrix_3d_sim *= -1
    elif params['first_matching_method'] == "dist_2d_full":
        detected_coordinates = [instance.bbox3d.kf_coordinates for instance in detected_instances]
        matrix_3d_sim = distance_2d_full_matrix(detected_coordinates, track_coordinates)
        matrix_3d_sim *= -1

    matched_indices, unmatched_det_ids, unmatched_track_ids = \
        perform_association_from_similarity(len(detected_instances), len(tracks), matrix_3d_sim)
    return filter_matches(matched_indices, unmatched_det_ids, unmatched_track_ids, matrix_3d_sim, params["iou_3d_threshold"],
                          track_classes, params["thresholds_per_class"])


def associate_instances_to_tracks_2d_iou(instances_leftover: Iterable[FusedInstance],
                                         tracks_leftover: Iterable[Track],
                                         iou_threshold: float,
                                         ego_transform, angle_around_y,
                                         transformation, img_shape_per_cam: Mapping[str, Any],
                                         cam: str, frame_data: Mapping[str, Any]):
    detected_bboxes_2d = [instance.bbox_2d_best(cam) for instance in instances_leftover]
    tracked_bboxes_2d = [track.predicted_bbox_2d_in_cam(ego_transform, angle_around_y,
                                                        transformation, img_shape_per_cam,
                                                        cam, frame_data) for track in tracks_leftover]
    return associate_boxes_2d(detected_bboxes_2d, tracked_bboxes_2d, iou_threshold)


def associate_boxes_2d(detected_bboxes_2d, tracked_bboxes_2d, iou_threshold):
    """
    Links two sets of 2D bounding boxes based on their IoU.
    Returns 3 lists of matches, unmatched_detection_indices and unmatched_track_indices
    """
    iou_matrix = iou_bbox_2d_matrix(detected_bboxes_2d, tracked_bboxes_2d)
    matched_indices, unmatched_detection_indices, unmatched_track_indices = \
        perform_association_from_similarity(len(detected_bboxes_2d), len(tracked_bboxes_2d), iou_matrix)
    return filter_matches(matched_indices, unmatched_detection_indices, unmatched_track_indices, iou_matrix, iou_threshold)


def filter_matches(matched_indices, unmatched_first_indices, unmatched_second_indices, matrix,
                   threshold=None, classes_second=None, thresholds_per_class=None):
    assert threshold is None or not thresholds_per_class
    matches = []

    if threshold is not None:
        for m in matched_indices:
            if matrix[m[0], m[1]] < threshold:
                unmatched_first_indices.append(m[0])
                unmatched_second_indices.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
    else:
        for m in matched_indices:
            second_i = m[1]
            if matrix[m[0], second_i] < thresholds_per_class[classes_second[second_i]]:
                unmatched_first_indices.append(m[0])
                unmatched_second_indices.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

    matches = np.vstack(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)

    return matches, unmatched_first_indices, unmatched_second_indices


def perform_association_from_similarity(first_items_len, second_items_len, similarity_matrix):
    # return _perform_association_from_cost_hu(first_items_len, second_items_len, -similarity_matrix)
    return _perform_association_from_cost_greedy(first_items_len, second_items_len, -similarity_matrix)


def _perform_association_from_cost_hu(first_items_len, second_items_len, cost_matrix
                                      ) -> Tuple[np.ndarray, List[int], List[int]]:
    # Run the Hungarian algorithm for assignment from the cost matrix
    matched_indices = linear_sum_assignment(cost_matrix)
    matched_indices = np.asarray(matched_indices).T

    unmatched_first_items = [i for i in range(first_items_len) if i not in matched_indices[:, 0]]
    unmatched_second_items = [i for i in range(second_items_len) if i not in matched_indices[:, 1]]
    return matched_indices, unmatched_first_items, unmatched_second_items


def _perform_association_from_cost_greedy(first_items_len, second_items_len, cost_matrix):
    # Match greedily based on the cost matrix
    matched_indices = greedy_match(cost_matrix)
    matched_indices = matched_indices.reshape((-1, 2))

    unmatched_first_items = [i for i in range(first_items_len) if i not in matched_indices[:, 0]]
    unmatched_second_items = [i for i in range(second_items_len) if i not in matched_indices[:, 1]]
    return matched_indices, unmatched_first_items, unmatched_second_items


# Adapted from https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
def greedy_match(cost_matrix):
    """ Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    """
    num_detections, num_tracks = cost_matrix.shape
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1).astype(int, copy=False)

    matched_indices = []
    matched_firsts = set()
    matched_seconds = set()

    for (first_id, second_id) in index_2d:
        if first_id not in matched_firsts and second_id not in matched_seconds:
            matched_seconds.add(second_id)
            matched_firsts.add(first_id)
            matched_indices.append([first_id, second_id])
    return np.array(matched_indices)


def match_3d_2d_detections(dets_3d: Sequence[Bbox3d], cam: str, dets_2d: Sequence[Detection2D],
                           fusion_iou_threshold: Tuple[float, ...], classes_to_match: Iterable[int]
                           ) -> Tuple[Dict[int, int], Set[int], List[int]]:
    matched_indices: Dict[int, int] = {}
    unmatched_dets_3d_ids: Set[int] = set()
    unmatched_dets_2d_ids: List[int] = []
    for class_id in classes_to_match:
        indices_dets_3d_current_class = [i for (i, det_3d) in enumerate(dets_3d)
                                         if det_3d.seg_class_id == class_id]
        indices_dets_2d_current_class = [i for (i, det_2d) in enumerate(dets_2d)
                                         if det_2d.seg_class_id == class_id]

        bboxes_detections_3d_current_class = [
            dets_3d[i].bbox_2d_in_cam(cam) for i in indices_dets_3d_current_class]
        bboxes_detections_2d_current_class = [
            (dets_2d[i].bbox) for i in indices_dets_2d_current_class]

        matched_indices_class, unmatched_dets_3d_ids_class, unmatched_dets_2d_ids_class = \
            associate_boxes_2d(bboxes_detections_3d_current_class,
                               bboxes_detections_2d_current_class, fusion_iou_threshold[class_id - 1])

        for det_3d_i, det_2d_i in matched_indices_class:
            matched_indices[indices_dets_3d_current_class[det_3d_i]] = indices_dets_2d_current_class[det_2d_i]
        unmatched_dets_3d_ids.update([indices_dets_3d_current_class[det_3d_i]
                                      for det_3d_i in unmatched_dets_3d_ids_class])
        unmatched_dets_2d_ids.extend([indices_dets_2d_current_class[det_2d_i]
                                      for det_2d_i in unmatched_dets_2d_ids_class])
    return matched_indices, unmatched_dets_3d_ids, unmatched_dets_2d_ids


def match_multicam(candidate_matches: Mapping[int, Sequence[CamDetectionIndices]],
                   instances_3d: Sequence[ProjectsToCam]) -> Dict[int, CamDetectionIndices]:
    """ 
    Matches each 3D instance (3D detection / track) with a single 2D detection given 
    a list of possible detections to match to. Decides which candidate detection to assign
    based on the area of the 2D projection of the 3D instance in that camera.
    The intuition is that the cam where the 3D object is most prevalent 
    will most likely have its 2D projection recognized correctly

    :param candidate_matches: maps entities to a *sequence* of possible 2D detections
    :param instances_3d: entities that need unique matches
    :return: dict mapping original entities to a *single* 2D detection
    """
    matched_indices: Dict[int, CamDetectionIndices] = {}
    for instance_i, cam_det_2d_indices in candidate_matches.items():
        assert cam_det_2d_indices  # has to be at least 1 candidate match
        if len(cam_det_2d_indices) == 1:
            matched_indices[instance_i] = cam_det_2d_indices[0]
        else:
            # if matches were made in multiple cameras,
            # select the camera with the largest projection of the 3D detection
            instance_3d = instances_3d[instance_i]
            largest_area = 0.0

            for cam, det_2d_i in cam_det_2d_indices:
                area = box_2d_area(instance_3d.bbox_2d_in_cam(cam))
                assert area > 0, f"All of the candidate 2D projections have to be valid {instance_3d.bbox_2d_in_cam(cam)}"
                if area > largest_area:
                    largest_area = area
                    chosen_cam_det = (cam, det_2d_i)
            assert largest_area > 0, "3D instance has to have at least one valid 2D projection"
            matched_indices[instance_i] = chosen_cam_det
            # assuming that matches were correct in multiple cameras
            # simply discard duplicate 2D detections and don't treat them as unmatched
            # another option is to allow multiple 2D detections for each 3D instance and ignore this function
            #
            # if matches were incorrect, then these "duplicates" should be added
            # to the rest of unmatched detections, but we will assume matches are correct
    return matched_indices
