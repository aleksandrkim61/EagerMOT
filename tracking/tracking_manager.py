import time
from typing import Iterable, List, Dict, Set, Optional, Sequence, Any, Sequence

import numpy as np

from tracking.data_association import (
    associate_instances_to_tracks_3d_iou,
    associate_instances_to_tracks_2d_iou,
    match_multicam, CamDetectionIndices
)
from tracking.tracks import Track
from collections import defaultdict
from objects.fused_instance import FusedInstance
from transform.transformation import Transformation
import dataset_classes.nuscenes.classes as nu_classes
from utils.utils_geometry import project_bbox_3d_to_2d
from inputs.bbox import Bbox2d


class TrackManager(object):
    def __init__(self, cameras: Sequence[str], classes_to_track: Iterable[int]):
        self.trackers: List[Track] = []
        self.frame_count = 0
        self.cameras = cameras
        self.classes_to_track = classes_to_track

        Track.count = 0
        self.track_ids_map: Dict[int, int] = {}
        self.track_id_latest = 1  # evaluations expect positive track ids

        # will be set by the calling MOTSequence
        self.transformation: Optional[Transformation] = None
        # maps cameras to their image plane shapes whose len need to be 2: tuple, array, etc.
        self.img_shape_per_cam: Optional[Dict[str, Any]] = None

    def set_track_manager_params(self, params):
        self.is_angular = params.get('is_angular', False)
        # How many frames a track gets to match with something before it is killed (2?)
        self.max_ages = params.get('max_ages')
        self.max_age_2d = params.get("max_age_2d")
        # How many matches a track needs to get in its lifetime to be considered confirmed
        self.min_hits = params.get('min_hits')

        self.second_matching_method = params.get('second_matching_method', 'iou')
        self.leftover_thres = params.get('leftover_matching_thres')

    def update(self, fused_instances: Iterable[FusedInstance], params: Dict,
               frame_data: Dict, run_info: Dict, ego_transform=None, angle_around_y=None):
        """ Matches current frame's detections with existing tracks and manages their lifecycle. Should be called for each frame even with empty detections

        :param fused_instances: list of FusedInstance objects from the current frame
        :param params: dictionary of tracking parameters, see configs.params
        :param ego_transform: 4x4 transformation matrix to convert from current to world coordinates
        :param angle_around_y: [description], defaults to None
        :return: [description]
        :rtype: [type]
        """
        self.frame_count += 1
        for track in self.trackers:
            track.reset_for_new_frame()

        for class_target in self.classes_to_track:
            # select which objects to consider for tracking
            # filter instances suitable/not suitable for the 1st matching stage
            det_instances_3d = [instance for instance in fused_instances
                                if instance.bbox3d is not None and instance.class_id == class_target]
            det_instances_from_mask = [instance for instance in fused_instances
                                       if not instance.bbox3d is not None and instance.class_id == class_target]

            tracks_with_3d_models = []
            track_indices_with_motion_models = []
            track_indices_without_motion_models = []
            for track_i, track in enumerate(self.trackers):
                if track.class_id != class_target:
                    continue

                if track.has_motion_model:
                    tracks_with_3d_models.append(track)
                    track_indices_with_motion_models.append(track_i)
                else:
                    track_indices_without_motion_models.append(track_i)

            # print(f"Frame: {self.frame_count - 1}")
            # The 1st matching stage via 3D IoU
            matched_instances_to_tracks_first, unmatched_det_indices_first, unmatched_motion_track_indices_first = \
                associate_instances_to_tracks_3d_iou(det_instances_3d, tracks_with_3d_models, params)

            # print(f'det_instances_3d: {len(det_instances_3d)}, predicted_3d_states: {len(predicted_3d_states)}, total tracks: {len(self.trackers)}')
            # print(f'matched_instances_to_tracks_first {len(matched_instances_to_tracks_first)}')
            # print(f'unmatched_det_indices_first {len(unmatched_det_indices_first)}')
            # print(f'unmatched_motion_track_indices_first {len(unmatched_motion_track_indices_first)}')
            # print(f'matched_instances_to_tracks_first {matched_instances_to_tracks_first}')
            # print(f'unmatched_det_indices_first {unmatched_det_indices_first}')
            # print(f'unmatched_motion_track_indices_first {unmatched_motion_track_indices_first}')
            # print()

            # map indices of motion tracks back to original indices among all tracks
            for i in range(len(matched_instances_to_tracks_first)):
                matched_instances_to_tracks_first[i, 1] = \
                    track_indices_with_motion_models[matched_instances_to_tracks_first[i, 1]]

            # Gather tracks that have no motion model
            leftover_track_indices = [track_indices_with_motion_models[i]
                                      for i in unmatched_motion_track_indices_first]
            leftover_track_indices.extend(track_indices_without_motion_models)
            leftover_tracks = [self.trackers[track_i] for track_i in leftover_track_indices]

            run_info["matched_tracks_first_total"] += len(matched_instances_to_tracks_first)
            run_info["unmatched_tracks_first_total"] += len(leftover_tracks)

            assert len(unmatched_det_indices_first) == len(set(unmatched_det_indices_first))
            assert len(unmatched_motion_track_indices_first) == len(set(unmatched_motion_track_indices_first))
            assert len(leftover_track_indices) == len(set(leftover_track_indices))

            # Gather all unmatched detected instances (no 3D box + failed 1st stage)
            leftover_det_instance_multicam: Dict[str, List[FusedInstance]] = defaultdict(list)
            leftover_det_instances_no_2d: List[FusedInstance] = []
            total_leftover_det_instances = len(det_instances_from_mask)
            for instance in det_instances_from_mask:
                assert instance.detection_2d
                leftover_det_instance_multicam[instance.detection_2d.cam].append(instance)
            for det_i in unmatched_det_indices_first:
                instance = det_instances_3d[det_i]
                # #431 do not use instances with 3D in the 2nd stage
                if instance.detection_2d and instance.bbox3d is None:
                    leftover_det_instance_multicam[instance.detection_2d.cam].append(instance)
                    total_leftover_det_instances += 1
                else:
                    leftover_det_instances_no_2d.append(instance)

            # print(f"leftover_det_instance_multicam:\n{leftover_det_instance_multicam}")
            # print(f"leftover_det_instances_no_2d:\n{leftover_det_instances_no_2d}")

            # The 2nd matching stage via 2D bbox IoU (multicam)
            matched_indices: Dict[int, List[CamDetectionIndices]] = defaultdict(list)
            unmatched_track_indices_final: Set[int] = set(range(len(leftover_tracks)))
            unmatched_det_indices_final: Dict[str, Set[int]] = {
                cam: set(range(len(det_instances))) for cam, det_instances in leftover_det_instance_multicam.items()
            }

            if self.leftover_thres is not None and self.leftover_thres < 1.0:
                # second_start_time = time.time()
                for cam, instances_list in leftover_det_instance_multicam.items():
                    assert self.second_matching_method == "iou"
                    assert all(instance.bbox3d is None for instance in instances_list)  # 431
                    (matched_instances_to_tracks_second_cam, unmatched_det_indices_cam,
                        unmatched_track_indices_cam) = \
                        associate_instances_to_tracks_2d_iou(
                            instances_list, leftover_tracks, self.leftover_thres,
                            ego_transform, angle_around_y, self.transformation, self.img_shape_per_cam, cam, frame_data)

                    for instance_i, track_i in matched_instances_to_tracks_second_cam:
                        matched_indices[track_i].append((cam, instance_i))
                        unmatched_det_indices_final[cam].discard(instance_i)
                # print(f"2nd stage: {time.time() - second_start_time:.2f}")
                # print()

            # remove matched track indices from the unmatched indices set
            unmatched_track_indices_final -= matched_indices.keys()

            # print(f"matched_indices:\n{matched_indices}")
            # print(f"matched_indices.keys():\n{matched_indices.keys()}")
            # print(f"unmatched_track_indices_final:\n{unmatched_track_indices_final}")
            # print(f"unmatched_det_indices_final:\n{unmatched_det_indices_final}")
            # print()

            assert unmatched_track_indices_final.union(
                matched_indices.keys()) == set(range(len(leftover_tracks)))

            total_matched_leftover_instances = 0
            for list_matches in matched_indices.values():
                assert not len(list_matches) > len(self.cameras)
                total_matched_leftover_instances += len(list_matches)
                for (cam, det_i) in list_matches:
                    assert det_i not in unmatched_det_indices_final[cam]

            total_unmatched_leftover_instances = 0
            for unmatched_det_indices in unmatched_det_indices_final.values():
                total_unmatched_leftover_instances += len(unmatched_det_indices)
            assert total_matched_leftover_instances + total_unmatched_leftover_instances == total_leftover_det_instances

            matched_tracks_to_cam_instances_second = match_multicam(matched_indices, leftover_tracks)
            run_info["matched_tracks_second_total"] += len(matched_tracks_to_cam_instances_second)
            run_info["unmatched_tracks_second_total"] += len(unmatched_track_indices_final)
            run_info["unmatched_dets2d_second_total"] += total_unmatched_leftover_instances

            # update tracks that were matched with fully fused instances
            for track_i in matched_instances_to_tracks_first[:, 1]:
                assert track_i not in leftover_track_indices
                track = self.trackers[track_i]
                assert track.class_id == class_target
                matched_det_id = matched_instances_to_tracks_first[np.where(
                    matched_instances_to_tracks_first[:, 1] == track_i)[0], 0]
                matched_instance = det_instances_3d[matched_det_id[0]]
                track.update_with_match(matched_instance)

            # update tracks that were watched with instances based on 2D IoU (may or may not have 3D boxes)
            for track_i_secondary, (cam, instance_i) in matched_tracks_to_cam_instances_second.items():
                track = leftover_tracks[track_i_secondary]
                assert track.class_id == class_target
                track.update_with_match(leftover_det_instance_multicam[cam][instance_i])

            # create and initialise new tracks for all unmatched detections
            for cam, indices in unmatched_det_indices_final.items():
                for instance_i in indices:
                    instance = leftover_det_instance_multicam[cam][instance_i]
                    if instance.bbox3d is not None:
                        self.trackers.append(Track(instance, self.is_angular))
            self.trackers.extend([Track(instance, self.is_angular)
                                  for instance in leftover_det_instances_no_2d if instance.bbox3d is not None])

        # Report, remove obsolete tracks for all classes
        instances_tracked = self.report_tracks(ego_transform, angle_around_y)
        self.remove_obsolete_tracks()
        return instances_tracked

    def report_tracks(self, ego_transform, angle_around_y):
        # from [x,y,z,theta,l,w,h] to [h, w, l, x, y, z, theta]
        reorder_back = [6, 5, 4, 0, 1, 2, 3]
        instances_tracked = []
        for track in reversed(self.trackers):
            if self.is_recent(track.class_id, track.time_since_update):
                track_id = self.unique_track_id(track.id)
                instance = track.current_instance(
                    ego_transform, angle_around_y, self.min_hits[track.class_id - 1])
                instance.track_id = track_id

                instance.report_mot = (self.is_confirmed_track(track.class_id, track.hits, track.age_total)
                                       and track.time_since_update == 0)

                if track.has_motion_model:  # report MOT
                    bbox_3d = track.current_bbox_3d(ego_transform, angle_around_y)  # current 3D bbox
                    instance.coordinates_3d = bbox_3d.kf_coordinates[reorder_back]
                    instance.bbox3d.obs_angle = track.obs_angle

                    if len(self.cameras) < 2:  # KITTI
                        bbox_2d = project_bbox_3d_to_2d(
                            bbox_3d, self.transformation, self.img_shape_per_cam, self.cameras[0], None)
                        instance.projected_bbox_3d = Bbox2d(*bbox_2d) if bbox_2d is not None else None

                    max_age_2d_for_class = self.max_age_2d[track.class_id - 1]
                    if track.time_since_2d_update < max_age_2d_for_class:
                        instance.bbox3d.confidence = track.confidence
                    else:
                        frames_since_allowed_no_2d_update = track.time_since_2d_update + 1 - max_age_2d_for_class
                        instance.bbox3d.confidence = track.confidence / \
                            (2.0 * frames_since_allowed_no_2d_update)
                else:
                    instance.report_mot = False

                instances_tracked.append(instance)
        return instances_tracked

    def remove_obsolete_tracks(self):
        track_i = len(self.trackers) - 1
        for track in reversed(self.trackers):
            if track.time_since_update >= self.max_ages[track.class_id - 1]:
                del self.trackers[track_i]
            track_i -= 1

    def is_confirmed_track(self, class_id, hits, age_total):
        required_hits = self.min_hits[class_id - 1]
        if self.frame_count < required_hits:
            return hits >= age_total
        else:
            return hits >= required_hits

    def is_recent(self, class_id, time_since_update):
        return time_since_update < self.max_ages[class_id - 1]

    def unique_track_id(self, original_track_id):  
        # this was a necessary workaround for MOTS, which has a limit on submitted track IDs
        # If the submitted track.id was larger than some threshold, evaluation did not work correctly
        # This might have been patched since then, we alerted the MOTS team, but this might be safer still
        if original_track_id not in self.track_ids_map:
            self.track_ids_map[original_track_id] = self.track_id_latest
            self.track_id_latest += 1
        return self.track_ids_map[original_track_id]
