from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Iterable, Mapping, Dict, Any, Optional, IO

import numpy as np

# from inputs.bbox import bbox.Bbox3d
import inputs.bbox as bbox
from tracking.tracking_manager import TrackManager
import utils.io as io
import dataset_classes.mot_frame as mot_frame
from inputs.detection_2d import Detection2D
from objects.fused_instance import FusedInstance
from transform.transformation import Transformation
from configs.params import variant_name_from_params


class MOTSequence(ABC):
    def __init__(self, det_source: str, seg_source: str, split_dir: str, name: str, frame_names: Iterable[str]):
        self.det_source = det_source
        self.seg_source = seg_source
        self.split_dir = split_dir
        self.name = name
        self.frame_names = frame_names

        # Image size for each camera - needed for 3D->2D projections. The dict is set in dataset-specific classes
        self.img_shape_per_cam: Dict[str, Any] = {}

        # Detections 3D {frame_name: [bboxes_3d]}
        self.dets_3d_per_frame: Dict[str, List[bbox.Bbox3d]] = {}

        # Detections 2D {frame_name: {cam_name: [bboxes_3d]}}
        self.dets_2d_multicam_per_frame: Dict[str, Dict[str, List[Detection2D]]] = {}

        # need to set its Transformation object and img_shape_per_cam in subclasses
        self.mot = TrackManager(self.cameras, self.classes_to_track)

        det_seg_source_folder_name = f'{self.det_source}_{self.seg_source}'
        self.work_split_input_dir = os.path.join(self.split_dir, det_seg_source_folder_name)
        self.tracking_res_dir = os.path.join(self.work_split_input_dir, 'tracking')

    ##########################################################
    # Evaluation

    def perform_tracking_for_eval(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        folder_identifier = 'cleaning_0'  # add an easy identifier for code-related changes

        mot_3d_file = io.create_writable_file_if_new(
            self.get_results_folder_name(params, folder_identifier, "3d"), self.name)
        mot_2d_from_3d_file = io.create_writable_file_if_new(
            self.get_results_folder_name(params, folder_identifier, "2d_projected_3d"), self.name)

        run_info: Dict[str, Any] = defaultdict(int)

        if mot_3d_file is None:
            print(f'Sequence {self.name} already has results. Skipped')
            print('=====================================================================================')
            return run_info

        run_info["mot_3d_file"] = mot_3d_file.name.split(self.name)[0]
        run_info["mot_2d_from_3d_file"] = mot_2d_from_3d_file.name.split(self.name)[0]

        self.load_ego_motion_transforms()
        for frame_i, frame_name in enumerate(self.frame_names):
            if frame_i % 100 == 0:
                print(f'Processing frame {frame_name}')

            frame = self.get_frame(frame_name)

            predicted_instances = frame.perform_tracking(params, run_info)

            start_reporting = time.time()
            self.report_mot_results(frame.name, predicted_instances, mot_3d_file, mot_2d_from_3d_file)
            run_info["total_time_reporting"] += time.time() - start_reporting

        self.save_mot_results(mot_3d_file, mot_2d_from_3d_file)
        self.save_ego_motion_transforms_if_new()
        return run_info

    def get_results_folder_name(self, params: Mapping[str, Any], folder_identifier: str, suffix: str):
        folder_suffix_full = f"{variant_name_from_params(params)}_{folder_identifier}_{suffix}"
        return f"{self.tracking_res_dir}_{folder_suffix_full}"

    ##########################################################
    # Lazy getters for frame-specific data

    def get_segmentations_for_frame(self, frame_name: str) -> Dict[str, List[Detection2D]]:
        """ Return a dict of Detection2D for each camera for the requested frame"""
        if not self.dets_2d_multicam_per_frame:
            self.dets_2d_multicam_per_frame = self.load_detections_2d()
        return self.dets_2d_multicam_per_frame.get(frame_name, defaultdict(list))

    def get_bboxes_for_frame(self, frame_name: str) -> List[bbox.Bbox3d]:
        """ Return a list of bbox.Bbox3d for the requested frame"""
        if not self.dets_3d_per_frame:
            self.dets_3d_per_frame = self.load_detections_3d()
        return self.dets_3d_per_frame.get(frame_name, [])

    ##########################################################
    # Required methods and fields that need to be overridden by subclasses
    # This sadly results in some extra code, but is the best way to ensure compile-time errors

    @abstractmethod
    def load_ego_motion_transforms(self) -> None: pass

    @abstractmethod
    def save_ego_motion_transforms_if_new(self) -> None: pass

    @abstractmethod
    def load_detections_3d(self) -> Dict[str, List[bbox.Bbox3d]]: pass

    @abstractmethod
    def load_detections_2d(self) -> Dict[str, Dict[str, List[Detection2D]]]: pass

    @abstractmethod
    def get_frame(self, frame_name: str) -> mot_frame.MOTFrame: pass

    @property
    @abstractmethod
    def transformation(self) -> Transformation: pass

    @property
    @abstractmethod
    def cameras(self) -> List[str]: pass

    @property
    @abstractmethod
    def camera_default(self) -> str: pass

    @property
    @abstractmethod
    def classes_to_track(self) -> List[int]: pass

    @abstractmethod
    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[FusedInstance],
                           mot_3d_file: IO,
                           mot_2d_from_3d_only_file: Optional[IO]) -> None:
        pass

    @abstractmethod
    def save_mot_results(self, mot_3d_file: IO,
                         mot_2d_from_3d_file: Optional[IO]) -> None:
        pass
