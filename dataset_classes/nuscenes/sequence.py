from __future__ import annotations
import os
from typing import Optional, List, Dict, Set, Any, Iterable, IO
from collections import defaultdict

import numpy as np
from nuscenes.nuscenes import NuScenes

from inputs.bbox import Bbox3d
from inputs.detection_2d import Detection2D
from objects.fused_instance import FusedInstance
from dataset_classes.mot_sequence import MOTSequence
from dataset_classes.nuscenes.frame import MOTFrameNuScenes
import dataset_classes.nuscenes.classes as nu_classes
import dataset_classes.nuscenes.reporting as reporting
from transform.nuscenes import TransformationNuScenes
from configs.local_variables import NUSCENES_DATA_DIR
import inputs.loading as loading
import inputs.utils as input_utils


class MOTSequenceNuScenes(MOTSequence):
    def __init__(self, det_source: str, seg_source: str, split_dir: str, split: str,
                 nusc: NuScenes, scene, dataset_submission: Dict[str, Dict[str, Any]],
                 dataset_detections_3d: Dict[str, List[Bbox3d]]):
        self.nusc = nusc
        self.scene = scene
        self.frame_tokens = self._parse_frame_tokens()
        self.dataset_submission = dataset_submission
        self.dataset_detections_3d = dataset_detections_3d

        super().__init__(det_source, seg_source, split_dir, self.scene["name"], self.frame_tokens)
        self.data_dir = os.path.join(NUSCENES_DATA_DIR, split)

        self._transformation: Optional[TransformationNuScenes] = None
        self.mot.transformation = self.transformation

        fusion_name = 'det_%s_%s_%s_%s_%s_%s_%s_seg_%s_%s_%s_%s_%s_%s_%s_iou_%s_%s_%s_%s_%s_%s_%s'
        self.instance_fusion_bbox_dir = os.path.join(
            self.work_split_input_dir, 'instance_fusion_bbox', fusion_name, self.name)

        self.first_frame = self.nusc.get("sample", self.frame_tokens[0])
        self.token = self.scene["token"]

        self.center_world_point: Optional[np.ndarray] = None

    @property
    def transformation(self) -> TransformationNuScenes:
        if self._transformation is None:
            self._transformation = TransformationNuScenes(self.nusc, self.scene)
        return self._transformation

    def _parse_frame_tokens(self) -> List[str]:
        frame_tokens: List[str] = []
        frame_token = self.scene['first_sample_token']  # first frame token
        while frame_token:  # should break when loading the last frame, which has None for "next"
            frame_nu = self.nusc.get("sample", frame_token)
            frame_tokens.append(frame_token)
            assert frame_nu["scene_token"] == self.scene["token"]
            # update token to the next frame
            frame_token = frame_nu["next"]

        expected_num_frames = self.scene["nbr_samples"]
        assert (len(frame_tokens) ==
                expected_num_frames), f"Expected {expected_num_frames} frames but parsed {len(frame_tokens)}"
        return frame_tokens

    def get_frame(self, frame_token: str) -> MOTFrameNuScenes:
        frame = MOTFrameNuScenes(self, frame_token, self.nusc)
        if not self.img_shape_per_cam:
            for cam in self.cameras:
                frame.get_image_original(cam)
            self.mot.img_shape_per_cam = self.img_shape_per_cam
        return frame

    def load_detections_3d(self) -> Dict[str, List[Bbox3d]]:
        if not self.dataset_detections_3d:
            self.dataset_detections_3d.update(loading.load_detections_3d(self.det_source, self.name))
        return self.dataset_detections_3d

    def load_detections_2d(self) -> Dict[str, Dict[str, List[Detection2D]]]:
        frames_cam_tokens_detections = loading.load_detections_2d_nuscenes(self.seg_source, self.token)
        frames_cams_detections: Dict[str, Dict[str, List[Detection2D]]
                                     ] = defaultdict(lambda: defaultdict(list))

        for frame_token, cam_detections in frames_cam_tokens_detections.items():
            for cam_data_token, detections in cam_detections.items():
                cam = self.nusc.get('sample_data', cam_data_token)["channel"]
                for detection in detections:
                    detection.cam = cam
                frames_cams_detections[frame_token][cam] = detections
        return frames_cams_detections

    @property
    def cameras(self) -> List[str]:
        return ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

    @property
    def camera_default(self) -> str:
        return "CAM_FRONT"

    @property
    def classes_to_track(self) -> List[int]:
        return nu_classes.ALL_NUSCENES_CLASS_IDS

    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[FusedInstance],
                   mot_3d_file: IO,
                   mot_2d_from_3d_only_file: Optional[IO]) -> None:
        reporting.add_results_to_submit(self.dataset_submission, frame_name, predicted_instances)

    def save_mot_results(self, mot_3d_file: IO, mot_2d_from_3d_file: Optional[IO]) -> None: pass

    def load_ego_motion_transforms(self) -> None:
        """ Not needed for NuScenes """

    def save_ego_motion_transforms_if_new(self) -> None:
        """ Not needed for NuScenes """
