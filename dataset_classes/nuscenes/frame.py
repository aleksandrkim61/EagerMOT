from __future__ import annotations
import os
from typing import Optional, List, Callable, Iterable, Dict
from collections import defaultdict

import matplotlib.image as mpimg
import imageio
import numpy as np
from pyquaternion import Quaternion
from inputs.detection_2d import Detection2D
from inputs.bbox import Bbox2d, Bbox3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.common.utils import quaternion_yaw
from utils.utils_geometry import clip_bbox_to_four_corners

from dataset_classes.mot_frame import MOTFrame
import dataset_classes.nuscenes.classes as nu_classes
from transform.nuscenes import TransformationNuScenes, ROTATION_NEGATIVE_X_FULL

AB3DMOT = 'ab3dmot'
CENTER_POINT = 'center_point'

EFFICIENT_DET = 'efficient_det'


class MOTFrameNuScenes(MOTFrame):
    def __init__(self, sequence, name: str, nusc: NuScenes):
        super().__init__(sequence, name)
        self.nusc = nusc
        self.frame = self.nusc.get("sample", name)
        assert self.frame["scene_token"] == self.sequence.scene["token"]
        self.data = self.frame["data"]

        self._points_world: Optional[np.ndarray] = None

    @property
    def transformation(self) -> TransformationNuScenes:
        return self.sequence.transformation

    def get_image_original(self, cam: str):
        return self._read_image(cam, mpimg.imread)

    def get_image_original_uint8(self, cam: str):
        return self._read_image(cam, imageio.imread)

    def _read_image(self, cam: str, read_function: Callable):
        image_path = self.nusc.get_sample_data_path(self.data[cam])
        image = read_function(image_path)
        # need to remember actual image size
        self.sequence.img_shape_per_cam[cam] = image.shape[:2]
        return image

    def load_raw_pcd(self):
        lidar_data = self.nusc.get('sample_data', self.data["LIDAR_TOP"])
        assert lidar_data["is_key_frame"]
        lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data["filename"])
        nu_pcd: LidarPointCloud = LidarPointCloud.from_file(lidar_filepath)
        return nu_pcd.points[:3].T

    @property
    def points_world(self) -> np.ndarray:
        if self._points_world is None:
            self._points_world = self.transformation.world_from_lidar(
                self.raw_pcd, self.data)
        if self.sequence.center_world_point is None:
            self.sequence.center_world_point = self._points_world.mean(axis=0)
            self.sequence.center_world_point[2] = 0.0
        return self._points_world

    @property
    def center_world_point(self) -> np.ndarray:
        if self.sequence.center_world_point is None:
            self.points_world
        return self.sequence.center_world_point

    def bbox_3d_annotations(self, world: bool = False) -> List[Bbox3d]:  # List[Box]
        bboxes = (self.bbox_3d_annotation(token, world) for token in self.frame["anns"])
        return [bbox for bbox in bboxes if bbox is not None]

    def bbox_3d_annotation(self, annotation_token: str, world: bool = False) -> Optional[Bbox3d]:  # Box
        bbox_nu = self.nusc.get_box(annotation_token)  # annotations are in world coordinates
        if not world:
            bbox_nu = self.transformation.ego_box_from_world(bbox_nu, self.data)
        bbox_nu.score = 1.0
        bbox_nu.velocity = [1.0, 1.0]

        instance_id = hash(annotation_token)
        name_parts = bbox_nu.name.split(".")
        bbox_class = name_parts[1] if len(name_parts) > 1 else name_parts[0]
        if bbox_class in nu_classes.ALL_NUSCENES_CLASS_NAMES:
            return Bbox3d.from_nu_box_convert(bbox_nu, instance_id)
        else:
            return None

    def bbox_2d_annotation_projections(self) -> Dict[str, List[Detection2D]]:
        # use annotation projections
        dets_2d_multicam: Dict[str, List[Detection2D]] = {cam: [] for cam in self.sequence.cameras}
        bboxes_3d = self.bbox_3d_annotations(world=True)
        for bbox_3d in bboxes_3d:
            for cam in self.sequence.cameras:
                bbox_projected = self.transformation.img_from_tracking(bbox_3d.corners_3d, cam, self.data)
                box_coords = clip_bbox_to_four_corners(
                    bbox_projected, self.sequence.img_shape_per_cam[cam])
                if box_coords is not None:
                    dets_2d_multicam[cam].append(Detection2D(
                        Bbox2d(*box_coords), cam, bbox_3d.confidence, bbox_3d.seg_class_id))
        return dets_2d_multicam

    @property
    def bboxes_3d_world(self) -> List[Bbox3d]:
        return self.bboxes_3d

    @property
    def bboxes_3d_ego(self) -> List[Bbox3d]:
        lidar_data = self.nusc.get('sample_data', self.data["LIDAR_TOP"])
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        transform_matrix = np.ones((4, 4), float)
        rotation_quaternion = Quaternion(ego_pose_data["rotation"])
        transform_matrix[:3, :3] = rotation_quaternion.rotation_matrix
        transform_matrix[:3, 3] = ego_pose_data["translation"]
        angle_around_vertical = -1 * quaternion_yaw(rotation_quaternion)

        bboxes = self.bboxes_3d.copy()
        for bbox in bboxes:
            # need to go back to Nu coordinates
            bbox.inverse_transform(ROTATION_NEGATIVE_X_FULL, 0)
            # transform to ego
            bbox.inverse_transform(transform_matrix, angle_around_vertical)
            # back to internal tracking frame, i.e. KITTI's original/ego frame
            bbox.transform(ROTATION_NEGATIVE_X_FULL, 0)
        return bboxes

    def transform_instances_to_world_frame(self):
        return None, None
