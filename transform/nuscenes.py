from typing import Tuple, Optional, Mapping
import copy

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

from transform.transformation import Transformation, to_homogeneous, cam_points_to_image_coordinates


class TransformationNuScenes(Transformation):
    """
    Coordinate frames schema: https://www.nuscenes.org/public/images/data.png
    World: z - up
    Lidar: x, y, z - right, forward face, up
    Cams: x, y, z - right, down, forward face
    Radars: x, y, z - forward face, left, up

    Rect points are points in the ego frame for NuScenes"

    Lidar sensor reports points in sensor coordinates
    1) Using calibrated_sensor lidar data, points can be transformed to ego vehicle frame
            lidar_sensor_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            transform_pcd_with_pose(nuscenes_pcd, lidar_sensor_data)

    2a) Using ego_pose data, ego points can be transformed to world frame
            ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            transform_pcd_with_pose(nuscenes_pcd, ego_pose_data)
    2b) Using calibrated_sensor cam data, ego points can be transformed to camera frame
            sd_record = self.get('sample_data', sample_data_token)
            cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            #  Move a pcd to sensor coord system.
            inverse_transform_pcd_with_pose(nuscenes_pcd, cs_record)
            #  Move a box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
    """

    def __init__(self, nusc: NuScenes, scene):
        super().__init__()
        self.nusc = nusc
        self.scene = scene

    def img_from_tracking(self, track_points: np.ndarray, cam: str, frame_data: Optional[Mapping]) -> np.ndarray:
        """
        :param track_points: nx3 3D points in the tracking frame i.e. world coordinates in KITTI rect frame 
        :param camera: to which camera plane perform the projection
        :return: nx2 2D coordinates of points in the specified camera's image coordinates
        """
        assert frame_data is not None

        # Rotate points from KITTI frame to NuScenes frame, coordinates are relative to world origin
        nuscenes_world_points = kitti_to_nuscenes(track_points)

        # Move the pcd from world to ego frame
        cam_data = self.nusc.get('sample_data', frame_data[cam])
        ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_points = inverse_transform_points_with_pose(nuscenes_world_points, ego_pose_data)

        # Move the pcd from ego to cam sensor frame
        # cam_data = self.nusc.get('sample_data', frame_data[cam])
        cam_sensor_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_points = inverse_transform_points_with_pose(ego_points, cam_sensor_data)

        # Only keep points in front of the frame
        cam_front_points = cam_points[cam_points[:, 2] > 0]

        # Project to image plane
        intrinsic = np.array(cam_sensor_data['camera_intrinsic'])
        assert intrinsic.shape == (3, 3)
        img_points = cam_front_points @ intrinsic.T

        return cam_points_to_image_coordinates(img_points).astype(int, copy=False)

    def rect_from_lidar(self, lidar_points: np.ndarray, frame_data: Optional[Mapping],
                        only_forward=False) -> np.ndarray:
        """
        :param lidar_points: Nx3 points in LiDAR coordinates as np.ndarray
        :return: Nx3 3D points in ego vehicle coordinates
        """
        assert frame_data is not None
        lidar_data = self.nusc.get('sample_data', frame_data["LIDAR_TOP"])
        return self.lidar_to_ego(lidar_points, lidar_data)

    def world_from_lidar(self, lidar_points: np.ndarray, frame_data: Mapping) -> np.ndarray:
        """
        :param lidar_points: Nx3 points as np.ndarray
        :return: [world 3D points centered around origin] and [original mean point in world frame]
        """
        # from lidar frame to ego frame
        lidar_data = self.nusc.get('sample_data', frame_data["LIDAR_TOP"])
        ego_points = self.lidar_to_ego(lidar_points, lidar_data)

        # from ego frame to world frame
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        return transform_points_with_pose(ego_points, ego_pose_data)

    def ego_box_from_world(self, bbox: Box, frame_data: Mapping) -> np.ndarray:
        """
        :param bbox: NuScenes.Box object read from annotations in world coordinates
        :return: same box but in ego vehicle frame
        """
        lidar_data = self.nusc.get('sample_data', frame_data["LIDAR_TOP"])
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        inverse_transform_box_with_pose(bbox, ego_pose_data)
        return bbox

    def lidar_to_ego(self, lidar_points: np.ndarray, lidar_data: Mapping) -> np.ndarray:
        lidar_sensor_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        return transform_points_with_pose(lidar_points, lidar_sensor_data)


def transform_points_with_pose(points: np.ndarray, pose_data: Mapping) -> np.ndarray:
    result = points @ Quaternion(pose_data['rotation']).rotation_matrix.T
    result += np.array(pose_data['translation'])
    return result


def inverse_transform_points_with_pose(points: np.ndarray, pose_data) -> np.ndarray:
    result = points.copy()
    result -= np.array(pose_data['translation'])
    return result @ Quaternion(pose_data['rotation']).rotation_matrix


def inverse_transform_box_with_pose(box, pose_data) -> None:
    box.translate(-np.array(pose_data['translation']))
    box.rotate(Quaternion(pose_data['rotation']).inverse)


def project_bbox_to_image(bbox, camera_intrinsic):
    corners = view_points(bbox.corners(), camera_intrinsic, normalize=True)[:2, :]
    return corners.T  # to format as Nx3


# Rotation of [-90] degrees (counter-clockwise) around the x axis
# To bring NuScenes world frame to KITTI rect camera frame to work with Bboxes
ROTATION_NEGATIVE_X = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]], dtype=float)
ROTATION_NEGATIVE_X_FULL = np.array([[1, 0, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1]], dtype=float)


def nuscenes_to_kitti(points):
    """
    Performs left rotation
    :param points: np array Nx3 - to be rotated
    :return: Nx3 after rotation
    """
    return points @ ROTATION_NEGATIVE_X.T  # = np.transpose(R @ points.T)


def kitti_to_nuscenes(points):
    return points @ ROTATION_NEGATIVE_X


def convert_nu_bbox_coordinates_to_kitti(center: np.ndarray, wlh: np.ndarray,
                                         orientation: Quaternion) -> np.ndarray:
    # Keep in mind that NuScenes world frame is different from KITTI rect frame
    # This Bbox class and all other tracking code expects KITTI rect frame
    # See transform/nuscenes and transform/kitti for details on their coordinate frame configs

    angle_around_vertical = -1 * quaternion_yaw(orientation)  # vertical axis points down in KITTI
    center_kitti = nuscenes_to_kitti(center.reshape(1, 3)).reshape(-1,)
    assert center_kitti[0] == center[0], f"original {center}, center_kitti {center_kitti}"
    assert center_kitti[1] == -center[2], f"original {center}, center_kitti {center_kitti}"
    assert center_kitti[2] == center[1], f"original {center}, center_kitti {center_kitti}"

    # Box dimensions:
    # original for Nu: w, l, h describing y, x, z
    # KITTI tracking expects: x, y, z that they call l, h, w
    # but bbox_coordinates expect them in order l, w, h; so  x, z, y
    # need to give x, y, z dimensions from Nu to account for rotation, so l, w, h from Nu's [wlh]
    wlh_kitti = [wlh[1], wlh[0], wlh[2]]

    # KITTI had center of the bottom plate, not center of the box!
    center_kitti[1] = center_kitti[1] + (wlh_kitti[2] * 0.5)
    return np.array([*center_kitti, angle_around_vertical, *wlh_kitti])


def convert_kitti_bbox_coordinates_to_nu(coordinates: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """coordinates: [h, w, l, x, y, z, theta] """

    # Keep in mind that NuScenes world frame is different from KITTI rect frame
    # See transform/nuscenes and transform/kitti for details on their coordinate frame configs
    center_kitti = np.array([coordinates[3], coordinates[4], coordinates[5]])
    center_nu = kitti_to_nuscenes(center_kitti.reshape(1, 3)).reshape(-1,)

    # sizes in KITTI are h,w,l -> y, z, x in KITTI frame
    # Nu expects: y, x, z in Nu frame [wlh]
    # to account for rotation in frames, should give z, x, y
    wlh = np.array([coordinates[1], coordinates[2], coordinates[0]])

    # KITTI had center of the bottom plate, not center of the box!
    center_nu[2] = center_nu[2] + (wlh[2] * 0.5)
    rotation = Quaternion(axis=np.array([0, 0, 1]), radians=-1 * coordinates[6])
    return center_nu, wlh, rotation
