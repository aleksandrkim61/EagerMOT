from typing import Mapping, Optional
from abc import ABC, abstractmethod

import numpy as np
from numba import njit


class Transformation(ABC):
    """ Transformation object that will handle coordinate frame changes for datasets """

    def __init__(self):
        pass

    @abstractmethod
    def rect_from_lidar(self, lidar_points: np.ndarray, frame_data: Optional[Mapping],
                        only_forward: bool = False) -> np.ndarray:
        """ Get 3D points in ego/rect frame from points in LiDAR coordinates
        :param lidar_points: Nx3 points
        """

    @abstractmethod
    def img_from_tracking(self, track_points: np.ndarray, cam: str,
                          frame_data: Optional[Mapping]) -> np.ndarray:
        """ Get image place coordinates from tracking coordinates i.e. rect KITTI coordinate frame
        For KITTI, this would be img_from_rect
        For NuScenes, tracking coordinates need to be converted back to NuScenes world coordinates,
            then to ego frame, then to cam frame, then projected
        """


@njit
def inverse_rigid_transform(transform):
    """ Inverse a rigid body transform matrix (3x4 as [R|t]) [R'|-R't; 0|1] """
    inverse = np.zeros_like(transform)  # 3x4
    inverse[0:3, 0:3] = transform[0:3, 0:3].T
    inverse[0:3, 3] = (-transform[0:3, 0:3].T) @ transform[0:3, 3]
    return inverse


@njit
def to_homogeneous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def get_rotation_matrix_around_y(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


@njit
def cam_points_to_image_coordinates(img_points):
    """
    :param img_points: nx3 3D points in camera frame coordinates
    :return: nx2 2D coordinates of points in image coordinates
    """
    img_points[:, 0] /= img_points[:, 2]
    img_points[:, 1] /= img_points[:, 2]
    # img_points = img_points[:, :2] / img_points[:, 2].reshape(-1, 1)
    img_plane_points = np.rint(img_points)
    return img_plane_points[:, :2]
