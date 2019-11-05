from __future__ import annotations
from typing import Optional, Mapping, Any, Tuple, List
import math

import numpy as np
from numba import njit
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R

import inputs.bbox as bbox
from transform.transformation import Transformation, get_rotation_matrix_around_y
from tracking.utils_tracks import correct_new_angle_and_diff


@njit
def box_2d_overlap_union(a, b):
    """ Computes intersection over union for bbox a and b in KITTI format

    :param a, b: Bbox2d (x1, y1, x2, y2)
    :param criterion: what to divide the overlap by - area of union/a, defaults to "union"
    :return: overlap over union/a
    """
    if a is None or b is None:
        return 0.0

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    w = x2 - x1
    h = y2 - y1
    if w <= 0. or h <= 0.:
        return 0.0

    inter = w * h
    aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
    barea = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / float(aarea + barea - inter)


def iou_3d_from_corners(corners1, corners2, dims_1, dims_2):
    """ Compute 3D bounding box IoU.

    :param corners1: numpy array (8,3), assume up direction is negative Y
    :param corners2: numpy array (8,3), assume up direction is negative Y
    :return (iou, iou_2d): (3D bounding box IoU, bird's eye view 2D bounding box IoU)

    """
    # 3D corner points (corners1) are in clockwise order
    # if drawing X-Z plane, Y points into paper
    # 2D corner points (rect1) are in counter clockwise order
    corners1 = corners1.round(4)
    corners2 = corners2.round(4)
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    inter_area = shapely_polygon_intersection(rect1, rect2)
    iou = iou_3d_from_inter_area_corners_dims(inter_area, corners1, corners2, dims_1, dims_2)
    assert iou <= 1.02, f"iou {iou} corners1 {corners1}, corners2 {corners2}"
    return iou


@njit
def iou_3d_from_inter_area_corners_dims(inter_area, corners1, corners2, dims_1, dims_2):
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    return inter_vol / (dims_1.prod() + dims_2.prod() - inter_vol)


def compute_box_3d(x, y, z, l, w, h, yaw=None, rotation_matrix=None):
    """ Converts detection coordinates of 3D bounding box into 8 corners """
    assert yaw is not None or rotation_matrix is not None
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix_around_y(yaw)
    corners_3d = np.dot(rotation_matrix, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return np.transpose(corners_3d)


def convert_bbox_coordinates_to_corners(bbox_coordinates):
    return compute_box_3d(*bbox_coordinates[:3], *bbox_coordinates[4:7], yaw=bbox_coordinates[3])


def shapely_polygon_intersection(poly1, poly2) -> float:
    # Slower than `polygon_clip` but definitely correct
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    return poly1.intersection(poly2).area


@njit
def box_3d_vol(corners):
    """ Compute volume of a 3D bounding box represented with 8 corners

    :param corners: np.array(8, 3) 3D coordinates of its corners
    :return: volume of the bbox
    """
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box_2d_area(box_2d) -> float:
    if box_2d is None:
        return 0
    length = box_2d.x2 - box_2d.x1
    height = box_2d.y2 - box_2d.y1
    return length * height


def project_bbox_3d_to_2d(bbox_3d, transformation: Transformation,
                          img_shape_per_cam: Mapping[str, Any], cam: str,
                          frame_data: Mapping[str, Any]) -> Optional[bbox.Bbox2d]:
    corners = convert_bbox_coordinates_to_corners(bbox_3d.kf_coordinates)
    bbox_projected = transformation.img_from_tracking(corners, cam, frame_data)
    rect_coords = clip_bbox_to_four_corners(bbox_projected, img_shape_per_cam[cam])
    return rect_coords if rect_coords is not None else None


def clip_bbox_to_four_corners(bbox_projected, img_shape_real
                              ) -> Optional[bbox.Bbox2d]:
    def clip(value: float, min_value: float, max_value: float) -> float:
        return min(max(value, min_value), max_value)

    if len(bbox_projected) < 4:
        return None

    x_0 = clip(min(bbox_projected[:, 0]), 0, img_shape_real[1])
    y_0 = clip(min(bbox_projected[:, 1]), 0, img_shape_real[0])
    x_1 = clip(max(bbox_projected[:, 0]), 0, img_shape_real[1])
    y_1 = clip(max(bbox_projected[:, 1]), 0, img_shape_real[0])

    rect_coords = bbox.Bbox2d(x_0, y_0, x_1, y_1)
    if any(i < 0 for i in rect_coords) or x_0 == x_1 or y_0 == y_1:
        return None
    return rect_coords


@njit
def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def angles_from_rotation_matrix(rotation_matrix):
    return R.from_dcm(rotation_matrix).as_euler('xyz', degrees=False)


def bbox_center(bbox):
    return ((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2)  # center of the bbox


@njit
def tracking_center_distance_2d(center_0: np.ndarray, center_1: np.ndarray) -> float:
    return np.linalg.norm(center_0[np.array((0, 2))] - center_1[np.array((0, 2))])


@njit
def tracking_distance_2d_dims(coords_0: np.ndarray, coords_1: np.ndarray) -> float:
    return np.linalg.norm(coords_0[np.array((0, 1, 2, 4, 5, 6))] - coords_1[np.array((0, 1, 2, 4, 5, 6))])


def tracking_distance_2d_full(coords_0: np.ndarray, coords_1: np.ndarray) -> float:
    dist = tracking_distance_2d_dims(coords_0, coords_1)
    _, angle_diff = correct_new_angle_and_diff(coords_0[3], coords_1[3])
    assert angle_diff <= np.pi / 2, f"angle_diff {angle_diff}"
    cos_dist = 1 - np.cos(angle_diff)  # in [0, 1] since angle_diff in [0, pi/2]
    return dist * (1 + cos_dist)  # multiplier is in [1, 2]
