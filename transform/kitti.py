import os

import numpy as np
from numba import njit
from transform.transformation import Transformation, to_homogeneous, cam_points_to_image_coordinates


class TransformationKitti(Transformation):
    """
    Calibration matrices and utils
    objects XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in lidar coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_lidar_to_cam * x_lidar
    x_ref = Tr_lidar_to_cam * x_lidar
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
             = K * [1|t]

    image2 coord:
     ----> x-axis (u)
    |
    |
    v y-axis (v)

    lidar coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    Description credit to https://github.com/xinshuoweng/AB3DMOT
    """

    def __init__(self, prefix, sequence, calib_folder='calib'):
        super().__init__()
        file = os.path.join(prefix, calib_folder, sequence + ".txt")
        with open(file) as f:
            self.P0 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P1 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Projection matrix from rectified camera coord to image2/3 coord
            self.P2 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P3 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Rotation from reference camera coord to rectified camera coord
            line = f.readline()
            self.R_rect = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 3))

            # Rigid transform from lidar coord to reference camera coord
            line = f.readline()
            self.Tr_lidar_to_cam = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))

            line = f.readline()
            self.Tr_imu_to_lidar = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))

    """ Transformations are applied in reverse order to maintain nx3 format """

    def img_from_tracking(self, track_points, cam="image_02", frame_data=None):
        """
        :param track_points: nx3 3D points in rectified reference camera coordinates
        :param camera: to which camera plane perform the projection
        :return: nx2 2D coordinates of points in the specified camera's image coordinates
        """
        if cam == "image_02":
            img_points = dot_product(to_homogeneous(track_points), self.P2.T)  # nx3
        elif cam == "image_03":
            img_points = dot_product(to_homogeneous(track_points), self.P3.T)  # nx3
        else:
            raise NotImplementedError(f"Unknown cam {cam}")
        return cam_points_to_image_coordinates(img_points).astype(int, copy=False)

    def rect_from_ref_cam(self, ref_points, only_forward=False):
        """
        :param ref_points: nx3 3D points in reference camera coordinates
        :param only_forward: whether to clip points that are behind the image plane
        :return: nx3 3D points in rectified reference camera coordinates
        """
        rect_points = dot_product(ref_points, self.R_rect.T)
        if only_forward:
            rect_points = rect_points[rect_points[:, 2] > 0]
        return rect_points

    def ref_cam_from_lidar(self, lidar_points):
        """
        :param lidar_points: nx3 3D points in lidar coordinates
        :return: nx3 3D points in reference camera coordinates
        """
        return dot_product(to_homogeneous(lidar_points), self.Tr_lidar_to_cam.T)

    def rect_from_lidar(self, lidar_points, frame_data=None, only_forward=False):
        """
        :param lidar_points: nx3 3D points in lidar coordinates
        :param only_forward: clip points that are behind the image plane
        :return: nx3 3D points in rectified reference camera coordinates
        """
        return self.rect_from_ref_cam(self.ref_cam_from_lidar(lidar_points), only_forward)

    def img_from_lidar(self, lidar_points, cam="image_02"):
        """
        :param lidar_points: nx3 3D points lidar coordinates
        :param camera: to which camera plane perform the projection
        :return: nx2 2D points in the specified camera's image coordinates
        """
        return self.img_from_tracking(self.rect_from_lidar(lidar_points, only_forward=True), cam=cam)


@njit
def dot_product(first, second):
    return first @ second
