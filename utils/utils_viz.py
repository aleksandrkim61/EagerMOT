import imageio
import cv2
import numpy as np

import colorsys
from typing import Sequence, Iterable, Optional
from math import ceil

import imageio
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np


def draw_bbox(image, bbox_2d, color, thickness=1):
    if bbox_2d is None:
        return
    cv2.rectangle(image,
                  (int(bbox_2d[0]), int(bbox_2d[1])), (int(bbox_2d[2]), int(bbox_2d[3])),
                  color, thickness)


def save_image(image_array, path_to_file, convert_to_uint8=False):
    if convert_to_uint8:
        imageio.imwrite(path_to_file, (image_array[:, :, :3] * 255).astype(np.uint8, copy=False))
    else:
        imageio.imwrite(path_to_file, image_array[:, :, :3])



def show_image(image, path_to_save: Optional[str] = None):
    plt.figure(figsize=(25, 12))
    plt.cla()
    plt.imshow(image[:, :, :3])
    if path_to_save:
        plt.savefig(path_to_save)
    plt.close()


def show_images(fig: plt.Figure, images: Sequence, titles: Iterable, path_to_save: Optional[str] = None, render: bool = True):
    columns = max(min(3, len(images)), 1)
    rows = ceil(len(images) / columns)
    for i, (image, title) in enumerate(zip(images, titles)):
        fig.add_subplot(rows, columns, i + 1).set_title(title)
        plt.imshow(image, aspect="auto")
    
    fig.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save, dpi=300)
    if render:
        plt.show()
        plt.pause(0.01)
    plt.cla()
    plt.clf()
    # plt.close('all')


def create_subplot_figure():
    fig = plt.figure(figsize=(25, 12))
    fig.suptitle("show_images")
    return fig


def generate_colors():
    """
    Aadapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    :return list of colors (each color is a list of len=3)
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6,
            10]
    colors = [list(colors[idx]) for idx in perm]
    return colors


def create_line_bbox_from_vertices(list_of_vertices, color, radius=0.06):
    lines = [[0, 1], [0, 3], [1, 2], [2, 3],  # bottom horizontal rectangle
             [4, 5], [4, 7], [5, 6], [6, 7],  # top horizontal rectangle
             [0, 4], [1, 5], [2, 6], [3, 7]]  # 4 vertical pillars

    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(list_of_vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # return line_set

    cylinders = LineMesh(line_set.points, line_set.lines, line_set.colors, radius=radius)
    return cylinders.cylinder_segments


def plot_box(values, path_to_save):
    fig = plt.figure(figsize=(7, 4))
    axes = fig.add_subplot(1, 1, 1)
    red_square_marker = dict(markerfacecolor='r', marker='s')
    axes.boxplot(values, labels=['car', 'ped'], notch=True, flierprops=red_square_marker)
    plt.savefig(path_to_save, dpi=300)


"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""
import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)
            self.merge_cylinder_segments()

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

    def merge_cylinder_segments(self):
         vertices_list = [np.asarray(mesh.vertices) for mesh in self.cylinder_segments]
         triangles_list = [np.asarray(mesh.triangles) for mesh in self.cylinder_segments]
         triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
         triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]
        
         vertices = np.vstack(vertices_list)
         triangles = np.vstack([triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])
        
         merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices), 
                                                 o3d.open3d.utility.Vector3iVector(triangles))
         color = self.colors if self.colors.ndim == 1 else self.colors[0]
         merged_mesh.paint_uniform_color(color)
         self.cylinder_segments = [merged_mesh]