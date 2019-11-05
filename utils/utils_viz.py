import imageio
import cv2
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
