from __future__ import annotations
import math
from collections import namedtuple
from typing import Optional

import numpy as np

from transform.transformation import to_homogeneous, inverse_rigid_transform, get_rotation_matrix_around_y
import utils.utils_geometry as utils_geometry
from inputs.bbox import ProjectsToCam, Bbox2d
# import inputs.bbox as inputs_bbox


class Detection2D(ProjectsToCam):
    def __init__(self,
                 bbox: Bbox2d,
                 cam: str,
                 score: float,
                 seg_class_id: int,
                 *,
                 mask=None,
                 mask_decoded=None,
                 reid=None):
        self.bbox = bbox
        self.cam = cam
        self.score = score
        self.seg_class_id = seg_class_id
        self.mask = mask
        self.mask_decoded = mask_decoded
        self.reid = reid

    def bbox_2d_in_cam(self, cam: str) -> Optional[Bbox2d]:
        return self.bbox if cam == self.cam else None
