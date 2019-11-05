from typing import List
from enum import Enum


class KITTIClasses(Enum):
    car = 1
    pedestrian = 2


ALL_KITTI_CLASS_NAMES: List[str] = [m.name for m in KITTIClasses]
ALL_KITTI_CLASS_IDS: List[int] = [m.value for m in KITTIClasses]


def name_from_id(class_id: int) -> str:
    return KITTIClasses(class_id).name


def id_from_name(name: str) -> int:
    return KITTIClasses[name].value
