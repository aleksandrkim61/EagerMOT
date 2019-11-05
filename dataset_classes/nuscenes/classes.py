# Detection classes
# barrier	movable_object.barrier
# bicycle	vehicle.bicycle
# bus	vehicle.bus.bendy
# bus	vehicle.bus.rigid
# car	vehicle.car
# construction_vehicle	vehicle.construction
# motorcycle	vehicle.motorcycle
# pedestrian	human.pedestrian.adult
# pedestrian	human.pedestrian.child
# pedestrian	human.pedestrian.construction_worker
# pedestrian	human.pedestrian.police_officer
# traffic_cone	movable_object.trafficcone
# trailer	vehicle.trailer
# truck	vehicle.truck

# Tracking classes
# vehicle.bicycle	bicycle
# vehicle.bus.bendy	bus
# vehicle.bus.rigid	bus
# vehicle.car	car
# vehicle.motorcycle	motorcycle
# human.pedestrian.adult	pedestrian
# human.pedestrian.child	pedestrian
# human.pedestrian.construction_worker	pedestrian
# human.pedestrian.police_officer	pedestrian
# vehicle.trailer	trailer
# vehicle.truck	truck

from typing import List
from enum import Enum


class NuScenesClasses(Enum):
    car = 1
    pedestrian = 2
    bicycle = 3
    bus = 4
    motorcycle = 5
    trailer = 6
    truck = 7


ALL_NUSCENES_CLASS_NAMES: List[str] = [m.name for m in NuScenesClasses]
ALL_NUSCENES_CLASS_IDS: List[int] = [m.value for m in NuScenesClasses]


def name_from_id(class_id: int) -> str:
    return NuScenesClasses(class_id).name


def id_from_name(name: str) -> int:
    return NuScenesClasses[name].value
