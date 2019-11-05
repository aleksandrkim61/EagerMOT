from typing import Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

PI = np.pi
TWO_PI = 2 * np.pi


def normalize_angle(angle: float) -> float:
    """ Keep the angle in [0; 2 PI] range"""
    while angle < 0:
        angle += TWO_PI
    while angle > TWO_PI:
        angle -= TWO_PI
    assert angle >= 0 and angle <= TWO_PI, f"angle {angle}"
    return angle


def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float) -> Tuple[float, float]:
    """ Return an angle equivalent to the new_angle_to_correct with regards to difference to the current_angle
    Calculate the difference between two angles [-PI/2, PI/2]
    
    TODO: This can be refactored to just return the difference 
    and be compatible with all angle values without worrying about quadrants, but this works for now
    """
    abs_diff = normalize_angle(new_angle_to_correct) - normalize_angle(current_angle)

    if abs(abs_diff) <= PI / 2:  # if in adjacent quadrants
        return new_angle_to_correct, abs_diff

    if abs(abs_diff) >= 3 * PI / 2:  # if in 1st and 4th quadrants and the angle needs to loop around
        abs_diff = TWO_PI - abs(abs_diff)
        if current_angle < new_angle_to_correct:
            return current_angle - abs_diff, abs_diff
        else:
            return current_angle + abs_diff, abs_diff

    # if the difference is > PI/2 and the new angle needs to be flipped
    return correct_new_angle_and_diff(current_angle, PI + new_angle_to_correct)


def default_kf_3d(is_angular: bool) -> KalmanFilter:
    if is_angular:  # add angular velocity to the state vector
        kf = KalmanFilter(dim_x=11, dim_z=7)
        kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    else:
        kf = KalmanFilter(dim_x=10, dim_z=7)  # [x,y,z,theta,l,w,h] + [vx, vy, vz]
        kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

    # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    kf.P[7:, 7:] *= 1000.
    kf.P *= 10.
    kf.Q[7:, 7:] *= 0.01
    kf.R *= 0.01  # measurement uncertainty (own addition)
    return kf
