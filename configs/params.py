from typing import Mapping, Any, Dict

# KITTI splits
# Object detection splits
TRAIN_SEQ = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
VAL_SEQ = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
# Trackinng val split, others are train
TRACK_VAL_SEQ = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']
BOTH_VAL_SEQ = ['0001', '0002', '0006', '0007', '0008', '0010',
                '0012', '0013', '0014', '0015', '0016', '0018', '0019']


###################################################################################
# Alternative methods to match 3D tracks/detections in the first matching stage
# 3D IoU: first_matching_method="iou_3d"
# 2D horizontal distance: first_matching_method="dist_2d"
# Euclidean distance on all coordinates except angle: first_matching_method="dist_2d_dims"

def build_params_dict(det_scores, seg_scores, max_ages, min_hits,
                      max_age_2d,
                      fusion_iou_threshold,
                      fusion_mode="bbox",
                      is_angular=False, compensate_ego=True,
                      first_matching_method='iou_3d', iou_3d_threshold=None, thresholds_per_class={},
                      second_matching_method='iou', leftover_matching_thres=0.01,
                      ) -> Dict[str, Any]:
    assert (iou_3d_threshold and not thresholds_per_class) or (not iou_3d_threshold and thresholds_per_class)
    return {"det_scores": det_scores, "seg_scores": seg_scores,
            "fusion_iou_threshold": fusion_iou_threshold,
            "max_ages": max_ages,  # How many frames a track gets to match with something before it is killed, per class
            "min_hits": min_hits,  # How many matches a track needs to get in its lifetime to be considered confirmed
            "fusion_mode": fusion_mode,
            "is_angular": is_angular,  # If add angular velocity to the KF motion model
            "compensate_ego": compensate_ego,  # Compensate for ego motion, track objects in world frame
            "first_matching_method": first_matching_method,
            "iou_3d_threshold": iou_3d_threshold,
            "leftover_matching_thres": leftover_matching_thres,
            "second_matching_method": second_matching_method,
            "thresholds_per_class": thresholds_per_class,
            "max_age_2d": max_age_2d,
            }


def variant_name_from_params(params: Mapping[str, Any]) -> str:
    fusion_str = (f'det_{"_".join(str(score) for score in params["det_scores"])}'
                  f'_seg_{"_".join(str(score) for score in params["seg_scores"])}'
                  f'_{params["fusion_mode"]}_{"_".join(str(thres) for thres in params["fusion_iou_threshold"])}')

    kf_str = f'{"akf" if params.get("is_angular", False) else "kf"}'
    thresholds_per_class = params.get("thresholds_per_class", None)
    thresholds_str = f'[{"_".join(str(i) for i in thresholds_per_class.values())}]' if thresholds_per_class else params["iou_3d_threshold"]
    first_matching_str = f'{params["first_matching_method"]}_{thresholds_str}_{params["leftover_matching_thres"]}'

    max_age_str = f'a{"_".join(str(i) for i in params["max_ages"])}'
    min_hits_str = f'h{"_".join(str(i) for i in params["min_hits"])}'
    max_age_2_str = f'2d_age_{"_".join(str(i) for i in params["max_age_2d"])}'
    return f"{fusion_str}_{kf_str}_{first_matching_str}_{max_age_str}_{min_hits_str}_{max_age_2_str}"


KITTI_BEST_PARAMS = build_params_dict(max_ages=(3, 3),
                                      min_hits=(1, 2),
                                      det_scores=(0, 0),
                                      seg_scores=(0.0, 0.9),
                                      fusion_iou_threshold=(0.01, 0.01),
                                      first_matching_method="dist_2d_full",
                                      thresholds_per_class={1: -3.5,  # car
                                                            2: -0.3  # ped
                                                            },
                                      max_age_2d=(3, 3),
                                      leftover_matching_thres=0.3,
                                      compensate_ego=True)

# car, pedestrian, bicycle, bus, motorcycle, trailer, truck
NUSCENES_BEST_PARAMS = build_params_dict(max_ages=(3, 3, 3, 3, 3, 3, 3),
                                         min_hits=(1, 1, 1, 1, 1, 1, 1),
                                         det_scores=(0, 0, 0, 0, 0, 0, 0),
                                         seg_scores=(0, 0, 0, 0, 0, 0, 0),
                                         fusion_iou_threshold=(0.3, 0.3, 0.3, 0.3, 0.3, 0.01, 0.3),
                                         first_matching_method="dist_2d_full",
                                         thresholds_per_class={1: -7.5,   # car
                                                               2: -1.8,   # pedestrian
                                                               3: -4.4,   # bicycle
                                                               4: -8.15,  # bus
                                                               5: -7.5,   # motorcycle
                                                               6: -4.9,   # trailer
                                                               7: -7.5,   # truck
                                                               },
                                         max_age_2d=(2, 3, 1, 3, 3, 2, 2),
                                         leftover_matching_thres=0.5,
                                         compensate_ego=False)
