from typing import Optional, IO, Mapping, Any, Iterable, List
from objects.fused_instance import FusedInstance
import inputs.detections_2d as detections_2d


def write_to_mot_file(frame_name: str, predicted_instances: Iterable[FusedInstance],
                      mot_3d_file: IO,
                      mot_2d_from_3d_only_file: Optional[IO]) -> None:
    mot_3d_results_str, mot_2d_results_str = "", ""
    tracking_3d_format = "%d %d %s 0 0 %f -1 -1 -1 -1 %f %f %f %f %f %f %f %f\n"
    tracking_2d_format = "%d %d %s 0 0 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n"

    for instance in predicted_instances:
        if not instance.report_mot:
            continue

        assert instance.class_id is not None
        track_type = detections_2d.SEG_TO_TRACK_CLASS[instance.class_id]

        bbox3d_coords = instance.coordinates_3d
        if bbox3d_coords is not None:
            bbox3d = instance.bbox3d

            res_3d = (tracking_3d_format % (int(frame_name), instance.track_id, track_type, bbox3d.obs_angle,
                                            bbox3d_coords[0], bbox3d_coords[1], bbox3d_coords[2],
                                            bbox3d_coords[3], bbox3d_coords[4], bbox3d_coords[5], bbox3d_coords[6], bbox3d.confidence))
            mot_3d_results_str += res_3d

        if mot_2d_from_3d_only_file is not None:
            bbox2d = instance.projected_bbox_3d
            if bbox2d is not None:
                res_2d = (tracking_2d_format % (int(frame_name), instance.track_id, track_type,
                                                bbox2d[0], bbox2d[1], bbox2d[2], bbox2d[3], instance.bbox3d.confidence))
                mot_2d_results_str += res_2d

    mot_3d_file.write(mot_3d_results_str)
    if mot_2d_from_3d_only_file is not None:
        mot_2d_from_3d_only_file.write(mot_2d_results_str)
