from __future__ import annotations
import time
import os
import ujson as json
import argparse
from typing import Dict, List, Mapping

import numpy as np
import open3d as o3d
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import dataset_classes.kitti.mot_kitti as mot_kitti
from dataset_classes.mot_dataset import MOTDataset
from dataset_classes.mot_frame import MOTFrame
from dataset_classes.nuscenes.frame import MOTFrameNuScenes
from dataset_classes.nuscenes.sequence import MOTSequenceNuScenes
from dataset_classes.nuscenes.dataset import MOTDatasetNuScenes
from inputs.bbox import Bbox3d
from utils import utils_viz, io
from configs.params import NUSCENES_BEST_PARAMS
from configs.local_variables import SPLIT, NUSCENES_WORK_DIR, MOUNT_PATH
from transform.nuscenes import ROTATION_NEGATIVE_X_FULL
import transform.nuscenes as transform_nu
import inputs.utils as input_utils


VIS_HEIGHT = int(2 * 1200 / 3)
VIS_WIDTH = int(2 * 1920 / 3)


def refresh_visualizer(vis):
    vis.poll_events()
    vis.update_renderer()


def set_view_control(view_control):
    # Press P to take a screen capture: screenshot + json with params
    # view_control_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # print("view_control_params")
    # print(view_control_params.extrinsic)
    # print(view_control_params.intrinsic.get_focal_length())
    # print(view_control_params.intrinsic.get_principal_point())
    # print(view_control_params.intrinsic.width)
    # print(view_control_params.intrinsic.height)
    # print()

    extrinsic = np.array([[ 0.83659957, -0.54698259,  0.03018616, -8.59162573],
        [-0.28629649, -0.48353448, -0.8271812,   6.42272306],
        [ 0.46704976,  0.68337724, -0.56112393, -0.71447277],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    focal = (887.6760388790497, 887.6760388790497)
    principal_point = (959.5, 512.0)
    width_height = (1920, 1025)
    pinhole_params = o3d.camera.PinholeCameraParameters()
    pinhole_params.extrinsic = extrinsic
    # http://www.open3d.org/docs/0.14.1/python_api/open3d.camera.PinholeCameraIntrinsic.html?highlight=pinholecameraintrinsic#open3d.camera.PinholeCameraIntrinsic.set_intrinsics
    pinhole_params.intrinsic.set_intrinsics(*width_height, *focal, *principal_point)
    view_control.convert_from_pinhole_camera_parameters(pinhole_params)


def visualize_3d(dataset, params, mode: str, output_folder_prefix: str,
                 target_sequences=None, target_frame=0,
                 result_json: str = '', radius=0.05,
                 slowdown=None, save_img=False, world: bool = False, show_origin=False):
    is_first_sequence = True
    view_control_params = None
    all_colors = utils_viz.generate_colors()

    origin_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    # The x, y, z axis will be rendered as red, green, and blue arrows respectively

    if result_json:
        with open(result_json, 'r+') as f:
            tracking_results = json.load(f)["results"]

    if not target_sequences:
        target_sequences = dataset.sequence_names(SPLIT)

    for sequence_name in target_sequences:

        print(f'Sequence {sequence_name}')
        sequence = dataset.get_sequence(SPLIT, sequence_name)
        sequence.mot.set_track_manager_params(params)
        current_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        img_folder_path = os.path.join(
            f"{output_folder_prefix}_{mode}_{SPLIT}_rad{radius}_{'CP' if ('/centerpoint_val/') in result_json else 'ours'}_{current_time_str}", sequence_name)
        if save_img:
            io.makedirs_if_new(img_folder_path)
            if result_json:
                with open(f"{img_folder_path}/{(result_json.split('/')[-1]).split('.json')[0]}.txt", "w") as file:
                    file.write(f"{result_json} was rendered here.")
                    file.close()

        # Init 3D viz window
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=VIS_WIDTH, height=VIS_HEIGHT)

        all_frames = sequence.frame_names
        assert len(all_frames) > 1
        is_first_frame = True

        sequence.load_ego_motion_transforms()
        for frame_i, frame_name in enumerate(all_frames[target_frame:]):
            frame_i += target_frame
            if frame_i % 10 == 0:
                print(f'Processing frame {frame_name}')
            frame = sequence.get_frame(frame_name)

            if mode == "mot":
                geometries_to_add = vis_mot_predictions(frame, params, all_colors, world)
            elif mode == "mot_submitted":
                assert isinstance(sequence, MOTSequenceNuScenes)
                geometries_to_add = vis_mot_predictions_submitted(frame, params, all_colors, tracking_results, radius, world)
            elif mode == "det":
                geometries_to_add = vis_detections_3d(frame, params, all_colors, world)
            elif mode == "annotations":
                geometries_to_add = vis_annotations_3d(frame, all_colors, world)
            else:
                raise Exception(f"Unknown visualization mode {mode}")

            # Add geometries to the visualizer
            vis.clear_geometries()
            if show_origin:
                vis.add_geometry(origin_frame_mesh)
            for geometry in geometries_to_add:
                if world:
                    geometry.translate(-frame.center_world_point.reshape(3, 1))
                vis.add_geometry(geometry)

            # misc. visualizer-related code
            if not is_first_frame or not is_first_sequence:  # apply saved view settings
                vis.get_view_control().convert_from_pinhole_camera_parameters(view_control_params)
            else:
                set_view_control(vis.get_view_control())
                vis.get_render_option().load_from_json("render_option.json")
                is_first_frame = is_first_sequence = False

            refresh_visualizer(vis)
            if save_img:
                frame_i_str = str(frame_i).zfill(6)
                vis.capture_screen_image(f'{img_folder_path}/{frame_i_str}_{frame_name}.png', do_render=False)
            vis.run()

            view_control_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if slowdown is not None:
                time.sleep(slowdown)

    vis.destroy_window()


def vis_mot_predictions(frame: MOTFrame, params: Mapping, all_colors,
                        world: bool = False) -> List[o3d.geometry.Geometry3D]:
    if isinstance(frame, mot_kitti.MOTFrameKITTI):
        assert not world
    if isinstance(frame, MOTFrameNuScenes):
        assert world

    points = frame.points(world)
    predicted_instances = frame.perform_tracking(params)

    predicted_bboxes: List[Bbox3d] = []

    for instance in predicted_instances:  # x, y, z, theta, l, w, h, ID, other info , confidence
        if not instance.report_mot:
            continue
        bbox3d_coords = instance.coordinates_3d
        assert bbox3d_coords is not None
        ordered_bbo3d_coord = bbox3d_coords[[3, 4, 5, 6, 2, 1, 0]]
        predicted_bboxes.append(Bbox3d(ordered_bbo3d_coord, instance.track_id,
                                       seg_class_id=instance.class_id))

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.nuscenes_to_kitti(points)

    colors_np = np.full_like(points, np.array([0, 0, 0]))
    geometries_to_add: List[o3d.geometry.Geometry3D] = []
    for bbox in predicted_bboxes:
        track_id_full = 1000 * bbox.seg_class_id + bbox.instance_id
        color = all_colors[track_id_full % len(all_colors)]

        indices_points_inside = bbox.get_indices_of_points_inside(points)
        colors_np[indices_points_inside] = color

        if isinstance(frame, MOTFrameNuScenes):
            bbox.inverse_transform(ROTATION_NEGATIVE_X_FULL, 0)

        geometries_to_add.append(utils_viz.create_line_bbox_from_vertices(bbox.corners_3d, color))

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.kitti_to_nuscenes(points)

    current_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    current_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    geometries_to_add.append(current_pcd)
    return geometries_to_add


def vis_mot_predictions_submitted(frame: MOTFrame, params: Mapping, all_colors, tracking_results, radius,
                                 world: bool = False) -> List[o3d.geometry.Geometry3D]:
    assert world
    assert isinstance(frame, MOTFrameNuScenes)

    points = frame.points(world)
    colors_np = np.full_like(points, np.array([0, 0, 0]))
    geometries_to_add: List[o3d.geometry.Geometry3D] = []

    frame_results = tracking_results[frame.name]
    print(f"frame {frame.name} has {len(frame_results)} results")
    boxes_nu = [Box(result["translation"], result["size"], Quaternion(result["rotation"]), int(result["tracking_id"]),
                    name=result["tracking_name"], token=result["sample_token"]) for result in frame_results]
    # bboxes_3d = [frame.bbox_3d_from_nu(box, box.label, world=True) for box in boxes_nu]

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.nuscenes_to_kitti(points)

    for bbox in boxes_nu:
        bbox_label = bbox.label
        color = all_colors[bbox_label % len(all_colors)]
        # print(f"bbox.center {bbox.center}")
        # print(f"bbox.wlh {bbox.wlh}")
        # print(f"bbox.orientation.q {bbox.orientation.q}")
        # det_info = tuple(np.concatenate((bbox.center, bbox.wlh, bbox.orientation.q)))
        # color = all_colors[hash(det_info) % len(all_colors)]
        # print(f"det_info: {det_info}, hash: {hash(det_info)}")

        # bbox_internal = Bbox3d.from_nu_box_convert(bbox)
        bbox_internal = frame.bbox_3d_from_nu(bbox, bbox_label, world=True)
        indices_points_inside = bbox_internal.get_indices_of_points_inside(points)
        colors_np[indices_points_inside] = color

        geometries_to_add.append(*utils_viz.create_line_bbox_from_vertices(bbox.corners().T, color, radius))

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.kitti_to_nuscenes(points)

    current_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    current_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    geometries_to_add.append(current_pcd)
    return geometries_to_add


def vis_detections_3d(frame: MOTFrame, params: Mapping, all_colors,
                      world: bool = False) -> List[o3d.geometry.Geometry3D]:
    frame.det_score_thresholds = params["det_scores"]
    frame.seg_score_thresholds = params["seg_scores"]

    points = frame.points(world)
    bboxes_3d = frame.detections_3d(world)

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.nuscenes_to_kitti(points)

    geometries_to_add: List[o3d.geometry.Geometry3D] = []
    colors_np = np.full_like(points, np.array([0, 0, 0]))

    for bbox in bboxes_3d:
        if bbox is None:
            continue

        color = all_colors[bbox.seg_class_id % len(all_colors)]
        indices_points_inside = bbox.get_indices_of_points_inside(points)
        colors_np[indices_points_inside] = color

        if isinstance(frame, MOTFrameNuScenes):
            bbox.inverse_transform(ROTATION_NEGATIVE_X_FULL, 0)

        geometries_to_add.append(utils_viz.create_line_bbox_from_vertices(bbox.corners_3d, color))

    if isinstance(frame, MOTFrameNuScenes):
        points = transform_nu.kitti_to_nuscenes(points)

    current_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    current_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    geometries_to_add.append(current_pcd)
    return geometries_to_add


def vis_annotations_3d(frame: MOTFrame, all_colors, world: bool = False) -> List[o3d.geometry.Geometry3D]:
    points = frame.points(world)

    if isinstance(frame, MOTFrameNuScenes):  # to enable get_indices_of_points_inside
        points = transform_nu.nuscenes_to_kitti(points)
    colors_np = np.full_like(points, np.array([0, 0, 0]))

    geometries_to_add: List[o3d.geometry.Geometry3D] = []
    bboxes = frame.bbox_3d_annotations(world)
    for bbox in bboxes:
        if bbox is None:
            continue

        color = all_colors[bbox.seg_class_id % len(all_colors)]
        indices_points_inside = bbox.get_indices_of_points_inside(points)
        colors_np[indices_points_inside] = color

        if isinstance(frame, MOTFrameNuScenes):
            bbox.inverse_transform(ROTATION_NEGATIVE_X_FULL, 0)

        geometries_to_add.append(utils_viz.create_line_bbox_from_vertices(bbox.corners_3d, color))

    if isinstance(frame, MOTFrameNuScenes):
        points = transform_nu.kitti_to_nuscenes(points)

    current_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    current_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    geometries_to_add.append(current_pcd)
    return geometries_to_add


def visualize_tracking_results_in_2d(dataset: MOTDataset, params: Dict, 
        result_json: str, output_folder_prefix: str, target_sequences=None,
        default_cam_only: bool = False, save_img: bool = False, render: bool = True, suffix: str = ""):
    all_colors = utils_viz.generate_colors()

    if not target_sequences:
        target_sequences = dataset.sequence_names(SPLIT)
    print(target_sequences)

    current_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if save_img:
        img_folder_path = f"{output_folder_prefix}_{SPLIT}_{current_time_str}{suffix}"
        img_folder_path = f"_{SPLIT}_{current_time_str}{suffix}"
        io.makedirs_if_new(img_folder_path)

    with open(result_json, 'r+') as f:
        tracking_results = json.load(f)["results"]

    fig = utils_viz.create_subplot_figure()

    for sequence_name in target_sequences:
        print(f'Sequence {sequence_name}')
        sequence = dataset.get_sequence(SPLIT, sequence_name)
        if save_img:
            sequence_folder_path = os.path.join(img_folder_path, sequence_name)
            io.makedirs_if_new(sequence_folder_path)

        all_frames = sequence.frame_names

        for frame_i, frame_name in enumerate(all_frames):
            frame = sequence.get_frame(frame_name)
            frame_i_str = str(frame_i).zfill(6)

            frame.det_score_thresholds = params["det_scores"]
            frame.seg_score_thresholds = params["seg_scores"]

            frame_results = tracking_results[frame.name]
            print(f"frame {frame_i_str}: {frame_name} has {len(frame_results)} results")
            boxes_nu = [Box(result["translation"], result["size"], Quaternion(result["rotation"]), int(result["tracking_id"]),
                            name=result["tracking_name"], token=result["sample_token"]) for result in frame_results]
            bboxes_3d = [frame.bbox_3d_from_nu(box, box.label, world=True) for box in boxes_nu]
            track_dets_2d_multicam: Dict[str, List] = frame.bbox_2d_projections(bboxes_3d)
            # print("boxes_nu", boxes_nu)
            # print("bboxes_3d", bboxes_3d)
            # print("track_dets_2d_multicam", track_dets_2d_multicam)

            images = []
            for cam in sequence.cameras:
                if default_cam_only and cam != sequence.camera_default:
                    continue

                img = frame.get_image_original_uint8(cam)
                dets = track_dets_2d_multicam[cam]
                for det in dets:
                    color = [c * 255 for c in all_colors[det.instance_id % len(all_colors)]]
                    utils_viz.draw_bbox(img, det.bbox, tuple(color), 2)
                images.append(img)

            path_to_save = os.path.join(sequence_folder_path, f"{frame_i_str}_{frame_name}.png") if save_img else None
            utils_viz.show_images(fig, images, sequence.cameras, path_to_save, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('-suffix', type=str, help='Suffix for the output visualizer folder')
    parser.add_argument('--default_cam_only', default=False, action='store_true')
    args = parser.parse_args()
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    mot_dataset = MOTDatasetNuScenes(work_dir=NUSCENES_WORK_DIR,
                                     det_source=input_utils.CENTER_POINT,
                                     seg_source=input_utils.MMDETECTION_CASCADE_NUIMAGES,
                                     version="v1.0-mini")
    params_to_run = NUSCENES_BEST_PARAMS
    target_sequences: List[str] = ["scene-0103"]

    # mot_dataset = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
    #                                         det_source=input_utils.POINTGNN_T3,
    #                                         seg_source=input_utils.TRACKING_BEST)
    # params_to_run = KITTI_BEST_PARAMS
    # target_sequences: List[str] = ["0027"]

    json_to_parse = os.path.join(MOUNT_PATH, args.result_path)

    visualize_3d(mot_dataset, params_to_run, mode="mot_submitted", output_folder_prefix='output/polarmot', world=True, 
        target_sequences=target_sequences, target_frame=20, result_json=json_to_parse, radius=0.06, save_img=True)
    # visualize_3d(mot_dataset, params_to_run, mode="det", output_folder_prefix='output/polarmot', world=False, target_sequences=target_sequences, save_img=True)
    # visualize_3d(mot_dataset, params_to_run, mode="mot", output_folder_prefix='output/polarmot', world=True, target_sequences=target_sequences, save_img=True)
    # visualize_3d(mot_dataset, params_to_run, mode="annotations", output_folder_prefix='output/polarmot', world=True, target_sequences=target_sequences, save_img=True)

    # visualize_tracking_results_in_2d(mot_dataset, params_to_run, json_to_parse, output_folder_prefix='/output/polarmot',
    #                                  default_cam_only=args.default_cam_only, save_img=True, render=False, suffix=args.suffix)

