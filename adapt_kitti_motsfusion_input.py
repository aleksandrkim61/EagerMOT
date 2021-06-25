import inputs.adapt_input as adapt_input
from configs.local_variables import KITTI_WORK_DIR


def add_detection_info_to_motsfusion_trackrcnn_segmentations(target_sequences):
    for sequence_name in target_sequences:
        print(f'Sequence {sequence_name}')
        adapt_input.add_detection_info_to_motsfusion_trackrcnn_segmentations(sequence_name)


def add_detection_info_to_motsfusion_rrc_segmentations(target_sequences):
    for sequence_name in target_sequences:
        print(f'Sequence {sequence_name}')
        adapt_input.add_detection_info_to_motsfusion_rrc_segmentations(sequence_name)


if __name__ == "__main__":
    kitti = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
                                      det_source=input_utils.POINTGNN_T3,
                                      seg_source=input_utils.TRACKING_BEST)

    add_detection_info_to_motsfusion_rrc_segmentations([str(i).zfill(4) for i in range(29)])
    add_detection_info_to_motsfusion_trackrcnn_segmentations([str(i).zfill(4) for i in range(29)])
