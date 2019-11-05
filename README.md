# 3D Multi-Object Tracking and Segmentation from 3D detections and 2D segmentations
### [Work in progress]
#### The project was done as a guided research module at TUM

An online method to combine 3D bounding box detections and 2D segmentation masks to perform Multi-Object Tracking and Segmentation using point fusion to recover missing masks and description vectors to resolve ambiguities in association.
The repo also includes visualization code to view results of multi object tracking in 3D and resulting segmentation masks projected onto original images.

Inputs to the pipeline: 3D bounding box detections, stereo images, 2D segmenation masks and their description vectors.

Source of 3D detections: [Point-RCNN](https://github.com/sshaoshuai/PointRCNN)
Source of 2D segmentations and description: [Track-RCNN](https://www.vision.rwth-aachen.de/page/mots)

By combining these two signals, performance on both tasks has improved over other methods using a single source:
##### Multi-Object Tracking (MOT): 
| Method | sAMOTA | AMOTA | AMOTP | Single threshold: | MOTA |Recall | Precision | IDs | 
| :--- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| AB3DMOT | 0.9143 | 0.4400 | 0.8461 |  | 0.8279 | 0.9198 | 0.9357 | 2 |
| Ours | 0.9240 | 0.4662 | 0.8865 |  | 0.8205 | 0.9338 | 0.9183 | 2 |

Multi-Object Tracking and Segmentation (MOTS):
| Method | sMOTSA | MOTSA | MOTSP | Recall | Precision | IDs | 
| :--- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| TrackRCNN | 76.2 | 87.8 | **88.9** | 90.6 | **98.2** | 93 |
| Ours | **77.6** | **89.9** | 86.8 | 93.1 | 97.5 | 72 |
|  |
| constr p2p -> p2p | **77.6** | **89.9** | 86.7 | 93.2 | 97.5 | 70 |
| 60pts -> 30pts | 77.2 | 89.6 | 86.7 | **93.4** | 96.9 | 68 |
| no early report | 77.2 | 89.4 | 86.8 | 92.3 | 97.7 | **58** |
| no point masks | 76.7 | 88.6 | 87 | 91.6 | 97.9 | 79 |
| No association vectors | 77.1 | 89.4 | 86.8 | 92.8 | 97.4 | 81 |
| Max age = 6 | 77.2 | 89.6 | 86.7 | 93.3 | 97.1 | 76 |
| 3D Iou 0.1 | 77.4 | 89.8 | 86.8 | 93.2 | 97.4 | 73 |
|  |
| point masks instead of seg | 55.2 | 70.4 | 79.5 | 73.6 | 96.2 | 32 |














