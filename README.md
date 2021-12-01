# Setup NuScenes dataset
This can be done in terminal with wget
1. Download the metadata from https://www.nuscenes.org/download (go to Full Dataset, then download the metadata for mini/trainval/test)
2. Download the lidarseg and panoptic labels 
3. For each part in trainval and test, download the lidar blob

## NuScenes Resources for Panoptic Segmentation
- (nuscenes_lidarseg_panoptic_tutorial.ipynb)[https://colab.research.google.com/github/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_lidarseg_panoptic_tutorial.ipynb#scrollTo=WKzNbzmHtdY3]
- https://www.nuscenes.org/panoptic
- https://github.com/nutonomy/nuscenes-devkit
- (LidarPointCloud class)[https://github.com/nutonomy/nuscenes-devkit/blob/c44366daea8bba29673943c1fc86d0bfbfb7a99e/python-sdk/nuscenes/utils/data_classes.py#L236(LidarPointCloud]
