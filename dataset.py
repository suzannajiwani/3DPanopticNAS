from pathlib import Path
import os
import ipdb
import pandas as pd
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset
import numpy as np
import torch
import cloudpickle
from PIL import Image
from nuscenes import NuScenes
import torchvision

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.helper import angle_of_rotation
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from trainer import TrainingModule
from torch_geometric.data import Data, Dataset


def prepare_dataloaders(cfg):
    version = cfg.DATASET.VERSION

#     dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
    nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=cfg.DATASET.DATAROOT, verbose=False)

    traindata = PanopDataset(nusc, True, cfg)
    valdata = PanopDataset(nusc, False, cfg)

    if cfg.DATASET.VERSION == 'mini':
        traindata.indices = traindata.indices[:10]
        valdata.indices = valdata.indices[:10]

    nworkers = cfg.N_WORKERS
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)

    return trainloader, valloader

class PanopDataset(Dataset):
    def __init__(self, dataroot, is_train):
        self.dataroot = dataroot
        self.is_train = is_train
        
        self.MAX_LIDAR_POINTS = 69
        self.DATAROOT = dataroot #'/nobackup/users/sjiwani/nuscenes-dataset'
        self.VERSION = 'trainval'
        self.nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True)

        self.mode = 'train' if self.is_train else 'val'
        self.sequence_length = 1
        
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {True: 'train', False: 'val'},
                'v1.0-mini': {True: 'mini_train', False: 'mini_val'}}[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)
    
    def pad_to_max_len(self, lidar):
        max_len = self.MAX_LIDAR_POINTS
        cur_len = lidar.shape[0]
        if max_len > cur_len:
            return np.pad(lidar, ((0,max_len-cur_len), (0,0)), 'constant', constant_values=(0,0))
        else:
            return lidar

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def len(self):
        return len(self.indices)
    
    def get(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                lidar: torch.Tensor<float> [N, 4], where N is the number of points
                labels: torch.Tensor<float> 

        """
        data = {}
        keys = ['lidar', 'labels']
        for key in keys:
            data[key] = []

        # Loop over all the frames in the sequence.
        print(self.indices[index])
        for index_t in self.indices[index]:
            rec = self.ixes[index_t]

            lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
#             lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
#             yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
#             lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
#             lidar_translation = np.array(lidar_pose['translation'])[:, None]
#             lidar_to_world = np.vstack([
#                 np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
#                 np.array([0, 0, 0, 1])
#             ])
            lidar_filename = f'{self.DATAROOT}/{lidar_sample["filename"]}'
    
            if self.is_train:
                # this means that labels exist
                lidar_token = rec['data']['LIDAR_TOP']
                label_filename = f'{self.DATAROOT}/panoptic/v1.0-{self.VERSION}/{lidar_token}_panoptic.npz'
                panoptic_label_arr = load_bin_file(label_filename, 'panoptic')
                data["labels"].append(panoptic_label_arr)
                points = LidarPointCloud.from_file(lidar_filename).points.T
                # pc = LidarSegPointCloud(lidar_filename, label_filename)
                # points = pc.points
                
            else:
                points = LidarPointCloud.from_file(lidar_filename).points.T
            

            data['lidar'].append(self.pad_to_max_len(points))

        # return data
        lidar = np.asarray(data["lidar"][0])
        labels = np.asarray(data["labels"][0])
        return Data(points=lidar[:, :3], x=lidar[:,3], y=labels)


if __name__ == '__main__':
    ## feel free to copy the following lines in terminal if you don't have any of the nuscenes dataset and want to try
    # mkdir -p /data/sets/nuscenes  # Make the directory to store the nuScenes dataset in.
    # wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.
    # wget https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2  # Download the nuScenes-lidarseg mini split.
    # tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.
    # tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2 -C /data/sets/nuscenes   # Uncompress the nuScenes-lidarseg mini split.
    # pip install nuscenes-devkit &> /dev/null  # Install nuScenes.
    # wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.
    # wget https://www.nuscenes.org/data/nuScenes-panoptic-v1.0-mini.tar.gz  # Download the Panoptic nuScenes mini split.
    # tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.
    # tar -xf nuScenes-panoptic-v1.0-mini.tar.gz -C /data/sets/nuscenes   # Uncompress the Panoptic nuScenes mini split.
    
    dataroot = '/nobackup/users/sjiwani/nuscenes-dataset'
    version = 'trainval'
    #trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    trainer = TrainingModule()

    device = torch.device('cuda:0')
    trainer = trainer.to(device)
    trainer.eval()

    model = trainer.model

    cfg = model.cfg
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version
    
    # test
    #nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    dataset = PanopDataset(dataroot, True)
    i = 0
    for d in dataset:
        # print(d)
        lidar = d['lidar']
        labels = d['labels']
        
        # print(len(lidar), len(labels))
        print(lidar[0].shape, labels[0].shape)
        
        i += 1
        if i == 10:
            break
        
