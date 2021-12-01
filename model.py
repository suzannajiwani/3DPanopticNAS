import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from torchvision import models
from torchvision.models import AlexNet

from kpconv.blocks import KPConv


class Panoster(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(self.cfg.DATASET.MAX_LIDAR_POINTS, 2048), nn.ReLU(), nn.Linear(2048, 1024), nn.ReLU(), 
                                     nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3))
        
        # Decoders
        self.semantic_decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), 
                                              nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 2028), nn.ReLU(), nn.Linear(2048, self.cfg.DATASET.MAX_LIDAR_POINTS))
        self.instance_decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), 
                                              nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 2028), nn.ReLU(), nn.Linear(2048, self.cfg.DATASET.MAX_LIDAR_POINTS))

        # Semantic Head
#         self.semantic_head = models.resnet101(pretrained=True)
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
#         # resnet variants:
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
#         # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        
#         # Instance Head
#         # not sure if this is an appropriate head
#         self.instance_head = models.resnet101(pretrained=True)
        
    def forward(self, lidar_input):
        print("in forward!")
        output = {}

        print(lidar_input)
        encoder_out = self.encoder(lidar_input)
        
        # semantic
#         semantic = self.semantic_decoder(encoder_out)
        
#         # instance
#         instance = self.instance_decoder(encoder_out)

        return encoder_out
    