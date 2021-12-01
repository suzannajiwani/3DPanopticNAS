import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from torchvision import models
from torchvision.models import AlexNet

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3


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
        self.semantic_head = models.resnet101(pretrained=True)
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        # resnet variants:
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        # self.semantic_head = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        
        # Instance Head
        # not sure if this is an appropriate head
        self.instance_head = models.resnet101(pretrained=True)
        
    def forward(self, lidar_input):
        output = {}
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        encoder_out = self.encoder(self.pad_to_max_len(lidar_input))
        
        # semantic
        semantic = self.semantic_decoder(encoder_out)
        
        # instance
        instance = self.instance_decoder(encoder_out)

        return output

    def pad_to_max_len(self, lidar):
        return lidar