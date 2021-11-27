import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn import init
import math


class conv_block(nn.Module): # 0.5*0.5 size
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, img):
        img = self.conv(img)
        return img


class XuNet(nn.Modle):
    def __init__(self, in_ch, out_ch=2):
        super(XuNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            conv_block(in_ch=in_ch, out_ch=16), # 240-120
            conv_block(in_ch=16, out_ch=32), # -60
            conv_block(in_ch=32, out_ch=64), # -30
            conv_block(in_ch=64, out_ch=128), # -15
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True), # N*16*15*15
            nn.Linear(16*15*15, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_ch, bias=True)
        )

    def forward(self, img):
        x = self.cnn_layers.forward(img)
        return x

