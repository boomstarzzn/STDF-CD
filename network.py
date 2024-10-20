import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import timm
from matplotlib import pyplot as plt

from elsemodels.pvtv2 import pvt_v2_b1
from xlstmnet.conv_dnm import conv_dnm
from xlstmnet.myxLSTM import STxlstmfs, MultiLayerGuidedDNM



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class predetails(nn.Module):
    def __init__(self):
        super(predetails, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x



class STDF_CD(nn.Module):
    def __init__(self, ):
        super(STDF_CD, self).__init__()
        # self.preblock = predetails()
        self.backbone = pvt_v2_b1()  # [64, 128, 320, 512]
        path = './pretrained/pvt_v2_b1.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.st1 = STxlstmfs(128)
        self.st2 = STxlstmfs(320)
        self.st3 = STxlstmfs(512)

        self.conv_reduce_1 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(320 * 2, 320, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512 * 2, 512, 3, 1, 1)

        self.decoder = BasicConv2d(512, 64, 3, 1, 1)
        self.decoder_x = nn.Conv2d(64, 1, 3, 1, 1)

        self.decoder1 = BasicConv2d(320, 64, 3, 1, 1)
        self.decoder1_x = nn.Conv2d(64, 1, 3, 1, 1)

        self.decoder2 = BasicConv2d(128, 64, 3, 1, 1)
        self.decoder2_x = nn.Conv2d(64, 1, 3, 1, 1)

        self.decoder_final = nn.Sequential(BasicConv2d(64, 32, 3, 1, 1), nn.Conv2d(32, 1, 1))
        self.Dnm_conv2d = MultiLayerGuidedDNM(64, 64)
        self.Dnm_final = conv_dnm(64, 1, 10)
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B):
        # A = self.preblock(A)
        # B = self.preblock(B)
        size = A.size()[2:]
        pvt = self.backbone(A)
        pvt_img2 = self.backbone(B)

        layer2 = self.st1(pvt[1], pvt_img2[1], 32, 32)
        layer3 = self.st2(pvt[2], pvt_img2[2], 16, 16)
        layer4 = self.st3(pvt[3], pvt_img2[3], 8, 8)

        layer1 = torch.cat((pvt[0], pvt_img2[0]), dim=1)
        layer1 = self.conv_reduce_1(layer1)

        feature_map = F.interpolate(self.decoder(layer4), layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_map1 = F.interpolate(self.decoder1(layer3), layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_map2 = F.interpolate(self.decoder2(layer2), layer1.size()[2:], mode='bilinear', align_corners=True)

        final_dnm = self.Dnm_conv2d(layer1,feature_map,feature_map1,feature_map2)
        final_map = self.Dnm_final(final_dnm)

        feature_map = self.decoder_x(feature_map)
        feature_map1 = self.decoder1_x(feature_map1)
        feature_map2 = self.decoder2_x(feature_map2)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)
        feature_map = F.interpolate(feature_map, size, mode='bilinear', align_corners=True)
        feature_map1 = F.interpolate(feature_map1, size, mode='bilinear', align_corners=True)
        feature_map2 = F.interpolate(feature_map2, size, mode='bilinear', align_corners=True)

        return feature_map, final_map, feature_map1, feature_map2


