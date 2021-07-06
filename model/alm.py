import torch
import torch.nn as nn
# import torch.tensor as tensor
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url
from model.bninception import BNInception
from model.resnet import resnet50
import os

'''
Refer to: https://github.com/chufengt/iccv19_attribute/blob/master/model/inception_iccv.py
'''

def inception_iccv(pretrained=True, debug=False, **kwargs):
    model = InceptionNet(**kwargs)
    """
        pretrained model: 'https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/bninception.py'
    """
    if pretrained:
        if os.path.isfile('/content/tinhtn/MyDrive/pretrain/bn_inception-52deb4733.pth'):
            pretrained_dict = torch.load('/content/tinhtn/MyDrive/pretrain/bn_inception-52deb4733.pth')
        else:
            pretrained_dict = load_state_dict_from_url('http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth')
        model_dict = model.state_dict()
        new_dict = {}
        for k,_ in model_dict.items():
            raw_name = k.replace('main_branch.', '')
            if raw_name in pretrained_dict:
                new_dict[k] = pretrained_dict[raw_name]
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    return model


class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_rate, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)

class SpatialTransformBlock(nn.Module):
    def __init__(self, num_classes, pooling_size, channels):
        super(SpatialTransformBlock, self).__init__()
        self.num_classes = num_classes
        self.spatial = pooling_size

        self.global_pool = nn.AvgPool2d((pooling_size, pooling_size//2), stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        self.gap_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.att_list = nn.ModuleList()
        self.stn_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.gap_list.append(nn.AvgPool2d((pooling_size, pooling_size//2), stride=1, padding=0, ceil_mode=True, count_include_pad=True))
            self.fc_list.append(nn.Linear(channels, 1))
            self.att_list.append(ChannelAttn(channels))
            self.stn_list.append(nn.Linear(channels, 4))

    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        return x.cuda()

    def transform_theta(self, theta_i, region_idx):
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,0,0] = torch.sigmoid(theta_i[:,0])
        theta[:,1,1] = torch.sigmoid(theta_i[:,1])
        theta[:,0,2] = torch.tanh(theta_i[:,2])
        theta[:,1,2] = torch.tanh(theta_i[:,3])
        theta = theta.cuda()
        return theta

    def forward(self, features):
        pred_list = []
        bs = features.size(0)
        for i in range(self.num_classes):
            stn_feature = features * self.att_list[i](features) + features
            theta_i = self.stn_list[i](F.avg_pool2d(stn_feature, stn_feature.size()[2:]).view(bs,-1)).view(-1,4)
            theta_i = self.transform_theta(theta_i, i)

            sub_feature = self.stn(stn_feature, theta_i)
            pred = self.gap_list[i](sub_feature).view(bs,-1)
            pred = self.fc_list[i](pred)
            pred_list.append(pred)
        pred = torch.cat(pred_list, 1)
        return pred

class InceptionNet(nn.Module):
    def __init__(self, num_classes=51):
        super(InceptionNet, self).__init__()
        self.num_classes = num_classes
        self.main_branch = BNInception()
        self.global_pool = nn.AvgPool2d((8,4), stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.finalfc = nn.Linear(1024, num_classes)

        self.st_3b = SpatialTransformBlock(num_classes, 32, 256*3)
        self.st_4d = SpatialTransformBlock(num_classes, 16, 256*2)
        self.st_5b = SpatialTransformBlock(num_classes, 8, 256)

        # Lateral layers
        self.latlayer_3b = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_4d = nn.Conv2d(608, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_5b = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, input):
        bs = input.size(0)
        feat_3b, feat_4d, feat_5b = self.main_branch(input)
        main_feat = self.global_pool(feat_5b).view(bs,-1)
        main_pred = self.finalfc(main_feat)

        fusion_5b = self.latlayer_5b(feat_5b)
        fusion_4d = self._upsample_add(fusion_5b, self.latlayer_4d(feat_4d))
        fusion_3b = self._upsample_add(fusion_4d, self.latlayer_3b(feat_3b))

        pred_3b = self.st_3b(fusion_3b)
        pred_4d = self.st_4d(fusion_4d)
        pred_5b = self.st_5b(fusion_5b)

        return pred_3b, pred_4d, pred_5b, main_pred

    def name(self) -> str:
        return 'inception_iccv'

class ResnetAlm(nn.Module):
    def __init__(self, num_classes):
        super(ResnetAlm, self).__init__()
        self.num_classes = num_classes
        self.main_branch = resnet50()
        self.global_pool = nn.AvgPool2d((8,4), stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.latlayer = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.finalfc = nn.Linear(2048, num_classes)
        self.st = SpatialTransformBlock(num_classes, 8, 1024)

    def forward(self, input):
        bs = input.size(0)
        feat = self.main_branch(input)
        main_feat = self.global_pool(feat).view(bs,-1)
        main_feat = self.finalfc(main_feat)
        fusion = self.latlayer(feat)
        pred = self.st(fusion)
        return pred, main_feat

    def name(self):
        return "inception_iccv"