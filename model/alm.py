import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url
from model.bninception import BNInception
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

class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SpatialAttn(nn.Module):
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x

class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y

class HardAttn(nn.Module):
    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta

class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta

class VisualAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate, num_classes):
        super(VisualAttn, self).__init__()

        assert in_channels%reduction_rate == 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_rate, num_classes, kernel_size=1, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(num_classes, affine=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return self.conv2_bn(x)

class ConfidenceWeighting(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConfidenceWeighting, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
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
        # self.visual_att = nn.ModuleList()
        self.conf_weight = nn.ModuleList()
        self.stn_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.gap_list.append(nn.AvgPool2d((pooling_size, pooling_size//2), stride=1, padding=0, ceil_mode=True, count_include_pad=True))
            self.fc_list.append(nn.Linear(channels, 1))
            self.att_list.append(ChannelAttn(channels))
            # self.visual_att.append(VisualAttn(channels, 16, num_classes))
            self.conf_weight.append(ConfidenceWeighting(channels, num_classes))
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
            # stn_feature = self.visual_att[i](features)*self.conf_weight[i](features)
            theta_i = self.stn_list[i](F.avg_pool2d(stn_feature, stn_feature.size()[2:]).view(bs,-1)).view(-1,4)
            theta_i = self.transform_theta(theta_i, i)

            sub_feature = self.stn(stn_feature, theta_i)
            pred = self.gap_list[i](stn_feature).view(bs,-1)
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
