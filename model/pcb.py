import torch.nn as nn
from .bninception import BNInception
from .alm import SpatialTransformBlock
import torch.nn.functional as F
import torch

class PartBaseConvolution(nn.Module):
    def __init__(self, num_stripes=3, local_conv_out_channels=256, num_classes=26):
        super(PartBaseConvolution, self).__init__()

        self.base = BNInception()
        self.global_pool = nn.AvgPool2d((8,4), stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.num_stripes = num_stripes
        self.local_conv_list_5b = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list_5b.append(
                nn.Sequential(
                nn.Conv2d(1024, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.local_conv_list_4d = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list_4d.append(
                nn.Sequential(
                nn.Conv2d(608, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.local_conv_list_3b = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list_3b.append(
                nn.Sequential(
                nn.Conv2d(320, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.st_list_5b = nn.ModuleList()
        for _ in range(num_stripes):
            st = SpatialTransformBlock(num_classes, 4, local_conv_out_channels)
            self.st_list_5b.append(st)

        self.st_list_4d = nn.ModuleList()
        for _ in range(num_stripes):
            st = SpatialTransformBlock(num_classes, 8, local_conv_out_channels*2)
            self.st_list_4d.append(st)

        self.st_list_3b = nn.ModuleList()
        for _ in range(num_stripes):
            st = SpatialTransformBlock(num_classes, 16, local_conv_out_channels*3)
            self.st_list_3b.append(st)

        self.finalfc = nn.Linear(1024, num_classes)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, input):
        bs = input.size(0)
        feat_3b, feat_4d, feat_5b = self.base(input)
        stripe_h = int(feat_5b.size(2) / self.num_stripes)
        
        main_feat = self.global_pool(feat_5b).view(bs,-1)
        main_pred = self.finalfc(main_feat)
        logits_list = [main_pred]
        logits_list = [main_pred]

        local_feat_5b_1 = self.local_conv_list_5b[0](feat_5b[:, :, 0: 4, :])
        logits_list.append(self.st_list_5b[0](local_feat_5b_1))

        local_feat_5b_2 = self.local_conv_list_5b[1](feat_5b[:, :, 2: 6, :])
        logits_list.append(self.st_list_5b[1](local_feat_5b_2))

        local_feat_5b_3 = self.local_conv_list_5b[2](feat_5b[:, :, 4: 8, :])
        logits_list.append(self.st_list_5b[2](local_feat_5b_3))

        # local_feat_4d_1 = self.local_conv_list_4d[0](feat_4d[:, :, 0: 8, :])
        # local_feat_4d_1 = self._upsample_add(local_feat_5b_1, local_feat_4d_1)
        # logits_list.append(self.st_list_4d[0](local_feat_4d_1))

        # local_feat_4d_2 = self.local_conv_list_4d[1](feat_4d[:, :, 4: 12, :])
        # local_feat_4d_2 = self._upsample_add(local_feat_5b_2, local_feat_4d_2)
        # logits_list.append(self.st_list_4d[1](local_feat_4d_2))

        # local_feat_4d_3 = self.local_conv_list_4d[2](feat_4d[:, :, 8: 16, :])
        # local_feat_4d_3 = self._upsample_add(local_feat_5b_3, local_feat_4d_3)
        # logits_list.append(self.st_list_4d[2](local_feat_4d_3))

        # local_feat_3b_1 = self.local_conv_list_3b[0](feat_3b[:, :, 0: 16, :])
        # local_feat_3b_1 = self._upsample_add(local_feat_4d_1, local_feat_3b_1)
        # logits_list.append(self.st_list_3b[0](local_feat_3b_1))

        # local_feat_3b_2 = self.local_conv_list_3b[1](feat_3b[:, :, 8: 24, :])
        # local_feat_3b_2 = self._upsample_add(local_feat_4d_2, local_feat_3b_2)
        # logits_list.append(self.st_list_3b[1](local_feat_3b_2))

        # local_feat_3b_3 = self.local_conv_list_3b[2](feat_3b[:, :, 16: 32, :])
        # local_feat_3b_3 = self._upsample_add(local_feat_4d_3, local_feat_3b_3)
        # logits_list.append(self.st_list_3b[2](local_feat_3b_3))

        return logits_list

    def name(self) -> str:
        return 'inception_iccv'
