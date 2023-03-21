import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddr import BottleneckDDR3D


class ASPP(nn.Module):

    def __init__(self, channels, dilations):
        super().__init__()
        self.blks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
            ) for d in dilations
        ])

    def forward(self, x):
        outs = [x]
        for blk in self.blks:
            outs.append(blk(x))
        outs = torch.stack(outs).sum(dim=0)
        return F.relu_(outs)


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Adapted from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, in_channels, channels, num_classes, dilations):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.aspp = ASPP(channels, dilations)
        self.conv_cls = nn.Conv3d(channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu_(self.conv0(x))
        x = self.aspp(x)
        x = self.conv_cls(x)
        return x


class Process(nn.Module):

    def __init__(self, channels, dilations=(1, 2, 3), norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.blks = nn.Sequential(*[
            BottleneckDDR3D(channels, channels // 4, dilation=(d, d, d), norm_layer=norm_layer)
            for d in dilations
        ])

    def forward(self, x):
        return self.blks(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.up_bn = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up_bn(x)


class Downsample(nn.Module):

    def __init__(self, channels, expansion=8, norm_layer=None):
        super().__init__()
        s = 2
        self.btnk = BottleneckDDR3D(
            channels,
            channels // 4,
            stride=s,
            expansion=expansion,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=s, stride=s),
                nn.Conv3d(
                    channels,
                    int(channels * expansion / 4),
                    kernel_size=1,
                    bias=False,
                ),
                norm_layer(int(channels * expansion / 4)),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.btnk(x)
