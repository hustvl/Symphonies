import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):

    def __init__(self, channels, norm_layer):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
        )

    def forward(self, x):
        out = x + self.convs(x)
        return F.relu_(out)


class BottleneckDDR3D(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 dilation=(1, 1, 1),
                 expansion=4,
                 downsample=None,
                 norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels)

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 1, kernel_size),
            stride=(1, 1, stride),
            padding=(0, 0, dilation[0]),
            dilation=(1, 1, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(channels)
        self.conv3 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, kernel_size, 1),
            stride=(1, stride, 1),
            padding=(0, dilation[1], 0),
            dilation=(1, dilation[1], 1),
            bias=False,
        )
        self.bn3 = norm_layer(channels)
        self.conv4 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(dilation[2], 0, 0),
            dilation=(dilation[2], 1, 1),
            bias=False,
        )
        self.bn4 = norm_layer(channels)

        self.conv5 = nn.Conv3d(channels, channels * expansion, kernel_size=1, bias=False)
        self.bn5 = norm_layer(channels * expansion)

        self.stride = stride
        self.downsample = downsample
        if stride != 1:
            self.downsample2 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )
            self.downsample3 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )
            self.downsample4 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )

    def forward(self, x):
        out1 = F.relu_(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = F.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out2 + out3
        out3_relu = F.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out2 + out3 + out4
        out4_relu = F.relu(out4)

        out5 = self.bn5(self.conv5(out4_relu))
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu_(x + out5)
        return out
