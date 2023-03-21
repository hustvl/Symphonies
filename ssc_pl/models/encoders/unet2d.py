# Adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleBN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=concat_with.shape[2:], mode='bilinear', align_corners=False)
        f = torch.cat([up_x, concat_with], dim=1)
        return self.cbn(f)


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, scales, deep_decoder=True):
        super().__init__()

        def log2(x):
            LOOK_UP = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}
            return LOOK_UP[x]

        self.conv = nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1, padding=1)
        self.out_scales = scales
        self.out_indices = [log2(x) for x in scales]  # index represents resolution of 1/2**i
        self.deep_decoder = deep_decoder

        channels = [in_channels[-1] // 2**(5 - i) for i in range(len(in_channels))]

        if self.deep_decoder:
            self.resizes = nn.ModuleList(
                [nn.Conv2d(channels[i], out_channels, 1) for i in self.out_indices])
            self.upsamples = nn.ModuleList([
                UpSampleBN(in_channels=channels[i + 1] + c_in, out_channels=channels[i])
                for i, c_in in enumerate(in_channels[:-1])
            ])
        else:
            self.resizes = nn.ModuleList(
                [nn.Conv2d(in_channels[i], out_channels * (2**i), 1) for i in self.out_indices])

    def forward(self, xs):
        xs[-1] = self.conv(xs[-1])
        if self.deep_decoder:
            for i, upsample in reversed(list(enumerate(self.upsamples))):
                xs[i] = upsample(xs[i + 1], xs[i])

        outs = {f'1_{2**i}': resize(xs[i]) for i, resize in zip(self.out_indices, self.resizes)}
        if not self.deep_decoder:
            outs['global'] = xs[-1].flatten(2).mean(2)
        return outs


class Encoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        outs = [x]
        for k, v in self.backbone._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    outs.append(vi(outs[-1]))
            else:
                outs.append(v(outs[-1]))
        outs = [outs[i] for i in (0, 4, 5, 6, 8, 15)]
        return outs


class UNet2D(nn.Module):

    def __init__(self, hub_cfg, in_channels, channels, scales, deep_decoder=True):
        super().__init__()
        backbone = torch.hub.load(**hub_cfg)
        backbone.global_pool = nn.Identity()
        backbone.classifier = nn.Identity()

        self.encoder = Encoder(backbone)
        self.decoder = Decoder(in_channels, channels, scales, deep_decoder)

    def forward(self, x):
        outs = self.encoder(x)
        outs = self.decoder(outs)
        return outs
