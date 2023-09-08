import torch.nn as nn

from ..layers import CPMegaVoxels, Downsample, Process, SegmentationHead, Upsample


class UNet3D(nn.Module):

    def __init__(
        self,
        channels,
        scene_size,
        num_classes,
        num_relations=4,
        project_scale=1,
        context_prior=True,
        norm_layer=nn.BatchNorm3d,
    ):
        super().__init__()
        feature_l1 = channels
        feature_l2 = feature_l1 * 2
        feature_l3 = feature_l2 * 2
        scene_size_l3 = [int(s / 4 / project_scale) for s in scene_size]

        self.process_l1 = nn.Sequential(
            Process(feature_l1, dilations=(1, 2, 3), norm_layer=norm_layer),
            Downsample(feature_l1, norm_layer=norm_layer),
        )
        self.process_l2 = nn.Sequential(
            Process(feature_l2, dilations=(1, 2, 3), norm_layer=norm_layer),
            Downsample(feature_l2, norm_layer=norm_layer),
        )
        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                feature_l3,
                scene_size_l3,
                num_relations=num_relations,
            )
        self.up_13_l2 = Upsample(feature_l3, feature_l2, norm_layer)
        self.up_12_l1 = Upsample(feature_l2, feature_l1, norm_layer)

        if project_scale != 1:  # 2 for KITTI
            self.up_l1_full = Upsample(channels, channels // 2, norm_layer)
            channels = channels // 2
        else:
            self.up_l1_full = nn.Identity()
        self.ssc_head = SegmentationHead(channels, channels, num_classes, (1, 2, 3))

    def forward(self, x):
        # Volume resolution is 1/2**l for KITTI and 1/2**(l+1) for NYU
        x3d_l1 = x
        x3d_l2 = self.process_l1(x3d_l1)
        x3d_l3 = self.process_l2(x3d_l2)

        outs = {}
        if self.context_prior:
            x3d_l3, p_logits = self.CP_mega_voxels(x3d_l3)
            outs['P_logits'] = p_logits

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_full = self.up_l1_full(x3d_up_l1)
        outs['ssc_logits'] = self.ssc_head(x3d_full)
        return outs
