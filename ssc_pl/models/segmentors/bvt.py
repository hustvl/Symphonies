import torch
import torch.nn as nn
import math

from .monoscene import MonoScene
from ..decoders import BilateralVoxelTransformer


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, 0::2].sin(), sin_x[:, 1::2].cos()), dim=1).flatten(1)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=1)
    return pos_res


class VoxelEmbed(nn.Module):

    def __init__(self, channels, scene_size, project_scale) -> None:
        super().__init__()
        scene_size = [s // project_scale for s in scene_size]
        coords = torch.stack(
            torch.meshgrid(*[torch.linspace(-1.0, 1.0, s) for s in scene_size]),
            dim=-1).view(-1, 3)
        coords = get_sine_pos_embed(coords, num_pos_feats=channels)
        self.register_buffer('coords', coords)

        self.voxel_embed = nn.Parameter(torch.zeros(1, channels, *scene_size))
        self.coord_mlp = nn.Linear(channels * 3, channels)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

        nn.init.trunc_normal_(self.voxel_embed, std=0.02)
        nn.init.trunc_normal_(self.coord_mlp.weight, std=0.02)
        nn.init.constant_(self.coord_mlp.bias, 0)

    def forward(self, x):
        coord_embed = self.coord_mlp(self.coords)
        coord_embed = coord_embed.transpose(0, 1).reshape(1, *x.shape[1:])
        return self.out_proj(coord_embed + x + self.voxel_embed)


class BVT(MonoScene):

    def __init__(self,
                 encoder,
                 channels,
                 scene_size,
                 view_scales,
                 volume_scale,
                 num_classes,
                 num_layers=3,
                 class_weights=None,
                 criterions=None,
                 **kwargs):
        super().__init__(
            encoder,
            channels,
            scene_size,
            view_scales,
            volume_scale,
            num_classes,
            class_weights=class_weights,
            criterions=criterions,
            **kwargs)
        self.voxel_embed = VoxelEmbed(channels, scene_size, volume_scale)
        self.decoder = BilateralVoxelTransformer(
            channels, num_classes, num_layers, scene_size, project_scale=volume_scale)

    def forward(self, inputs):
        x2ds = self.encoder(inputs['img'])

        projected_pix = inputs['projected_pix_{}'.format(self.volume_scale)]
        fov_mask = inputs['fov_mask_{}'.format(self.volume_scale)]
        x3ds = []
        for i, scale_2d in enumerate(self.view_scales):
            x3d = self.projects[i](x2ds['1_' + str(scale_2d)],
                                   torch.div(projected_pix, scale_2d, rounding_mode='trunc'),
                                   fov_mask)
            x3ds.append(x3d)
        x3d = torch.stack(x3ds).sum(dim=0)

        x3d = self.voxel_embed(x3d)
        outs = self.decoder(x3d, x3d, fov_mask)
        return outs
