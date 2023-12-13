import torch
import torch.nn as nn

from ..layers import DeformableTransformerLayer
from ..utils import (flatten_multi_scale_feats, get_level_start_index, index_fov_back_to_voxels,
                     nlc_to_nchw)


class VoxelProposalLayer(nn.Module):

    def __init__(self, embed_dims, scene_shape, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape

    def forward(self, scene_embed, feats, scene_pos=None, vol_pts=None, ref_pix=None):
        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        assert vol_pts.shape[0] == 1
        geom = vol_pts.squeeze()[keep.squeeze()]

        pts_mask = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)
        pts_mask[geom[:, 0], geom[:, 1], geom[:, 2]] = True
        pts_mask = pts_mask.flatten()

        feat_flatten, shapes = flatten_multi_scale_feats(feats)
        pts_embed = self.attn(
            scene_embed[:, pts_mask],
            feat_flatten,
            query_pos=scene_pos[:, pts_mask] if scene_pos is not None else None,
            ref_pts=ref_pix[:, pts_mask].unsqueeze(2).expand(-1, -1, len(feats), -1),
            spatial_shapes=shapes,
            level_start_index=get_level_start_index(shapes))
        return index_fov_back_to_voxels(
            nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask)
