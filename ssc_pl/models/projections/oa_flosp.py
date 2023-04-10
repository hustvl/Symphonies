import torch
import torch.nn as nn

from .flosp import FLoSP, MultiScaleFLoSP


class OccAwareFLoSP(nn.Module):

    def __init__(self,
                 embed_dims,
                 scene_size,
                 view_scales,
                 volume_scale,
                 depth_scale=1,
                 distance_thr=0.5):
        super().__init__()
        self.project = MultiScaleFLoSP(scene_size, view_scales, volume_scale)
        self.project_d = FLoSP(scene_size, volume_scale)
        self.occ_embed = nn.Embedding(2, embed_dims)  # Free / Occluded

        self.depth_scale = depth_scale
        self.distance_thr = distance_thr

    def forward(self, feats, depth, projected_pix, pix_z, fov_mask):
        x3d = self.project(feats, projected_pix, fov_mask)
        bs, c, *scene_shape = x3d.shape

        v_z = self.project_d(depth.unsqueeze(1), projected_pix, fov_mask).flatten(1)
        v_z = ((pix_z - v_z) / (v_z + 1e-4)).flatten()

        dn, df = self.distance_thr, 1. / self.distance_thr
        free_embed, occluded_embed = self.occ_embed.weight

        x3d = x3d.flatten(2).transpose(0, 1).reshape(c, -1)  # c, bs*x*y*z
        fov = fov_mask.flatten()
        x3d = torch.where(fov & (v_z >= dn) & (v_z <= 1),
                          x3d * v_z + free_embed.unsqueeze(1) * (1 - v_z).unsqueeze(0),
                          torch.where(
                              fov & (v_z > 1) & (v_z <= df),
                              x3d / v_z + occluded_embed.unsqueeze(1) * (1 - 1 / v_z).unsqueeze(0),
                              x3d))  # (＠_＠;)
        x3d = x3d.transpose(0, 1)
        x3d[fov & (v_z > df)] = occluded_embed
        x3d[fov & (v_z < dn)] = free_embed
        return x3d.reshape(bs, *scene_shape,
                           c).permute(0, 4, 1, 2,
                                      3), (fov & (v_z >= dn * 0.8)).reshape(*fov_mask.shape)
