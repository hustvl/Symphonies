import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import reduce

from ..layers import (TransformerLayer, DeformableTransformerLayer, nlc_to_nchw, nchw_to_nlc,
                      MultiScaleDeformableAttention3D, DeformableSqueezeAttention, Upsample)
from ..projections.cvt import generate_grid
from .getr_decoder import flatten_fov_from_voxels, index_fov_back_to_voxels, interpolate_flatten


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def pix2vox(pix_coords, depth, K, E, voxel_origin, voxel_size, offset=0.5, downsample_z=1):
    p_x = torch.cat([pix_coords * depth, depth], dim=1)  # bs, 3, h, w
    p_c = K.inverse() @ p_x.flatten(2)
    p_w = E.inverse() @ F.pad(p_c, (0, 0, 0, 1), value=1)
    p_v = ((p_w[:, :-1].transpose(1, 2) - voxel_origin.unsqueeze(1)) / voxel_size - offset)
    if downsample_z != 1:
        p_v[..., -1] /= downsample_z
    return p_v.long()


class SymphoniesLayer(nn.Module):

    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.query_self_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)
        self.query_scene_cross_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points)

    def forward(self, scene_embed, inst_queries, feats, ref_2d, ref_3d, ref_vox, fov_mask):
        inst_queries = self.query_self_attn(inst_queries, inst_queries, inst_queries)

        scene_embed_fov = flatten_fov_from_voxels(scene_embed, fov_mask)
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        scene_level_index = get_level_start_index(scene_shape)

        feats_flatten, feat_shapes = flatten_multi_scale_feats(feats)
        feats_level_index = get_level_start_index(feat_shapes)

        scene_embed_fov = self.scene_query_cross_attn(scene_embed_fov, inst_queries, inst_queries)
        scene_embed_fov = self.scene_self_deform_attn(
            scene_embed_fov,
            scene_embed_flatten,
            ref_pts=ref_vox[:, fov_mask.squeeze()],  # TODO: assert bs == 1
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)

        scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])

        inst_queries = self.query_scene_cross_deform_attn(
            inst_queries,
            scene_embed_flatten,
            ref_pts=ref_3d,
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)
        inst_queries = self.query_image_cross_defrom_attn(
            inst_queries,
            feats_flatten,
            ref_pts=ref_2d,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)
        return scene_embed, inst_queries


class VoxelProposal(nn.Module):

    def __init__(self,
                 embed_dims,
                 scene_shape,
                 image_shape,
                 voxel_size,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 downsample_z=1):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape
        self.voxel_size = voxel_size
        self.downsample_z = downsample_z

        # TODO: the depth pred shape is not aligned with the original image,
        # find the reason and fix by interpolate/crop in the Dataset.getitem()
        image_grid = generate_grid(image_shape, image_shape)  # 2(wh), h, w
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)
        self.register_buffer('image_grid', image_grid)

    def forward(self, scene_embed, feats, depth, K, E, voxel_origin, ref_pix):
        p_v = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z)

        keep = ((p_v[..., 0] >= 0) & (p_v[..., 0] < self.scene_shape[0]) & (p_v[..., 1] >= 0) &
                (p_v[..., 1] < self.scene_shape[1]) & (p_v[..., 2] >= 0) &
                (p_v[..., 2] < self.scene_shape[2]))
        assert p_v.shape[0] == 1
        geom = p_v.squeeze()[keep.squeeze()]

        pts_mask = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)
        pts_mask[geom[:, 0], geom[:, 1], geom[:, 2]] = True
        pts_mask = pts_mask.flatten()

        feat_flatten, shapes = flatten_multi_scale_feats(feats)
        pts_embed = self.attn(
            scene_embed[:, pts_mask],
            feat_flatten,
            ref_pts=ref_pix[:, pts_mask].unsqueeze(2).expand(-1, -1, len(feats), -1),
            spatial_shapes=shapes,
            level_start_index=get_level_start_index(shapes))
        return index_fov_back_to_voxels(
            nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask)


class SymphoniesDecoder(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size=0.2,
                 downsample_z=1):
        super().__init__()
        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = reduce(lambda x, y: x * y, scene_shape)
        self.image_shape = image_shape
        self.voxel_size = voxel_size * project_scale
        self.downsample_z = downsample_z

        self.layers = nn.ModuleList([SymphoniesLayer(embed_dims) for _ in range(num_layers)])
        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        self.voxel_proposal = VoxelProposal(
            embed_dims, scene_shape, image_shape, self.voxel_size, downsample_z=downsample_z)
        voxel_grid = generate_grid(scene_shape, scene_shape, offset=0.5, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)

        assert project_scale in (1, 2)
        self.cls_head = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(),
            ) if downsample_z != 1 else nn.Identity(),
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))

    def forward(self, pred_insts, feats, depth, K, E, voxel_origin, projected_pix, fov_mask):
        inst_queries = pred_insts['queries']  # bs, n, c
        bs = inst_queries.shape[0]

        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')

        ref_2d = pred_insts['ref_2d'].unsqueeze(2).expand(-1, -1, len(feats), -1)
        ref_3d = self.generate_vol_ref_pts(pred_insts['ref_3d'], depth, K, E,
                                           voxel_origin).unsqueeze(2)
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_vox = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2)

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        scene_embed = self.voxel_proposal(scene_embed, feats, depth, K, E, voxel_origin, ref_pix)

        outs = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, ref_2d, ref_3d,
                                              ref_vox, fov_mask)
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))
        return outs

    def generate_vol_ref_pts(self, ref_pts, depth, K, E, voxel_origin):
        ref_pts *= torch.flip(torch.tensor(self.image_shape), dims=[0]).to(ref_pts)
        ref_pts = ref_pts.transpose(1, 2)  # bs, 2, n
        assert ref_pts.shape[0] == 1
        coords = ref_pts.squeeze().long()
        depth = depth[:, coords[1], coords[0]].unsqueeze(1)
        ref_pts = pix2vox(
            ref_pts,
            depth,
            K,
            E,
            voxel_origin,
            self.voxel_size,
            offset=0,
            downsample_z=self.downsample_z)
        return ref_pts.to(depth) / torch.tensor(self.scene_shape).to(depth)
