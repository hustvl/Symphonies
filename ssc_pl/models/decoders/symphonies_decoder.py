import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from ..layers import TransformerLayer, DeformableTransformerLayer, nlc_to_nchw, nchw_to_nlc, Upsample
from ..projections.cvt import generate_grid
from .getr_decoder import index_fov_back_to_voxels


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


class SymphoniesLayer(nn.Module):

    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.query_self_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels=1, num_points=num_points, num_dims=3)
        self.query_scene_cross_deform_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels=1, num_points=num_points, num_dims=3)
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points, num_dims=2)

    def forward(self, scene_embed, inst_queries, feats, ref_2d, ref_3d, ref_vol, fov_mask):
        inst_queries = self.query_self_attn(inst_queries)
        inst_queries = self.scene_query_cross_attn(scene_embed, inst_queries, inst_queries)
        scene_embed = self.scene_self_deform_attn(
            scene_embed, scene_embed, reference_pts=ref_vol, spatial_shapes=volume_shape)
        scene_embed = self.query_scene_cross_deform_attn(
            inst_queries, scene_embed, reference_pts=ref_3d, spatial_shapes=volume_shape)
        scene_embed = self.query_image_cross_defrom_attn(
            inst_queries, feats, reference_pts=ref_2d, spatial_shapes=view_shapes)
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
                 offset=0.5):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape
        self.voxel_size = voxel_size
        self.offset = offset

        # TODO: the depth pred shape is not aligned with the original image,
        # find the reason and fix by interpolate/crop in the Dataset.getitem()
        image_grid = generate_grid((370, 1226), image_shape)  # 2(wh), h, w
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)
        self.register_buffer('image_grid', image_grid)

    def forward(self, scene_embed, feats, depth, K, E, voxel_origin, ref_pix):
        depth = depth.unsqueeze(1)
        p_x = torch.cat([self.image_grid * depth, depth], dim=1)  # bs, 3, h, w
        p_c = K.inverse() @ p_x.flatten(2)
        p_w = E.inverse() @ F.pad(p_c, (0, 0, 0, 1), value=1)
        p_v = ((p_w[:, :-1].transpose(1, 2) - voxel_origin.unsqueeze(1)) / self.voxel_size -
               self.offset).long()

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
            ref_pts=ref_pix[:, pts_mask].unsqueeze(2),
            spatial_shapes=shapes,
            level_start_index=get_level_start_index(shapes))
        return nchw_to_nlc(
            index_fov_back_to_voxels(
                nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask))


class SymphoniesDecoder(nn.Module):

    def __init__(self, embed_dims, num_classes, num_layers, scene_shape, project_scale, image_shape,
                 voxel_size):
        super().__init__()
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList([SymphoniesLayer(embed_dims) for _ in range(num_layers)])

        scene_shape = [s // project_scale for s in scene_shape]
        self.scene_shape = scene_shape
        self.num_queries = reduce(lambda x, y: x * y, scene_shape)
        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)

        self.voxel_proposal = VoxelProposal(embed_dims, scene_shape, image_shape,
                                            voxel_size * project_scale)
        self.image_shape = image_shape

        assert project_scale in (1, 2)
        self.cls_head = nn.Sequential(
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))

    def forward(self, pred_insts, feats, depth, K, E, voxel_origin, projected_pix, fov_mask):
        inst_queries = pred_insts['queries']  # bs, n, c
        bs = inst_queries.shape[0]

        ref_2d = pred_insts['ref_2d']
        ref_3d = pred_insts['ref_3d']  # TODO
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_vox = generate_grid(self.scene_shape, (1, 1, 1), offset=0.5).to(ref_pix)

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        scene_embed = self.voxel_proposal(scene_embed, feats, depth, K, E, voxel_origin, ref_pix)

        outs = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, ref_2d, ref_3d,
                                              ref_vox, fov_mask)
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))
        return outs
