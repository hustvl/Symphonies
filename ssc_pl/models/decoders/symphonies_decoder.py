import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import reduce
from itertools import product
from torch.cuda.amp import autocast

from ..layers import (TransformerLayer, DeformableTransformerLayer, nlc_to_nchw, nchw_to_nlc,
                      DeformableSqueezeAttention, Upsample, LearnableSqueezePositionalEncoding)
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

    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4, query_update=True):
        super().__init__()
        self.query_self_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)

        self.query_update = query_update
        if query_update:
            self.query_scene_cross_deform_attn = DeformableTransformerLayer(
                embed_dims,
                num_heads,
                num_levels=1,
                num_points=num_points * 2,
                attn_layer=DeformableSqueezeAttention)
            self.query_image_cross_defrom_attn = DeformableTransformerLayer(
                embed_dims, num_heads, num_levels, num_points)

    def forward(self,
                scene_embed,
                inst_queries,
                feats,
                scene_pos=None,
                inst_pos=None,
                ref_2d=None,
                ref_3d=None,
                ref_vox=None,
                fov_mask=None):
        inst_queries = self.query_self_attn(inst_queries, query_pos=inst_pos)

        scene_embed_fov = flatten_fov_from_voxels(scene_embed, fov_mask)
        scene_pos_fov = flatten_fov_from_voxels(scene_pos,
                                                fov_mask) if scene_pos is not None else None
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        scene_level_index = get_level_start_index(scene_shape)

        feats_flatten, feat_shapes = flatten_multi_scale_feats(feats)
        feats_level_index = get_level_start_index(feat_shapes)

        scene_embed_fov = self.scene_query_cross_attn(scene_embed_fov, inst_queries, inst_queries,
                                                      scene_pos_fov, inst_pos)
        scene_embed_fov = self.scene_self_deform_attn(
            scene_embed_fov,
            scene_embed_flatten,
            query_pos=scene_pos_fov,
            ref_pts=ref_vox[:, fov_mask.squeeze()],  # TODO: assert bs == 1
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)

        scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])

        if not self.query_update:
            return scene_embed, inst_queries

        inst_queries = self.query_scene_cross_deform_attn(
            inst_queries,
            scene_embed_flatten,
            query_pos=inst_pos,
            ref_pts=ref_3d,
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)
        inst_queries = self.query_image_cross_defrom_attn(
            inst_queries,
            feats_flatten,
            query_pos=inst_pos,
            ref_pts=ref_2d,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)
        return scene_embed, inst_queries


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

        self.voxel_proposal = VoxelProposalLayer(embed_dims, scene_shape)
        self.layers = nn.ModuleList([
            SymphoniesLayer(embed_dims, query_update=True if i != num_layers - 1 else False)
            for i in range(num_layers)
        ])

        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        self.scene_pos = LearnableSqueezePositionalEncoding((128, 128, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))

        image_grid = generate_grid(image_shape, image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
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

    @autocast(dtype=torch.float32)
    def forward(self, pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                fov_mask):
        inst_queries = pred_insts['queries']  # bs, n, c
        inst_pos = pred_insts.get('query_pos', None)
        bs = inst_queries.shape[0]

        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')
        vol_pts = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z)

        ref_2d = pred_insts['pred_pts'].unsqueeze(2).expand(-1, -1, len(feats), -1)
        ref_3d = self.generate_vol_ref_pts_from_masks(
            pred_insts['pred_boxes'], pred_masks,
            vol_pts).unsqueeze(2) if pred_masks else self.generate_vol_ref_pts_from_pts(
                pred_insts['pred_pts'], vol_pts).unsqueeze(2)
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_vox = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2)

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        scene_pos = self.scene_pos().repeat(bs, 1, 1)
        scene_embed = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix)
        scene_pos = nlc_to_nchw(scene_pos, self.scene_shape)

        outs = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, scene_pos, inst_pos,
                                              ref_2d, ref_3d, ref_vox, fov_mask)
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))
        return outs

    def generate_vol_ref_pts_from_masks(self, pred_boxes, pred_masks, vol_pts):
        pred_boxes *= torch.tensor((self.image_shape + self.image_shape)[::-1]).to(pred_boxes)
        pred_pts = pred_boxes[..., :2].int()
        cx, cy, w, h = pred_boxes.split((1, 1, 1, 1), dim=-1)
        pred_boxes = torch.cat([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)],
                               dim=-1).int()
        pred_boxes[0::2] = pred_boxes[0::2].clamp(0, self.image_shape[1] - 1)
        pred_boxes[1::2] = pred_boxes[1::2].clamp(1, self.image_shape[1] - 1)

        pred_masks = F.interpolate(
            pred_masks.float(), self.image_shape, mode='bilinear').to(pred_masks.dtype)
        bs, n = pred_masks.shape[:2]

        for b, i in product(range(bs), range(n)):
            if pred_masks[b, i].sum().item() != 0:
                continue
            boxes = pred_boxes[b, i]
            pred_masks[b, i, boxes[1]:boxes[3], boxes[0]:boxes[2]] = True
            if pred_masks[b, i].sum().item() != 0:
                continue
            pred_masks[b, i, pred_pts[b, i, 1], pred_pts[b, i, 0]] = True
        pred_masks = pred_masks.flatten(2).unsqueeze(-1).to(vol_pts)  # bs, n, hw, 1
        vol_pts = vol_pts.unsqueeze(1) * pred_masks  # bs, n, hw, 3
        vol_pts = vol_pts.sum(dim=2) / pred_masks.sum(dim=2) / torch.tensor(
            self.scene_shape).to(vol_pts)
        return vol_pts.clamp(0, 1)

    def generate_vol_ref_pts_from_pts(self, pred_pts, vol_pts):
        pred_pts *= torch.tensor(self.image_shape[::-1]).to(pred_pts)
        pred_pts = pred_pts.long()
        pred_pts = pred_pts[..., 1] * self.image_shape[1] + pred_pts[..., 0]
        assert pred_pts.size(0) == 1
        ref_pts = vol_pts[:, pred_pts.squeeze()]
        ref_pts = ref_pts / torch.tensor(self.scene_shape).to(pred_pts)
        return ref_pts.clamp(0, 1)
