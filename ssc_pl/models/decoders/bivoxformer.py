import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Process, Upsample, Downsample, SegmentationHead
from .getr_decoder import flatten_fov_from_voxels, index_fov_back_to_voxels, interpolate_flatten


class DynamicUpdateLayer(nn.Module):

    def __init__(self, feature, img_feature, scene_size, num_points=1, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.num_points = num_points
        self.offsets = nn.Conv3d(feature, num_points * 3 * 2, 3, padding=1)
        # self.alpha = nn.Conv3d(feature, num_points * 2, 3, padding=1)
        nn.init.constant_(self.offsets.bias, 0)
        nn.init.constant_(self.offsets.weight, 0)

        points = torch.stack(
            torch.meshgrid(
                *[torch.linspace(0, 1, s).type_as(self.offsets.weight) for s in scene_size]),
            dim=-1)
        self.register_buffer('points', points.reshape(1, *scene_size, 3))

        self.vox_proj = nn.Sequential(
            nn.Conv3d(feature, feature, 1),
            norm_layer(feature),
            nn.ReLU(inplace=True),
        )
        self.img_proj = nn.Sequential(
            nn.Conv3d(img_feature, feature, 1),
            norm_layer(feature),
            nn.ReLU(inplace=True),
        )
        self.vox_alpha = nn.Conv3d(feature * 2, 1, 3, padding=1)
        self.img_alpha = nn.Conv3d(feature * 2, 1, 3, padding=1)

    def forward(self, x3d, img3d):
        scene_size = x3d.shape[2:]
        x3d_offsets = self.offsets(x3d).permute(0, 2, 3, 4, 1).reshape(-1, *scene_size, 6)

        image_pts = self.points + x3d_offsets[..., :3] / torch.tensor(img3d.shape[2:]).to(x3d)
        image_pts = (2 * image_pts - 1).reshape(-1, *scene_size, 3)
        x3d_img = F.grid_sample(
            img3d, image_pts, padding_mode='border', mode='bilinear', align_corners=False)
        x3d_img = self.img_proj(x3d_img)

        voxel_pts = self.points + x3d_offsets[..., -3:] / torch.tensor(scene_size).to(x3d)
        voxel_pts = (2 * voxel_pts - 1).reshape(-1, *scene_size, 3)
        x3d_vox = F.grid_sample(
            x3d, voxel_pts, padding_mode='border', mode='bilinear', align_corners=False)
        x3d_vox = self.vox_proj(x3d_vox)

        img3d_proj_alpha = self.img_alpha(torch.cat([x3d, x3d_img], dim=1)).sigmoid()
        vox3d_proj_alpha = self.vox_alpha(torch.cat([x3d, x3d_vox], dim=1)).sigmoid()
        x3d_img = x3d_img * img3d_proj_alpha + x3d * (1 - img3d_proj_alpha)
        x3d_vox = x3d_vox * vox3d_proj_alpha + x3d * (1 - vox3d_proj_alpha)
        x3d = (x3d_img + x3d_vox) / 2.0
        return x3d


class FusionBlock(nn.Module):

    def __init__(self, high_dims, low_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(low_dim, high_dims, 1),
            nn.BatchNorm3d(high_dims),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(high_dims, low_dim, 1),
            nn.BatchNorm3d(low_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x3d, ctx):
        x3d = x3d + self.conv1(F.interpolate(ctx, x3d.shape[2:], mode='trilinear'))
        ctx = ctx + self.conv2(F.interpolate(x3d, ctx.shape[2:], mode='trilinear'))
        return x3d, ctx


class AxialFormerLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, batch_first=True)
            for _ in range(3)
        ])

        self.norm3 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )
        self.norm2 = norm_layer(embed_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.MultiheadAttention):
        #     nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
        #     nn.init.constant_(m.in_proj_bias, 0)

    def forward(self, x3d, fov_mask):
        query_embed = flatten_fov_from_voxels(x3d, fov_mask.squeeze())
        query_embed_ = self.norm1(query_embed)
        reduce_dims = ((2, 3), (2, 4), (3, 4))
        for i, attn in enumerate(self.attns):
            kv = x3d.mean(reduce_dims[i]).transpose(1, 2)
            kv = self.norm2(kv)
            query_embed = query_embed + attn(query_embed_, kv, kv)[0]

        query_embed = query_embed + self.ffn(self.norm3(query_embed))
        return index_fov_back_to_voxels(x3d, query_embed, fov_mask.squeeze())


class VoxFormerStage(nn.Module):

    def __init__(self,
                 scene_size,
                 high_dims,
                 in_dims,
                 out_dims,
                 num_layers,
                 num_heads=8,
                 qkv_bias=True,
                 fusion=True,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.scene_size = scene_size
        self.vox_conv = nn.Sequential(Process(high_dims, dilations=(1, ), norm_layer=norm_layer))
        self.ctx_conv = nn.Sequential(
            Process(in_dims, dilations=(1, 2, 3), norm_layer=norm_layer),
            Downsample(in_dims, expansion=int(out_dims / in_dims * 4), norm_layer=norm_layer))
        self.layers = nn.ModuleList([
            AxialFormerLayer(out_dims, num_heads=num_heads, qkv_bias=qkv_bias)
            for _ in range(num_layers)
        ])
        self.fusion = FusionBlock(high_dims, out_dims) if fusion else None

    def forward(self, x3d, ctx, fov_mask):
        x3d = self.vox_conv(x3d)
        ctx = self.ctx_conv(ctx)
        if ctx.shape[2:] != self.scene_size:
            fov_mask = interpolate_flatten(fov_mask, self.scene_size, ctx.shape[2:])
        for layer in self.layers:
            ctx = layer(ctx, fov_mask)
        if self.fusion:
            x3d, ctx = self.fusion(x3d, ctx)
        return x3d, ctx


class BilateralVoxelTransformer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 scene_size,
                 project_scale,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        scene_size = [s // project_scale for s in scene_size]
        layer_dims = ((embed_dims, 128, 2), (128, 256, 2), (256, 256, 2), (256, 256, 1))
        self.layers = nn.ModuleList([
            VoxFormerStage(
                scene_size,
                high_dims=embed_dims,
                in_dims=layer_dims[i][0],
                out_dims=layer_dims[i][1],
                num_layers=3,
                norm_layer=norm_layer,
            ) for i in range(num_layers)
        ])

        self.dynamic_updates = nn.ModuleList([
            DynamicUpdateLayer(embed_dims, embed_dims, scene_size, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.upsample = Upsample(embed_dims, embed_dims // project_scale,
                                 norm_layer) if project_scale != 1 else nn.Identity()
        self.ssc_head = SegmentationHead(embed_dims // project_scale, embed_dims // project_scale,
                                         num_classes, (1, 2, 3))

    def forward(self, x3d, img3d, fov_mask):
        ctx = x3d
        for idx, layer in enumerate(self.layers):
            x3d, ctx = layer(x3d, ctx, fov_mask)
            x3d = self.dynamic_updates[idx](x3d, img3d)
        x3d = self.upsample(x3d)
        x3d = self.ssc_head(x3d)
        return {'ssc_logits': x3d}
