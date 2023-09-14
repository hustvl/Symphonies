import torch
import torch.nn as nn
from einops import rearrange

from ..layers import Process, Upsample
from ..utils import cumprod, flatten_fov_from_voxels, index_fov_back_to_voxels, interpolate_flatten


class GeometryKernelAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5

        self.q_proj = nn.Sequential(
            norm_layer(embed_dims), nn.Linear(embed_dims, embed_dims, bias=qkv_bias))
        self.k_proj = nn.Sequential(
            norm_layer(embed_dims), nn.Linear(embed_dims, embed_dims, bias=qkv_bias))
        self.v_proj = nn.Sequential(
            norm_layer(embed_dims), nn.Linear(embed_dims, embed_dims, bias=qkv_bias))
        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key, value):
        """
        Args:
            query: [bs, n, c]
            key: [bs, n, k, c]
            value: [bs, n, k, c]
        """
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Group the head dim with batch dim
        query = rearrange(query, 'b n (h c) -> b h n c', h=self.num_heads, c=self.head_dims)
        key = rearrange(key, 'b n k (h c) -> b h n k c', h=self.num_heads, c=self.head_dims)
        value = rearrange(value, 'b n k (h c) -> b h n k c', h=self.num_heads, c=self.head_dims)

        attn = torch.einsum('b h n c, b h n k c -> b h n k', query, key)
        attn = torch.softmax(attn * self.scale, dim=-1)
        out = torch.einsum('b h n k, b h n k c -> b h n c', attn, value)
        out = rearrange(out, 'b h n c -> b n (h c)', h=self.num_heads, c=self.head_dims)
        out = self.out_proj(out)
        return out


class FactorizedAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Sequential(
            norm_layer(embed_dims), nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias))
        self.proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        x = self.scale * factor_att
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, C]
        x = self.proj(x)
        return x


class AxialAttention(nn.Module):

    def __init__(self, embed_dims, num_heads=8, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, batch_first=True)
            for _ in range(3)
        ])
        self.norm2 = norm_layer(embed_dims)

    def forward(self, query_embed, x3d):
        query_embed_ = self.norm1(query_embed)
        reduce_shapes = ((2, 3), (2, 4), (3, 4))
        for i, attn in enumerate(self.attns):
            kv = x3d.mean(reduce_shapes[i]).transpose(1, 2)
            kv = self.norm2(kv)
            query_embed = query_embed + attn(query_embed_, kv, kv)[0]
        return query_embed


class GeometryTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=4,
                 mlp_ratio=2,
                 qkv_bias=True,
                 kernel_size=3,
                 mlp_conv=False):
        super().__init__()
        self.cross_attn = GeometryKernelAttention(embed_dims, num_heads, qkv_bias=qkv_bias)
        self.self_attn = AxialAttention(embed_dims, num_heads, qkv_bias)

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dims), nn.Linear(embed_dims, embed_dims * mlp_ratio))
        self.ffn2 = nn.Sequential(nn.GELU(), nn.Linear(embed_dims * mlp_ratio, embed_dims))
        self.mlp_conv = nn.Conv3d(
            embed_dims * mlp_ratio,
            embed_dims * mlp_ratio,
            3,
            padding=1,
            groups=embed_dims * mlp_ratio) if mlp_conv else None
        if mlp_conv:
            self.pre_conv = nn.Conv3d(embed_dims, embed_dims *
                                      mlp_ratio, 1) if mlp_ratio != 1 else nn.Identity()

        x = torch.arange(kernel_size) - kernel_size // 2
        offsets = torch.stack(torch.meshgrid(x, x, indexing='xy')).permute(1, 2, 0).reshape(-1, 2)
        self.register_buffer('grid_offsets', offsets, persistent=False)  # k**2, 2

    def forward(self, query_embed, feats, scales, projected_pix, fov_mask, x3d=None):
        projected_pix = projected_pix[fov_mask].unsqueeze(0)  # n, 2 -> b(1), n, 2
        kernel_feats = []
        for scale, feat in zip(scales, feats):
            projected_pix_scale = torch.div(projected_pix, scale, rounding_mode='floor')
            projected_pix_scale = projected_pix_scale.unsqueeze(
                2) + self.grid_offsets  # bs, n, k, 2
            bs, c, h, w = feat.shape
            _, n, k, _ = projected_pix_scale.shape
            projected_pix_scale = projected_pix_scale.flatten(1, 2)

            indices = (
                projected_pix_scale[..., 1].clamp(0, h - 1) * w +
                projected_pix_scale[..., 0].clamp(0, w - 1))
            indices = indices.unsqueeze(1).expand(-1, c, -1).long()
            kernel_feats.append(torch.gather(feat.flatten(2), 2, indices).reshape(bs, c, n, k))
        feats = torch.cat(kernel_feats, dim=-1).permute(0, 2, 3, 1)  # bs, n, k*lvl, c

        query_embed = self.cross_attn(query_embed, feats, feats) + query_embed
        x3d = index_fov_back_to_voxels(x3d, query_embed, fov_mask)
        query_embed = self.self_attn(query_embed, x3d) + query_embed
        query_ = self.ffn1(query_embed)
        if self.mlp_conv is not None:
            x3d = index_fov_back_to_voxels(self.pre_conv(x3d), query_, fov_mask)
            x3d = self.mlp_conv(x3d)
            query_ = flatten_fov_from_voxels(x3d, fov_mask)
        query_embed = self.ffn2(query_) + query_embed
        return query_embed


class GeometryTransformerDecoder(nn.Module):

    def __init__(self, embed_dims, num_classes, num_layers, scene_shape, project_scale):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.scene_shape = [s // project_scale for s in scene_shape]
        self.project_scale = project_scale

        num_stages = len(num_layers)
        self.init_scene_shape = [s // 2**(num_stages - 1) for s in self.scene_shape]
        self.scene_embed = nn.Embedding(cumprod(self.init_scene_shape), embed_dims)

        self.stages = nn.ModuleList([
            nn.ModuleList([
                Process(embed_dims, dilations=(1, 2, 3)) if i != 0 else nn.Identity(),
                nn.ModuleList(
                    [GeometryTransformerDecoderLayer(embed_dims) for _ in range(num_layers[i])]),
                Upsample(embed_dims, embed_dims) if i != num_stages - 1 else nn.Identity(),
                nn.Sequential(*[
                    Upsample(embed_dims, embed_dims)
                    for _ in range(num_stages + project_scale - i - 2 -
                                   (1 if i != num_stages - 1 else 0))
                ])
            ]) for i in range(num_stages)
        ])

        self.cls_head = nn.Conv3d(embed_dims, num_classes, 1)

    def forward(self, x2ds, projected_pix, fov_mask):
        scales = [int(s[-1]) for s in x2ds.keys()]
        x2ds = [x for x in x2ds.values()]

        if self.training:
            sampling_queries = int(fov_mask.shape[-1] / 2)
            num_queries = fov_mask.sum().item()
            if num_queries > sampling_queries:
                idxes = torch.nonzero(fov_mask)[:, 1].tolist()
                from random import shuffle
                shuffle(idxes)
                idxes = idxes[:num_queries - sampling_queries]
                for i in idxes:
                    fov_mask[0, i] = False

        bs = x2ds[0].shape[0]
        assert bs == 1  # TODO
        x3d = self.scene_embed.weight.repeat(bs, 1, 1)
        x3d = x3d.transpose(1, 2).reshape(bs, -1, *self.init_scene_shape)

        outs = []
        for i, stage in enumerate(self.stages):
            projected_pix_i = interpolate_flatten(projected_pix, self.scene_shape, x3d.shape[2:])
            fov_mask_i = interpolate_flatten(fov_mask, self.scene_shape, x3d.shape[2:])
            conv, layers, upsample, aux_upsample = stage

            x3d = conv(x3d)
            query = flatten_fov_from_voxels(x3d, fov_mask_i.squeeze())
            for layer in layers:
                query = layer(query, x2ds, scales, projected_pix_i, fov_mask_i, x3d)
            x3d = index_fov_back_to_voxels(x3d, query, fov_mask_i)

            x3d = upsample(x3d)
            if self.training or i == len(self.stages) - 1:
                out = aux_upsample(x3d)
                outs.append(self.cls_head(out))
        return outs
