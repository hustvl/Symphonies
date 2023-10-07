import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention


class TransformerLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, batch_first=True)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None):
        if key is None and value is None:
            key = value = query
            key_pos = query_pos
        if key_pos is not None:
            key = key + key_pos
        if query_pos is not None:
            query = query + self.attn(self.norm1(query) + query_pos, key, value)[0]
        else:
            query = query + self.attn(self.norm1(query), key, value)[0]
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))
        return query


class DeformableTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = attn_layer(
            embed_dims, num_heads, num_levels, num_points, batch_first=True, **kwargs)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self,
                query,
                value=None,
                query_pos=None,
                ref_pts=None,
                spatial_shapes=None,
                level_start_index=None):
        query = query + self.attn(
            self.norm1(query),
            value=value,
            query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))
        return query
