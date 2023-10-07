from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention

from ..utils import cumprod


class DeformableSqueezeAttention(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 squeeze_axes=[0],
                 **kwargs):
        """Squeeze the Z dimension into the X / Y dimension and conduct 2D deformable attention.
        """
        super().__init__()
        assert all([squeeze_axis in [0, 1] for squeeze_axis in squeeze_axes])
        self.num_squeezes = len(squeeze_axes)
        self.squeeze_axes = squeeze_axes
        if self.num_squeezes > 1:
            if 'num_levels' in kwargs:
                kwargs['num_levels'] *= self.num_squeezes
        self.attns = nn.ModuleList([
            MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points, **kwargs)
            for _ in squeeze_axes
        ])

    def forward(self,
                query: torch.Tensor,
                value: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        for squeeze_axis, attn in zip(self.squeeze_axes, self.attns):
            value_, reference_points_, spatial_shapes_, level_start_index_ = self.squeeze_value_by_axis(
                value, reference_points, spatial_shapes,
                reference_points.size(-1) - squeeze_axis - 1)
            query = attn(
                query,
                value=value_,
                query_pos=query_pos,
                reference_points=reference_points_,
                spatial_shapes=spatial_shapes_,
                level_start_index=level_start_index_,
                **kwargs)
        return query

    def squeeze_value_by_axis(self, value, reference_points, spatial_shapes, squeeze_axis):
        bs, _, embed_dims = value.shape

        spatial_shapes_ = spatial_shapes.clone()
        spatial_shapes_[:, reference_points.size(-1) - squeeze_axis - 1] *= spatial_shapes_[:, -1]
        spatial_shapes_ = spatial_shapes_[:, :-1]

        reference_points_ = reference_points.clone()
        reference_points_[..., squeeze_axis] += ((reference_points_[..., 0] - 0.5) /
                                                 spatial_shapes[:, squeeze_axis])
        reference_points_ = reference_points_[..., 1:]
        # assert (reference_points_[..., squeeze_axis].max() <
        #         1) and (reference_points_[..., squeeze_axis].min() > 0)

        level_start_index = torch.cat((spatial_shapes_.new_zeros(
            (1, )), spatial_shapes_.prod(1).cumsum(0)[:-1]))

        if squeeze_axis in (1, 2):
            value_ = torch.cat([
                value_l.reshape(bs, *shape, embed_dims).flatten(2) if squeeze_axis == 1 else
                value_l.reshape(bs, *shape, embed_dims).transpose(2, 3).flatten(1, 3)
                for value_l, shape in zip(
                    value.split([cumprod(shape)
                                 for shape in spatial_shapes], dim=1), spatial_shapes)
            ],
                               dim=1)
        else:
            raise NotImplementedError
        return value_, reference_points_, spatial_shapes_, level_start_index
