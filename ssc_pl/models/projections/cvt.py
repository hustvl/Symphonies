import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TransformerLayer
from ..utils import generate_grid, nchw_to_nlc


class ProjectionLayer(nn.Module):

    def __init__(self, embed_dims, image_shape, feat_scale):
        super().__init__()
        self.cross_attn = TransformerLayer(embed_dims, num_heads=4)
        self.image_embed = nn.Linear(3, embed_dims, bias=False)
        self.scene_embed = nn.Linear(3, embed_dims, bias=False)
        self.camera_embed = nn.Linear(4, embed_dims, bias=False)

        self.key_proj = nn.Sequential(
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.Conv2d(embed_dims, embed_dims, 1, bias=False),
        )
        self.val_proj = nn.Sequential(
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.Conv2d(embed_dims, embed_dims, 1, bias=False),
        )

        grid = generate_grid([s // feat_scale for s in image_shape], image_shape)  # 1, 2, h, w
        grid = torch.flip(grid, dims=[0])
        self.register_buffer('grid', grid)

    def forward(self, query_embed, scene_embed, feature, K, E, depth=None):
        b, _, h, w = feature.shape
        grid = self.grid.repeat(b, 1, 1, 1)
        if depth is None:
            grid = F.pad(grid, (0, 0, 0, 0, 0, 1), value=1)
        else:
            depth = F.interpolate(depth.unsqueeze(1).float(), (h, w), mode='bilinear')
            depth /= 8000
            grid = torch.cat([grid * depth, depth], dim=1)

        image_embed = K.inverse() @ grid.flatten(2)  # b, 3, h*w
        image_embed = E.inverse() @ F.pad(image_embed, (0, 0, 0, 1), value=1)
        image_embed = image_embed[:, :-1].transpose(1, 2)
        image_embed = self.image_embed(image_embed)

        if depth is None:
            camera_embed = self.camera_embed(E.inverse()[..., -1:].transpose(1, 2))
            image_embed -= camera_embed
            image_embed = image_embed / (image_embed.norm(dim=1, keepdim=True) + 1e-6)

            scene_embed = self.scene_embed(scene_embed)
            scene_embed -= camera_embed
            scene_embed = scene_embed / (scene_embed.norm(dim=1, keepdim=True) + 1e-6)
        else:
            scene_embed = self.image_embed(scene_embed)

        return self.cross_attn(
            q=query_embed + scene_embed,
            k=nchw_to_nlc(self.key_proj(feature)) + image_embed,
            v=nchw_to_nlc(self.val_proj(feature)),
        )


class CrossViewTransformer(nn.Module):

    def __init__(self, embed_dims, scales, image_shape, scene_shape, ori_scene_shape, voxel_size):
        super().__init__()
        self.scales = scales
        self.projs = nn.ModuleList(
            [ProjectionLayer(embed_dims, image_shape, scale) for scale in scales])

        self.embed_dims = embed_dims
        self.scene_shape = scene_shape
        self.ori_scene_shape = ori_scene_shape
        self.num_queries = scene_shape[0] * scene_shape[1] * scene_shape[2]
        self.query_embed = nn.Embedding(self.num_queries, embed_dims)

        grid = generate_grid(scene_shape, ori_scene_shape, offset=0.5)
        grid = (grid * voxel_size).flatten(1).transpose(0, 1)
        self.register_buffer('query_pos', grid)

    def forward(self, features, K, E, voxel_origin, fov_mask, depth=None):
        fov_mask = fov_mask.reshape(-1, *self.ori_scene_shape)
        fov_mask = F.interpolate(
            fov_mask.unsqueeze(1).float(), size=tuple(self.scene_shape), mode='nearest').bool()
        fov_mask = fov_mask.flatten(1)

        point_thres = 3200
        num_points = fov_mask.sum().item()
        if num_points > point_thres:
            idxes = torch.nonzero(fov_mask)[:, 1].tolist()
            from random import shuffle
            shuffle(idxes)
            idxes = idxes[:num_points - point_thres]
            for i in idxes:
                fov_mask[0, i] = False

        features = [features[f'1_{s}'] for s in self.scales]
        b = features[0].shape[0]
        query_embed = self.query_embed.weight.repeat(b, 1, 1)
        scene_embed = self.query_pos.repeat(b, 1, 1) + voxel_origin

        query_embed_src = query_embed
        query_embed = query_embed[fov_mask].unsqueeze(0)
        scene_embed = scene_embed[fov_mask].unsqueeze(0)

        for proj, feature in zip(self.projs, features):
            query_embed = proj(query_embed, scene_embed, feature, K, E, depth)
            # x3d = query_embed.reshape(b, *self.scene_shape, self.embed_dims).permute(0, 4, 1, 2, 3)
            # x3d = conv3d(x3d)
            # query_embed = x3d.flatten(2).transpose(1, 2)

        query_embed_src[fov_mask] = query_embed.squeeze(0)
        query_embed = query_embed_src

        x3d = query_embed.reshape(b, *self.scene_shape, self.embed_dims).permute(0, 4, 1, 2, 3)
        return x3d
