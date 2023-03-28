import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TransformerLayer, Process
from .cvt import generate_grid


class I2ST(nn.Module):

    def __init__(self, embed_dims, out_channels, num_layers, scene_shape, image_shape,
                 ori_image_shape, project_scale, voxel_size):
        super().__init__()
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dims, num_heads=4) for _ in range(num_layers)])
        self.pos_embed = nn.Linear(3, embed_dims, bias=False)

        image_grid = generate_grid(image_shape, ori_image_shape)  # 2(wh), h, w
        image_grid = torch.flip(image_grid, dims=[0])
        self.register_buffer('image_grid', image_grid)

        scene_shape = [s // project_scale for s in scene_shape]
        self.scene_shape = scene_shape
        self.num_queries = scene_shape[0] * scene_shape[1] * scene_shape[2]
        self.query_embed = nn.Embedding(self.num_queries, embed_dims)

        offset = 0.5
        scene_grid = generate_grid(scene_shape, scene_shape)
        scene_grid = (scene_grid + offset) * voxel_size * project_scale
        scene_grid = scene_grid.flatten(1).transpose(0, 1)
        self.register_buffer('query_pos', scene_grid)

        self.out_conv = nn.Sequential(
            Process(embed_dims, dilations=(1, 2, 3)), nn.Conv3d(embed_dims, out_channels, 1))

        self.score_thr = 0.25
        self.mask_thr = 0.5
        self.iou_thr = 0.8

    def forward(self, pred_insts, depth, K, E, voxel_origin, fov_mask):
        inst_queries = pred_insts['queries']  # b, 300, 256
        pred_logits = pred_insts['pred_logits']  # b, 300, 133
        pred_masks = pred_insts['pred_masks']  # b, 300, 93, 305
        bs, _, c = inst_queries.shape

        query_embed = self.query_embed.weight.repeat(bs, 1, 1)
        query_pos = self.query_pos.repeat(bs, 1, 1) + voxel_origin

        assert bs == 1
        inst_masks, keep = self.panoptic_postprocess(pred_logits, pred_masks)
        if keep.sum().item() != 0:
            inst_queries = inst_queries[keep].unsqueeze(0)
            inst_masks = inst_masks.unsqueeze(0)
            inst_pos = self.pos_embed(self.gather_mask_pos(depth, inst_masks, K, E))

        query_embed_src = query_embed
        query_embed = query_embed[fov_mask].unsqueeze(0)
        query_pos = self.pos_embed(query_pos[fov_mask].unsqueeze(0))

        for layer in self.layers:
            if keep.sum().item() != 0:
                query_embed = layer(query_embed + query_pos, inst_queries + inst_pos, inst_queries)
            else:
                query_embed = layer(query_embed + query_pos, inst_queries, inst_queries)
        query_embed_src[fov_mask] = query_embed.squeeze(0)
        query_embed = query_embed_src

        x3d = query_embed.reshape(bs, *self.scene_shape, self.embed_dims).permute(0, 4, 1, 2, 3)

        x3d = self.out_conv(x3d)
        return x3d
        # TODO: Visuaize mask predictions and depth projection.

    def gather_mask_pos(self, depth, masks, K, E):
        bs, n, h, w = masks.shape
        depth = F.interpolate(depth.unsqueeze(1), (h, w))
        image_grid = self.image_grid.unsqueeze(0).repeat(bs, 1, 1, 1)
        image_grid = torch.cat([image_grid * depth, depth], dim=1)
        image_embed = K.inverse() @ image_grid.flatten(2)  # bs, 3, h*w
        image_embed = E.inverse() @ F.pad(image_embed, (0, 0, 0, 1), value=1)
        image_embed = image_embed[:, :-1].transpose(1, 2)  # bs, h*w, 3
        image_embed = image_embed.unsqueeze(1).repeat(1, n, 1, 1)

        for bi, masks_ in enumerate(masks):
            for ni, mask in enumerate(masks_):
                image_embed[bi, ni, ~mask.flatten()] = 0
        return image_embed.sum(dim=2)  # bs, n, 3

    def panoptic_postprocess(self, logits, pred_masks):
        num_classes = logits.shape[-1]
        scores, labels = F.softmax(logits.sigmoid() / 6e-2, dim=-1).max(-1)
        pred_masks = pred_masks.sigmoid()

        keep = labels.ne(num_classes) & (scores > self.score_thr)
        if keep.sum().item() == 0:
            return pred_masks, keep
        scores = scores[keep]
        pred_masks = pred_masks[keep]

        prob_masks = scores.reshape(-1, 1, 1) * pred_masks
        mask_ids = prob_masks.argmax(0)
        masks = mask_ids.unsqueeze(0).expand(len(prob_masks), -1, -1) == torch.arange(
            len(prob_masks)).to(prob_masks).reshape(-1, 1, 1)
        mask_area = masks.flatten(1).sum(dim=1)
        pred_masks = pred_masks >= self.mask_thr
        pred_mask_area = pred_masks.flatten(1).sum(dim=1)

        masks = masks & pred_masks
        keep_ = ((masks.flatten(1).sum(dim=1) > 0) & (mask_area / pred_mask_area >= self.iou_thr))
        keep_sub = keep[keep]
        keep_sub[~keep_] = False
        keep_ret = keep.clone()
        keep_ret[keep] = keep_sub
        return masks[keep_], keep_ret
