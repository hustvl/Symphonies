import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TransformerLayer, Process, Upsample
from ..projections.cvt import generate_grid
from .getr_decoder import index_fov_back_to_voxels


class SymphoniesDecoder(nn.Module):

    def __init__(self, embed_dims, num_classes, num_layers, scene_shape, project_scale, image_shape,
                 ori_image_shape, voxel_size):
        super().__init__()
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList([
            nn.ModuleList([TransformerLayer(embed_dims, num_heads=4) for _ in range(2)])
            for _ in range(num_layers)
        ])
        self.pos_embed = nn.Linear(3, embed_dims, bias=False)

        scene_shape = [s // project_scale for s in scene_shape]
        self.scene_shape = scene_shape
        self.num_queries = scene_shape[0] * scene_shape[1] * scene_shape[2]
        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)

        num_insts = 300
        self.inst_pos = nn.Embedding(num_insts, embed_dims)

        image_grid = generate_grid(image_shape, ori_image_shape)  # 2(wh), h, w
        image_grid = torch.flip(image_grid, dims=[0])
        self.register_buffer('image_grid', image_grid)

        offset = 0.5
        scene_grid = generate_grid(scene_shape, scene_shape)
        scene_grid = (scene_grid + offset) * voxel_size * project_scale
        scene_grid = scene_grid.flatten(1).transpose(0, 1)
        self.register_buffer('scene_pos', scene_grid)

        assert project_scale in (1, 2)  # TODO
        self.cls_head = nn.Sequential(
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))

        self.score_thr = 0.25
        self.mask_thr = 0.5
        self.iou_thr = 0.8

    def forward(self, pred_insts, x3d, depth, K, E, voxel_origin, fov_mask):
        inst_queries = pred_insts['queries']  # b, 300, 256
        pred_logits = pred_insts['pred_logits']  # b, 300, 133
        pred_masks = pred_insts['pred_masks']  # b, 300, 93, 305
        bs, num_insts, c = inst_queries.shape

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1) + x3d.flatten(2).transpose(1, 2)
        scene_pos = self.scene_pos.repeat(bs, 1, 1) + voxel_origin
        inst_pos = self.inst_pos.weight.repeat(bs, 1, 1)

        assert bs == 1
        inst_masks, keep = self.panoptic_postprocess(pred_logits, pred_masks)
        if (inst_masks.flatten(1).sum(dim=1) > 0).all().item():
            mask_pos = self.pos_embed(self.gather_mask_pos(depth, inst_masks.unsqueeze(0), K, E))
            inst_pos[keep] = inst_pos[keep] + mask_pos
        inst_queries = inst_queries[keep].unsqueeze(0)
        inst_pos = inst_pos[keep].unsqueeze(0)

        query_embed_src = scene_embed
        scene_embed = scene_embed[fov_mask].unsqueeze(0)
        scene_pos = self.pos_embed(scene_pos[fov_mask].unsqueeze(0))

        outs = []
        for i, layer in enumerate(self.layers):
            inst_queries = layer[0](inst_queries + inst_pos, scene_embed + scene_pos, scene_embed)
            scene_embed = layer[1](scene_embed + scene_pos, inst_queries + inst_pos, inst_queries)
            if self.training or i == len(self.layers) - 1:
                x3d_ = index_fov_back_to_voxels(
                    query_embed_src.transpose(1, 2).reshape(*x3d.shape), scene_embed,
                    fov_mask.squeeze())
                outs.append(self.cls_head(x3d_))
        return outs

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
        # if keep.sum().item() == 0:
        #     keep[:, scores.topk(30)[1].squeeze()] = True
        #     pred_masks = pred_masks[keep] >= self.mask_thr
        #     return pred_masks, keep

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
        keep_ = (masks.flatten(1).sum(dim=1) > 0) & (mask_area / pred_mask_area >= self.iou_thr)
        # if keep_.sum().item() == 0:
        #     return masks, keep

        keep_mask = keep.clone()
        keep_mask[keep] = keep[keep] & keep_
        return masks[keep_], keep_mask
