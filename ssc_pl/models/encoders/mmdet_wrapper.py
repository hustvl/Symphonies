import torch
import torch.nn as nn
from copy import deepcopy
from importlib import import_module

from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.models.layers import inverse_sigmoid


class MMDetWrapper(nn.Module):

    def __init__(self,
                 config_path,
                 custom_imports,
                 checkpoint_path=None,
                 embed_dims=256,
                 filter_topk=False,
                 freeze=False):
        super().__init__()
        import_module(custom_imports)
        config = Config.fromfile(config_path)
        self.hidden_dims = config.model.panoptic_head.decoder.hidden_dim
        self.model = MODELS.build(config.model)

        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu'))
            )  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM

        self.filter_topk = filter_topk
        if filter_topk:
            self.class_embed = self.model.panoptic_head.predictor.class_embed
        self.bbox_embed = self.model.panoptic_head.predictor.bbox_embed[-1]
        self.pts_embed = deepcopy(self.bbox_embed)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        num_levels = 3
        if embed_dims != self.hidden_dims:
            self.out_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_dims, embed_dims, 1),
                    nn.BatchNorm2d(embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dims, embed_dims, 1),
                ) for _ in range(num_levels)
            ])

    def forward(self, x):
        # TODO: The following is only devised for the MaskDINO implementation.
        feats = self.model.extract_feat(x)
        mask_feat, _, multi_scale_feats = self.model.panoptic_head.pixel_decoder.forward_features(
            feats, masks=None)
        preds = self.model.panoptic_head.predictor(
            multi_scale_feats, mask_feat, masks=None, return_queries=True)
        feats = (feats[0], *multi_scale_feats[:2])
        if hasattr(self, 'out_projs'):
            feats = [proj(feat) for proj, feat in zip(self.out_projs, feats)]
        queries, refs, pred_masks = list(
            map(lambda k: preds[k], ('queries', 'references', 'pred_masks')))
        pred_masks = pred_masks >= 0

        if self.filter_topk:
            queries, keep = self.filter_topk_queries(queries)
            refs, pred_masks = list(
                map(lambda x: self._batch_indexing(x, keep), (refs, pred_masks)))

        return dict(
            queries=queries,
            feats=feats,
            pred_masks=pred_masks,
            pred_pts=self.pred_box(self.pts_embed, queries, refs)[..., :2],
            pred_boxes=self.pred_box(self.bbox_embed, queries, refs))

    def filter_topk_queries(self, queries):
        scores = self.class_embed(queries)
        indices = scores.max(-1)[0].topk(self.filter_topk, sorted=False)[1]
        return self._batch_indexing(queries, indices), indices

    def pred_box(self, bbox_embed, hs, reference):
        delta_unsig = bbox_embed(hs)
        outputs_unsig = delta_unsig + inverse_sigmoid(reference)
        return outputs_unsig.sigmoid()

    def _batch_indexing(self, x, indices):
        """
        Args:
            x: shape (B, N, ...)
            indices: shape (B, N')
        Returns:
            shape (B, N', ...)
        """
        return torch.stack([q[i] for q, i in zip(x, indices)])
