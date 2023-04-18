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
        self.bbox_embed_2d = self.model.panoptic_head.predictor.bbox_embed[-1]
        self.bbox_embed_3d = deepcopy(self.bbox_embed_2d)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # TODO: The following is only designed for the MaskDINO implementation.
        feats = self.model.extract_feat(x)
        mask_feat, _, multi_scale_feats = self.model.panoptic_head.pixel_decoder.forward_features(
            feats, masks=None)
        preds = self.model.panoptic_head.predictor(
            multi_scale_feats, mask_feat, masks=None, return_queries=True)
        queries, refs = preds['queries'], preds['references']

        if self.filter_topk:
            queries, keep = self.filter_topk_queries(queries)
            refs = self._batch_indexing(refs, keep)
        return dict(
            queries=queries,
            feats=(feats[0], *multi_scale_feats[:1]),
            ref_2d=self.pred_box(self.bbox_embed_2d, queries, refs)[:2],
            ref_3d=self.pred_box(self.bbox_embed_3d, queries, refs)[:2])

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
