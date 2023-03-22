import torch.nn as nn
from importlib import import_module

from mmengine.config import Config
from mmdet.registry import MODELS


class MMDetWrapper(nn.Module):

    def __init__(self, config_path, custom_imports, checkpoint_path=None):
        super().__init__()
        import_module(custom_imports)
        config = Config.fromfile(config_path)
        self.model = MODELS.build(config.model)
        # TODO: Remove the prediction head as the unused parameters in DDP.

    def forward(self, x):
        # TODO: The following is only designed for the MaskDINO implementation.
        features = self.model.extract_feat(x)
        mask_features, _, multi_scale_features = self.model.panoptic_head.pixel_decoder.forward_features(
            features, masks=None)
        predictions = self.model.panoptic_head.predictor(
            multi_scale_features, mask_features, masks=None, return_queries=True)
        return predictions
