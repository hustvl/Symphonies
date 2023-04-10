import torch
import torch.nn as nn
from importlib import import_module

from mmengine.config import Config
from mmdet.registry import MODELS


class MMDetWrapper(nn.Module):

    def __init__(self, config_path, custom_imports, checkpoint_path=None, freeze=False):
        super().__init__()
        import_module(custom_imports)
        config = Config.fromfile(config_path)
        self.model = MODELS.build(config.model)
        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu'))
            )  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM
        self.hidden_dims = config.model.panoptic_head.decoder.hidden_dim
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # TODO: The following is only designed for the MaskDINO implementation.
        features = self.model.extract_feat(x)
        mask_features, _, multi_scale_features = self.model.panoptic_head.pixel_decoder.forward_features(
            features, masks=None)
        predictions = self.model.panoptic_head.predictor(
            multi_scale_features, mask_features, masks=None, return_queries=True)
        predictions = predictions[0]
        predictions['features'] = (features[0], multi_scale_features[0], multi_scale_features[1])
        predictions.pop('aux_outputs')
        predictions.pop('interm_outputs')
        return predictions  # dict_keys(['pred_logits', 'pred_masks', 'pred_boxes', 'queries', 'features'])
