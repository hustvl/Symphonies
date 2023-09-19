import torch.nn as nn

from ... import build_from_configs
from .. import encoders
from ..decoders import UNet3D
from ..losses import (ce_ssc_loss, context_relation_loss,
                      frustum_proportion_loss, geo_scal_loss, sem_scal_loss)
from ..projections import MultiScaleFLoSP


class MonoScene(nn.Module):

    def __init__(
        self,
        encoder,
        channels,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_relations=4,
        context_prior=True,
        class_weights=None,
        criterions=None,
        **kwargs,
    ):
        super().__init__()
        self.view_scales = view_scales
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(encoders, encoder, channels=channels, scales=view_scales)        
        self.decoder = UNet3D(channels, scene_size, num_classes, num_relations, volume_scale,
                              context_prior)
        self.project = MultiScaleFLoSP(scene_size, view_scales, volume_scale)

    def forward(self, inputs):
        img = inputs['img']
        x2ds = self.encoder(img)

        projected_pix = inputs[f'projected_pix_{self.volume_scale}']
        fov_mask = inputs[f'fov_mask_{self.volume_scale}']
        x3d = self.project([x2ds[f'1_{s}'] for s in self.view_scales], projected_pix, fov_mask)
        outs = self.decoder(x3d)
        return outs

    def loss(self, pred, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'relation': context_relation_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        target['class_weights'] = self.class_weights.type_as(pred['ssc_logits'])
        losses = {}
        for loss in self.criterions:
            losses['loss_' + loss] = loss_map[loss](pred, target)
        return losses
