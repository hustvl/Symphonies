import torch.nn as nn

from ... import build_from_configs
from .. import encoders
from ..decoders import GeometryTransformerDecoder
from ..losses import ce_ssc_loss, geo_scal_loss, sem_scal_loss


class GeometryTransformer(nn.Module):

    def __init__(
        self,
        encoder,
        channels,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers,
        class_weights=None,
        criterions=None,
    ):
        super().__init__()
        self.view_scales = view_scales
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(encoders, encoder, channels=channels, scales=view_scales)
        self.decoder = GeometryTransformerDecoder(channels, num_classes, num_layers, scene_size,
                                                  volume_scale)

    def forward(self, inputs):
        projected_pix = inputs[f'projected_pix_{self.volume_scale}']
        fov_mask = inputs[f'fov_mask_{self.volume_scale}']
        x2ds = self.encoder(inputs['img'])
        outs = self.decoder(x2ds, projected_pix, fov_mask)
        return {'ssc_logits': outs[-1], 'aux_outputs': outs}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                        'ssc_logits': pred
                    }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
