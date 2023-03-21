import torch
import torch.nn as nn

from .interface import PLModelInterface
from .. import encoders
from ..decoders import UNet3D
from ..projections import FLoSP
from ..losses import (ce_ssc_loss, sem_scal_loss, geo_scal_loss, context_relation_loss,
                      frustum_proportion_loss)


class MonoScene(PLModelInterface):

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
            **kwargs  # optimizer, scheduler, evaluator
    ):
        super().__init__(**kwargs)
        self.view_scales = view_scales
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = getattr(encoders, encoder.type)(
            **encoder.cfgs, channels=channels, scales=view_scales)
        self.decoder = UNet3D(channels, scene_size, num_classes, num_relations, volume_scale,
                              context_prior)
        self.projects = nn.ModuleList([FLoSP(scene_size, volume_scale) for _ in self.view_scales])

    def forward(self, inputs):
        img = inputs['img']
        x2ds = self.encoder(img)

        projected_pix = inputs['projected_pix_{}'.format(self.volume_scale)]
        fov_mask = inputs['fov_mask_{}'.format(self.volume_scale)]
        x3ds = []
        for i, scale_2d in enumerate(self.view_scales):
            x3d = self.projects[i](x2ds['1_' + str(scale_2d)],
                                   torch.div(projected_pix, scale_2d, rounding_mode='trunc'),
                                   fov_mask)
            x3ds.append(x3d)
        x3d = torch.stack(x3ds).sum(dim=0)

        outs = self.decoder(x3d)
        return outs

    def losses(self, pred, target):
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
