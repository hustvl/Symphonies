import torch.nn as nn

from .interface import PLModelInterface
from .. import encoders
from ..decoders import UNet3D
from ..projections import I2ST
from ..losses import (ce_ssc_loss, sem_scal_loss, geo_scal_loss, frustum_proportion_loss)


class Symphonies(PLModelInterface):

    def __init__(
            self,
            encoder,
            channels,
            embed_dims,
            scene_size,
            volume_scale,
            num_classes,
            class_weights=None,
            criterions=None,
            **kwargs  # optimizer, scheduler, evaluator
    ):
        super().__init__(**kwargs)
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = getattr(encoders, encoder.type)(**encoder.cfgs)
        self.decoder = UNet3D(channels, scene_size, num_classes, project_scale=volume_scale,
                              context_prior=False)
        self.project = I2ST(embed_dims, channels, 3, scene_size, (93, 305), (370, 1220),
                            volume_scale, voxel_size=0.2)
        self.insts_fc = nn.Sequential(
            nn.Linear(self.encoder.hidden_dims, embed_dims * 4), nn.GELU(),
            nn.Linear(embed_dims * 4, embed_dims))

    def forward(self, inputs):
        insts = self.encoder(inputs['img'])
        insts['queries'] = self.insts_fc(insts['queries'])

        depth, K, E, voxel_origin, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'fov_mask_{self.volume_scale}')))
        x3d = self.project(insts, depth, K, E, voxel_origin, fov_mask)
        outs = self.decoder(x3d)
        return outs

    def losses(self, pred, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        target['class_weights'] = self.class_weights.type_as(pred['ssc_logits'])
        losses = {}
        for loss in self.criterions:
            losses['loss_' + loss] = loss_map[loss](pred, target)
        return losses
