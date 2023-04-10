import torch.nn as nn

from .interface import PLModelInterface
from .. import encoders
from ..decoders import SymphoniesDecoder
from ..projections import OccAwareFLoSP
from ..layers import Process
from ..losses import (ce_ssc_loss, sem_scal_loss, geo_scal_loss, frustum_proportion_loss)


class Symphonies(PLModelInterface):

    def __init__(
            self,
            encoder,
            embed_dims,
            scene_size,
            view_scales,
            volume_scale,
            num_classes,
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

        self.encoder = getattr(encoders, encoder.type)(**encoder.cfgs)
        in_channels = self.encoder.hidden_dims
        self.project = OccAwareFLoSP(in_channels, scene_size, view_scales, volume_scale)
        self.decoder = SymphoniesDecoder(
            embed_dims,
            num_classes,
            num_layers=3,
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=(93, 305),
            ori_image_shape=(370, 1220),
            voxel_size=0.2)
        self.insts_ffn = nn.Sequential(
            nn.Linear(in_channels, embed_dims * 4),
            nn.GELU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.conv3d = nn.Sequential(Process(in_channels), nn.Conv3d(in_channels, embed_dims, 1))

    def forward(self, inputs):
        pred_insts = self.encoder(inputs['img'])
        pred_insts['queries'] = self.insts_ffn(pred_insts['queries'])
        feats = pred_insts.pop('features')

        depth, K, E, voxel_origin, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'fov_mask_{self.volume_scale}')))
        projected_pix, pix_z = list(
            map(lambda k: inputs[k],
                (f'projected_pix_{self.volume_scale}', f'pix_z_{self.volume_scale}')))
        # TODO: the depth pred shape is not aligned with the original image, find the reason and fix by interpolate/crop

        x3d, fov_mask = self.project(feats, depth, projected_pix, pix_z, fov_mask)
        x3d = self.conv3d(x3d)
        outs = self.decoder(pred_insts, x3d, depth, K, E, voxel_origin, fov_mask)
        return {'ssc_logits': outs[-1], 'aux_outputs': outs}

    def losses(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
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
