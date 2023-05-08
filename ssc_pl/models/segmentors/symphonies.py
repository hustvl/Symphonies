import torch.nn as nn

from .interface import PLModelInterface
from .. import encoders
from ..decoders import SymphoniesDecoder
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

        self.encoder = getattr(encoders, encoder.type)(embed_dims=embed_dims, **encoder.cfgs)
        self.decoder = SymphoniesDecoder(
            embed_dims,
            num_classes,
            num_layers=3,
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=(370, 1220),
            voxel_size=0.2,
            downsample_z=2)
        self.insts_ffn = nn.Sequential(
            nn.Linear(self.encoder.hidden_dims, embed_dims * 4), nn.GELU(),
            nn.Linear(embed_dims * 4, embed_dims))

    def forward(self, inputs):
        pred_insts = self.encoder(inputs['img'])
        pred_insts['queries'] = self.insts_ffn(pred_insts['queries'])
        feats = pred_insts.pop('feats')
        pred_masks = pred_insts.pop('pred_masks', None)

        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))

        outs = self.decoder(pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                            fov_mask)
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
