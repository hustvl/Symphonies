import lightning as L
import torch.nn as nn
import torch.optim as optim
from omegaconf import open_dict
from torch.cuda.amp import autocast

from .. import build_from_configs, evaluation, models


class LitModule(L.LightningModule):

    def __init__(self, *, model, optimizer, scheduler, criterion=None, evaluator=None, **kwargs):
        super().__init__()
        self.model = build_from_configs(models, model, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = build_from_configs(nn, criterion) if criterion else self.model.loss
        self.train_evaluator = build_from_configs(evaluation, evaluator)
        self.test_evaluator = build_from_configs(evaluation, evaluator)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, evaluator=None):
        x, y = batch
        pred = self(x)
        with autocast(enabled=False):
            loss = self.criterion(pred, y)
        if evaluator:
            evaluator.update(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_evaluator)
        if isinstance(loss, dict):
            loss['loss_total'] = sum(loss.values())
            self.log_dict({f'train/{k}': v for k, v in loss.items()})
        else:
            self.log('train/loss', loss)
        return sum(loss.values()) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test')

    def _shared_eval(self, batch, prefix):
        loss = self._step(batch, self.test_evaluator)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`

        if isinstance(loss, dict):
            loss['loss_total'] = sum(loss.values())
            self.log_dict({f'{prefix}/{k}': v for k, v in loss.items()}, sync_dist=True)
        else:
            self.log(f'{prefix}/loss', loss, sync_dist=True)

    def on_train_epoch_end(self):
        self._log_metrics(self.train_evaluator, 'train')

    def on_validation_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'val')

    def on_test_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'test')

    def _log_metrics(self, evaluator, prefix=None):
        metrics = evaluator.compute()
        iou_per_class = metrics.pop('iou_per_class')
        if prefix:
            metrics = {'/'.join((prefix, k)): v for k, v in metrics.items()}
        self.log_dict(metrics, sync_dist=True)

        if hasattr(self, 'class_names'):
            self.log_dict(
                {
                    f'{prefix}/iou_{c}': s.item()
                    for c, s in zip(self.class_names, iou_per_class)
                },
                sync_dist=True)
        evaluator.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer
        scheduler_cfg = self.scheduler
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg]

            for k, v in self.named_parameters():
                in_param_group = False
                for i, pg_cfg in enumerate(paramwise_cfg):
                    if 'name' in pg_cfg and pg_cfg.name in k:
                        pgs[i].append(v)
                        in_param_group = True
                    # USER: Customize more cfgs if needed
                if not in_param_group:
                    params.append(v)
        else:
            params = self.parameters()
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        if paramwise_cfg:
            for pg, pg_cfg in zip(pgs, paramwise_cfg):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
                # USER: Customize more cfgs if needed
                optimizer.add_param_group({'params': pg, **cfg})
        scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg.interval}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
