import torch.optim as optim
import lightning.pytorch as pl
from torch.cuda.amp import autocast

from ... import evaluation


class PLModelInterface(pl.LightningModule):

    def __init__(self, optimizer, scheduler, evaluator, **kwargs):
        super().__init__()
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.train_evaluator = getattr(evaluation, evaluator.type)(**evaluator.cfgs)
        self.test_evaluator = getattr(evaluation, evaluator.type)(**evaluator.cfgs)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        ...  # define your model afterward

    def forward(self, x):
        ...

    def losses(self, pred, y):
        ...

    def _step(self, batch, evaluator=None):
        x, y = batch
        pred = self(x)
        with autocast(enabled=False):
            loss = self.losses(pred, y)
        if evaluator:
            evaluator.update(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_evaluator)
        self.log('train_loss', {'loss_total': sum(loss.values()), **loss})
        return sum(list(loss.values())) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test')

    def _shared_eval(self, batch, prefix):
        loss = self._step(batch, self.test_evaluator)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`
        self.log(f'{prefix}_loss', loss, sync_dist=True)

    def training_epoch_end(self, outputs):
        self._log_metrics(self.train_evaluator, 'train')

    def validation_epoch_end(self, outputs):
        self._log_metrics(self.test_evaluator, 'val')

    def _log_metrics(self, evaluator, prefix=None):
        metrics = evaluator.compute()
        iou_per_class = metrics.pop('iou_per_class')
        if prefix:
            metrics = {'_'.join((prefix, k)): v for k, v in metrics.items()}
        self.log_dict(metrics, sync_dist=True)

        if hasattr(self, 'class_names'):
            self.log(
                prefix + '_iou_per_cls',
                {c: s.item()
                 for c, s in zip(self.class_names, iou_per_class)},
                sync_dist=True)
        evaluator.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        scheduler_cfg = self.scheduler_cfg
        if 'paramwise_cfg' in optimizer_cfg:
            paramwise_cfg = optimizer_cfg.paramwise_cfg
            params = []
            pgs = [[] for _ in paramwise_cfg]

            for k, v in self.named_parameters():
                for i, pg_cfg in enumerate(paramwise_cfg):
                    if 'name' in pg_cfg and pg_cfg.name in k:
                        pgs[i].append(v)
                    # USER: Customize more cfgs if needed
                    else:
                        params.append(v)
        else:
            params = self.parameters()
        optimizer = getattr(optim, optimizer_cfg.type)(params, **optimizer_cfg.cfgs)
        if 'paramwise_cfg' in optimizer_cfg:
            for pg, pg_cfg in zip(pgs, paramwise_cfg):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg.cfgs.lr * pg_cfg.lr_mult
                # USER: Customize more cfgs if needed
                optimizer.add_param_group({'params': pg, **cfg})
        scheduler = getattr(optim.lr_scheduler, scheduler_cfg.type)(optimizer, **scheduler_cfg.cfgs)
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg.interval}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
