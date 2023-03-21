import os
import lightning.pytorch as pl
from omegaconf import DictConfig, open_dict

from .tabular_logger import TabularLogger


def build_from_configs(cfg: DictConfig):
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = None
    with open_dict(cfg):
        cfg.trainer.enable_progress_bar = False

    if cfg.get('dataset'):
        cfg.data.datasets.type = cfg.dataset
    if cfg.get('data_root'):
        cfg.data.datasets.cfgs.root = cfg.data_root
    if cfg.get('preprocess_root'):
        cfg.data.datasets.cfgs.preprocess_root = cfg.preprocess_root

    output_dir = 'outputs'
    callbacks = {
        'logger': [
            # pl.loggers.TensorBoardLogger(save_dir=output_dir, name=None),
            TabularLogger(save_dir=output_dir, name=None)
        ],
        'callbacks': [
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, cfg.save_dir)
                if cfg.get('save_dir') else output_dir,
                filename='{epoch}-{val_mIoU:.4f}',
                monitor='val_mIoU',
                save_last=True,
                mode='max',
            ),
            # pl.callbacks.ModelSummary(max_depth=-1)
        ]
    }
    return cfg, callbacks
