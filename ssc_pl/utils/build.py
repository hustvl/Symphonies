from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf, open_dict

from .tabular_logger import TabularLogger


def pre_build_callbacks(cfg: DictConfig):
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = 'auto'
    with open_dict(cfg):
        cfg.trainer.enable_progress_bar = False

    if cfg.get('dataset'):
        cfg.data.datasets.type = cfg.dataset
    if cfg.get('data_root'):
        cfg.data.datasets.cfgs.data_root = cfg.data_root
    if cfg.get('label_root'):
        cfg.data.datasets.cfgs.label_root = cfg.label_root
    if cfg.get('depth_root'):
        cfg.data.datasets.cfgs.depth_root = cfg.depth_root

    output_dir = 'outputs'

    logger = [loggers.TensorBoardLogger(save_dir=output_dir, name=None)]
    callback = [
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.ModelCheckpoint(
            dirpath=logger[0].log_dir,
            filename='e{epoch}_miou{val/mIoU:.4f}',
            monitor='val/mIoU',
            mode='max',
            auto_insert_metric_name=False),
        callbacks.RichModelSummary(max_depth=1)
    ]

    if cfg.trainer.get('enable_progress_bar', True):
        callback.append(callbacks.RichProgressBar())
    else:
        logger.append(TabularLogger(save_dir=output_dir, name=None, version=logger[0].version))

    return cfg, dict(logger=logger, callbacks=callback)


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
