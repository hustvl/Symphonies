from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from .. import build_from_configs
from . import datasets


def build_data_loaders(cfg: DictConfig):
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    split_cfgs = cfg.datasets.pop('splits')
    if isinstance(split_cfgs, ListConfig):
        split_cfgs = {split: {'split': split} for split in split_cfgs}

    return [
        DataLoader(
            build_from_configs(datasets, dict(**cfg.datasets, **cfgs)),
            **cfg.loader,
            shuffle=split == 'train') for split, cfgs in split_cfgs.items()
    ], getattr(datasets, cfg.datasets.type).META_INFO
