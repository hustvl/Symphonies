from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from . import datasets


def build_data_loaders(cfg: DictConfig):
    dls = []
    for split in cfg.datasets.splits:
        cfgs = {}
        if isinstance(cfg.datasets.splits, dict):
            cfgs = OmegaConf.merge(cfgs, cfg.datasets.splits[split])
        else:
            cfgs = OmegaConf.merge(cfgs, {'split': split})
        shared_cfgs = cfg.datasets.get('cfgs')
        if isinstance(shared_cfgs, DictConfig):
            cfgs = OmegaConf.merge(cfgs, shared_cfgs)

        dl = DataLoader(getattr(datasets, cfg.datasets.type)(**cfgs),
                        **cfg.loader,
                        shuffle=(split == 'train'))
        dls.append(dl)
    return dls, getattr(datasets, cfg.datasets.type).META_INFO
