import sys

sys.path.append('.')  # run from project root

import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from ssc_pl import build_data_loaders, build_from_configs, models


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = build_from_configs(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = getattr(models, cfg.model.type)(**cfg.model.cfgs, **cfg.solver, **meta_info)
    trainer = pl.Trainer(**cfg.trainer, **callbacks)
    trainer.fit(model, *dls[:2])  # resume training by `ckpt_path=`


if __name__ == '__main__':
    main()
