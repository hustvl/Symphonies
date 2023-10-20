import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    trainer.fit(model, *dls[:2])


if __name__ == '__main__':
    main()
