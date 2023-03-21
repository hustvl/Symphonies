import sys

sys.path.append('.')  # run from project root

import hydra
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis

from ssc_pl import build_data_loaders, build_from_configs, models


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    cfg, callbacks = build_from_configs(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = getattr(models, cfg.model.type)(**cfg.model.cfgs, **cfg.solver, **meta_info)
    flops = FlopCountAnalysis(model, next(iter(dls[1]))[0])
    print('[fvcore] FLOPs: {:.2f} G'.format(flops.total() / 1e9))  # it is actually MACs


if __name__ == '__main__':
    main()
