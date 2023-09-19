import hydra
from fvcore.nn import FlopCountAnalysis, flop_count_table
from omegaconf import DictConfig, open_dict

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)

    if cfg.trainer.devices != 1:
        with open_dict(cfg.trainer):
            cfg.trainer.devices = 1

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, meta_info=meta_info)
    flops = FlopCountAnalysis(model, next(iter(dls[1]))[0])
    print(flop_count_table(flops))


if __name__ == '__main__':
    main()
