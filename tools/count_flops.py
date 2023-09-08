import hydra
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, meta_info=meta_info)
    flops = FlopCountAnalysis(model, next(iter(dls[1]))[0])
    print(f'[fvcore] FLOPs: {flops.total() / 1e9:.2f} G')  # it is actually MACs


if __name__ == '__main__':
    main()
