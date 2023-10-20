import os
import os.path as osp

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks

KITTI_LABEL_MAP = {
    0: 0,  # unlabeled
    1: 10,  # car
    2: 11,  # bicycle
    3: 15,  # motorcycle
    4: 18,  # truck
    5: 20,  # other-vehicle
    6: 30,  # person
    7: 31,  # bicyclist
    8: 32,  # motorcyclist
    9: 40,  # road
    10: 44,  # parking
    11: 48,  # sidewalk
    12: 49,  # other-ground
    13: 50,  # building
    14: 51,  # fence
    15: 70,  # vegetation
    16: 71,  # trunk
    17: 72,  # terrain
    18: 80,  # pole
    19: 81,  # traffic-sign
}


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[-1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No checkpoint path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)
    model.cuda()
    model.eval()

    assert cfg.data.datasets.type == 'SemanticKITTI'
    label_map = np.array([KITTI_LABEL_MAP[i] for i in range(len(KITTI_LABEL_MAP))], dtype=np.int32)

    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            outputs = model(batch_inputs)
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            for i in range(preds.shape[0]):
                pred = label_map[preds[i].reshape(-1)].astype(np.uint16)
                save_dir = osp.join(output_dir,
                                    f"test/sequences/{batch_inputs['sequence'][i]}/predictions")
                file_path = osp.join(save_dir, f"{batch_inputs['frame_id'][i]}.label")
                os.makedirs(save_dir, exist_ok=True)
                pred.tofile(file_path)
                print('saved to', file_path)


if __name__ == '__main__':
    main()
