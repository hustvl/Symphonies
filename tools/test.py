import sys

sys.path.append('.')  # run from project root

import os
import os.path as osp
import hydra
import torch
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from ssc_pl import build_data_loaders, build_from_configs, models

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


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, _ = build_from_configs(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[-1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)

    if cfg.model.get('ckpt_path'):
        model = getattr(models,
                        cfg.model.type).load_from_checkpoint(cfg.model.ckpt_path, **cfg.model.cfgs,
                                                             **cfg.solver, **meta_info)
    else:
        import warnings
        warnings.warn('No checkpoint being loaded.')
        model = getattr(models, cfg.model.type)(**cfg.model.cfgs, **cfg.solver, **meta_info)
    model.cuda()
    model.eval()

    assert cfg.data.datasets.type == 'SemanticKITTI'
    label_map = np.array([KITTI_LABEL_MAP[i] for i in range(len(KITTI_LABEL_MAP))], dtype=np.uint16)

    with torch.no_grad():
        for batch_inputs, targets in tqdm(data_loader):
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            outputs = model(batch_inputs)
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).astype(np.uint16)

            for i in range(preds.shape[0]):
                pred = label_map[preds[i].reshape(-1)]
                save_dir = osp.join(output_dir,
                                    f"test/sequences/{batch_inputs['sequence'][i]}/predictions")
                file_path = osp.join(save_dir, f"{batch_inputs['frame_id'][i]}.label")
                os.makedirs(save_dir, exist_ok=True)
                with open(file_path, 'wb') as f:
                    pred.tofile(f)
                    print('saved to', file_path)


if __name__ == '__main__':
    main()
