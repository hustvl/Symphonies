import sys

sys.path.append('.')  # run from project root

import os
import os.path as osp
import hydra
import numpy as np
import torch
import pickle

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from ssc_pl import build_data_loaders, build_from_configs, models


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, _ = build_from_configs(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[-1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)

    model = getattr(models,
                    cfg.model.type).load_from_checkpoint(cfg.model.ckpt_path, **cfg.model.cfgs,
                                                         **cfg.solver, **meta_info)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_inputs, targets in tqdm(data_loader):
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            outputs = model(batch_inputs)
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).astype(np.uint16)

            for i in range(cfg.data.loader.batch_size):
                output_dict = {'pred': preds[i]}
                if 'target' in targets:
                    output_dict['target'] = targets['target'][i].detach().cpu().numpy().astype(
                        np.uint16)

                if cfg.data.datasets.type == 'NYUv2':
                    save_dir = output_dir
                    file_path = osp.join(save_dir, batch_inputs['name'][i] + '.pkl')
                else:
                    save_dir = osp.join(output_dir, batch_inputs['sequence'][i])
                    file_path = osp.join(save_dir, batch_inputs['frame_id'][i] + '.pkl')

                keys = ('cam_pose', 'cam_K', 'voxel_origin', 'projected_pix_1', 'fov_mask_1')
                for key in keys:
                    output_dict[key] = batch_inputs[key][i].detach().cpu().numpy()

                keys_of_interest = []
                for key in keys_of_interest:
                    output_dict[key] = outputs[key].detach().cpu().numpy()

                os.makedirs(save_dir, exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(output_dict, f)
                    print('saved to', file_path)


if __name__ == '__main__':
    main()
