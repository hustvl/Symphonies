import os
import os.path as osp
import pickle

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No ckpt_path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            outputs = model(batch_inputs)
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).astype(np.uint16)

            for i in range(preds.shape[0]):
                output_dict = {'pred': preds[i]}
                if 'target' in targets:
                    output_dict['target'] = targets['target'][i].detach().cpu().numpy().astype(
                        np.uint16)

                if cfg.data.datasets.type == 'NYUv2':
                    file_path = osp.join(output_dir, batch_inputs['name'][i] + '.pkl')
                else:
                    file_path = osp.join(output_dir, batch_inputs['frame_id'][i] + '.pkl')

                keys = ('cam_pose', 'cam_K', 'voxel_origin', 'projected_pix_1', 'fov_mask_1')
                for key in keys:
                    output_dict[key] = batch_inputs[key][i].detach().cpu().numpy()

                keys_of_interest = []
                for key in keys_of_interest:
                    output_dict[key] = outputs[key].detach().cpu().numpy()

                os.makedirs(output_dir, exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(output_dict, f)
                    print('saved to', file_path)


if __name__ == '__main__':
    main()
