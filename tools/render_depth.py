import os

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, volume_rendering


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[1]

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

            vol = outputs['ssc_logits']  # (B, C, X, Y, Z)
            # vol = targets['target'].cuda()
            K = batch_inputs['cam_K']  # (B, 3, 3)
            E = batch_inputs['cam_pose']  # (B, 4, 4)
            vox_origin = batch_inputs['voxel_origin']  # (B, 3)
            vox_size = 0.2
            image_shape = batch_inputs['img'].shape[-2:]

            vol = 1 - vol.softmax(dim=1)[:, 0].unsqueeze(1)  # prob of non-empty
            # vol = ((vol.int() != 0) & (vol.int() != 255)).to(vol).unsqueeze(1)
            sigmas, d = volume_rendering(vol, K, E, vox_origin, vox_size, image_shape)
            T = torch.exp(-torch.cumsum(sigmas * 1, dim=-1))
            alpha = 1 - torch.exp(-sigmas * 1)
            depth_map = torch.sum(T * alpha * d.unsqueeze(0), dim=-1).squeeze(0)

            draw_depth(
                torch.cat([depth_map, batch_inputs['depth']], dim=1),
                f"1/2/depth_{batch_inputs['sequence'][0]}_{batch_inputs['frame_id'][0]}.png")


def draw_depth(depth_map, path):
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()
    plt.imsave(path, depth_map, cmap='jet')


if __name__ == '__main__':
    main()
