import os

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from ssc_pl import LitModule, build_data_loaders, generate_grid, pre_build_callbacks


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
        for batch_inputs, targets in data_loader:
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            # outputs = model(batch_inputs)

            # vol = outputs['ssc_logits']  # (B, C, X, Y, Z)
            vol = targets['target'].cuda()
            K = batch_inputs['cam_K']  # (B, 3, 3)
            E = batch_inputs['cam_pose']  # (B, 4, 4)
            vox_origin = batch_inputs['voxel_origin']  # (B, 3)
            vox_size = 0.2
            image_shape = batch_inputs['img'].shape[-2:]

            pix_coords = generate_grid(image_shape).to(vol)  # (2, H, W)
            pix_coords = torch.flip(pix_coords, dims=[0])
            depth = torch.arange(2, 50, step=1).to(pix_coords)  # (D,)
            p_x = F.pad(pix_coords, (0, 0, 0, 0, 0, 1), value=1)
            p_x = p_x.unsqueeze(-1).repeat(1, 1, 1, depth.size(0))  # (3, H, W, D)
            d_ = depth.reshape(1, 1, 1, -1)
            p_x = p_x * d_

            p_c = K.inverse() @ p_x.flatten(1)
            p_w = E.inverse() @ F.pad(p_c, (0, 0, 0, 1), value=1)
            p_v = (p_w[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(1)) / vox_size - 0.5
            p_v = p_v.reshape(1, *image_shape, depth.size(0), -1)  # (1, H, W, D, 3)
            p_v = p_v / (torch.tensor(vol.shape[-3:]) - 1).to(p_v)

            # vol = 1 - vol.softmax(dim=1)[:, 0].unsqueeze(1)  # prob of non-empty
            vol = ((vol.int() != 0) & (vol.int() != 255)).to(vol).unsqueeze(1)
            sigmas = F.grid_sample(vol, torch.flip(p_v, dims=[-1]) * 2 - 1, padding_mode='border')
            T = torch.exp(-torch.cumsum(sigmas * 1, dim=-1))
            alpha = 1 - torch.exp(-sigmas * 1)
            depth_map = torch.sum(T * alpha * d_.unsqueeze(0), dim=-1)
            draw_depth(depth_map, 'rendered_depth.png')
            draw_depth(batch_inputs['depth'], 'depth.png')
            import pdb; pdb.set_trace()


def draw_depth(depth_map, path):
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()
    plt.imsave(path, depth_map, cmap='jet')


if __name__ == '__main__':
    main()
