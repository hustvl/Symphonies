from functools import reduce

import torch
import torch.nn.functional as F


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)


def flatten_fov_from_voxels(x3d, fov_mask):
    assert x3d.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask].transpose(1, 2)


def index_fov_back_to_voxels(x3d, fov, fov_mask):
    assert x3d.shape[0] == fov.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    fov_concat = torch.zeros_like(x3d).flatten(2)
    fov_concat[..., fov_mask] = fov.transpose(1, 2)
    return torch.where(fov_mask, fov_concat, x3d.flatten(2)).reshape(*x3d.shape)


def interpolate_flatten(x, src_shape, dst_shape, mode='nearest'):
    """Inputs & returns shape as [bs, n, (c)]
    """
    if len(x.shape) == 3:
        bs, n, c = x.shape
        x = x.transpose(1, 2)
    elif len(x.shape) == 2:
        bs, n, c = *x.shape, 1
    assert cumprod(src_shape) == n
    x = F.interpolate(
        x.reshape(bs, c, *src_shape).float(), dst_shape, mode=mode,
        align_corners=False).flatten(2).transpose(1, 2).to(x.dtype)
    if c == 1:
        x = x.squeeze(2)
    return x


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    assert L == cumprod(shape), 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def pix2vox(pix_coords, depth, K, E, voxel_origin, voxel_size, offset=0.5, downsample_z=1):
    p_x = torch.cat([pix_coords * depth, depth], dim=1)  # bs, 3, h, w
    p_c = K.inverse() @ p_x.flatten(2)
    p_w = E.inverse() @ F.pad(p_c, (0, 0, 0, 1), value=1)
    p_v = (p_w[:, :-1].transpose(1, 2) - voxel_origin.unsqueeze(1)) / voxel_size - offset
    if downsample_z != 1:
        p_v[..., -1] /= downsample_z
    return p_v.long()


def vox2pix(voxel_pts, K, E, voxel_origin, scene_shape, image_shape, voxel_size):
    p_v = voxel_pts.squeeze(2) * torch.tensor(scene_shape).to(voxel_pts) * voxel_size + voxel_origin
    p_c = E @ F.pad(p_v.transpose(1, 2), (0, 0, 0, 1), value=1)
    p_x = (K @ p_c[:, :-1]) / p_c[:, 2]
    p_x = p_x[:, :-1].transpose(1, 2) / (torch.tensor(image_shape[::-1]).to(p_x) - 1)
    return p_x.clamp(0, 1)
