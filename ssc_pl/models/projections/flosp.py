import torch
import torch.nn as nn
import torch.nn.functional as F


class FLoSP(nn.Module):

    def __init__(self, scene_size, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        bs, c, h, w = x2d.shape
        src = x2d.flatten(2)
        src = F.pad(src, (0, 1), value=0)  # bs, c, h*w+1

        img_indices = (
            projected_pix[..., 1].clamp(0, h - 1) * w + projected_pix[..., 0].clamp(0, w - 1))
        img_indices[~fov_mask] = h * w
        feats = []
        for src_, indices in zip(src, img_indices):
            indices = indices.expand(c, -1).long()
            feat = torch.gather(src_, 1, indices)  # c, h*w*d
            feats.append(feat)

        feats = torch.stack(feats)
        x3d = feats.reshape(bs, c, *[s // self.project_scale for s in self.scene_size])
        return x3d


class MultiScaleFLoSP(nn.Module):

    def __init__(self, scene_size, view_scales, volume_scale):
        super().__init__()
        self.view_scales = view_scales
        self.projects = nn.ModuleList([FLoSP(scene_size, volume_scale) for _ in view_scales])

    def forward(self, feats, projected_pix, fov_mask):
        x3ds = []
        for i, scale_2d in enumerate(self.view_scales):
            x3d = self.projects[i](feats[i],
                                   torch.div(projected_pix, scale_2d, rounding_mode='floor'),
                                   fov_mask)
            x3ds.append(x3d)
        return torch.stack(x3ds).sum(dim=0)
