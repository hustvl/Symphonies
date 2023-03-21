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

        img_indices = (projected_pix[..., 1].clamp(0, h - 1) * w +
                       projected_pix[..., 0].clamp(0, w - 1))
        img_indices[~fov_mask] = h * w
        feats = []
        for src_, indices in zip(src, img_indices):
            indices = indices.expand(c, -1).long()
            feat = torch.gather(src_, 1, indices)  # c, h*w*d
            feats.append(feat)

        feats = torch.stack(feats)
        x3d = feats.reshape(bs, c, *[s // self.project_scale for s in self.scene_size])
        return x3d
