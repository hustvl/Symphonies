import torch
import torch.nn as nn

from .modules import ASPP, Process


class CPMegaVoxels(nn.Module):

    def __init__(self, channels, size, num_relations=4):
        super().__init__()
        self.size = size
        self.num_relations = num_relations
        self.context_channels = channels * 2
        self.flatten_size = size[0] * size[1] * size[2]

        s = 2
        self.flatten_context_size = (size[0] // s) * (size[1] // s) * (size[2] // s)
        padding = ((size[0] + 1) % s, (size[1] + 1) % s, (size[2] + 1) % s)

        self.mega_context = nn.Conv3d(
            channels,
            self.context_channels,
            kernel_size=3,
            stride=s,
            padding=padding,
        )
        self.context_prior_logits = nn.ModuleList([
            nn.Conv3d(channels, self.flatten_context_size, kernel_size=1)
            for _ in range(num_relations)
        ])
        self.aspp = ASPP(channels, (1, 2, 3))

        self.resize = nn.Sequential(
            nn.Conv3d(
                self.context_channels * self.num_relations + channels,
                channels,
                kernel_size=1,
                bias=False),
            Process(channels, dilations=(1, ), norm_layer=nn.BatchNorm3d),
        )

    def forward(self, x):
        outs = {}
        x_agg = self.aspp(x)
        mega_context = self.mega_context(x_agg).flatten(2).transpose(1, 2)

        context_prior_logits = []
        context_rels = []
        for rel in range(self.num_relations):
            # Compute the relation matrices
            x_context_prior_logit = self.context_prior_logits[rel](x_agg).flatten(2)
            context_prior_logits.append(x_context_prior_logit)
            x_context_prior = torch.sigmoid(x_context_prior_logit.transpose(1, 2))

            # Multiply the relation matrices with the mega context to gather context features
            context_rel = torch.bmm(x_context_prior, mega_context)  # bs, N, f
            context_rels.append(context_rel)

        context_rels = torch.cat(context_rels, dim=2).transpose(1, 2)
        context_rels = context_rels.reshape(*context_rels.shape[:2], *self.size)

        p_logits = torch.stack(context_prior_logits, dim=1)
        x3d = self.resize(torch.cat([x, context_rels], dim=1))
        return x3d, p_logits
