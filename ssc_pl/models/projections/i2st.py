import torch.nn as nn


class I2ST(nn.Module):

    def __init__(self, channels, scene_size):
        super().__init__()
        ...
    
    def forward(self, x, fov_mask):
        ...
