import torch.nn as nn


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class TransformerLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, batch_first=True)

        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self, q, k, v):
        q = q + self.attn(self.norm1(q), k, v)[0]
        q = q + self.ffn(self.norm2(q))
        return q
