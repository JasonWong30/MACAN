from .dat_arch import ResidualGroup
from einops.layers.torch import Rearrange
from einops import rearrange
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2], embed_dim=90, heads=[2, 2, 2, 2], img_size=256, split_size=[2, 4],
                 expansion_factor=4., qkv_bias=True, qk_scale=None, drop_rate=0., drop_path_rate=0.1,
                 attn_drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_chk=False, resi_connection='1conv'):
        super(Transformer, self).__init__()
        self.TAM = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )
        curr_dim = embed_dim
        self.norm = norm_layer(curr_dim)
        for i in range(1):
            layer = ResidualGroup(dim=embed_dim, num_heads=heads[i], reso=img_size, split_size=split_size,
                                  expansion_factor=expansion_factor, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])], act_layer=act_layer,
                                  norm_layer=norm_layer, depth=depth[i], use_chk=use_chk,
                                  resi_connection=resi_connection, rg_idx=i
                                  )
            self.TAM.append(layer)

    def forward(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]

        x = self.before_RG(x)
        for layer in self.TAM:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
