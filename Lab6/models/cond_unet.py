import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

    
    

class ConditionalUNet(nn.Module):
    def __init__(self, sample_size=64, label_dim=24, label_emb_size=128, time_dim=256,):
        super().__init__()
        self.cond_dim = label_emb_size

        self.label_proj = nn.Linear(label_dim, label_emb_size)

        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=3 + label_emb_size,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )


    def forward(self, x, t, y):
        B, ch, H, W = x.shape

        y_proj = self.label_proj(y) # (B, cond_dim)
        y_map = y_proj[:, :, None, None].expand(B, self.cond_dim, H, W)

        x_in = torch.cat([x, y_map], dim=1) # (B, 3+cond_dim, H, W)

        eps_hat = self.model(x_in, t).sample
        return eps_hat
    