#%%
import torch as t
import numpy as np
import einops
from einops.layers.torch import Rearrange
from torch.nn import Conv2d, ConvTranspose2d, Sequential, ReLU, BatchNorm2d, Linear
from torch import Tensor, nn
#%%
# ==== Baseline AutoEncoder ====
# img = c, 64, 64
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim_size:int, hidden_dim_size:int):
        """Creates Encoder and Decoder modules"""
        super().__init__()
        self.latent = latent_dim_size
        self.hidden = hidden_dim_size
        c2l = Rearrange('b c h w -> b (c h w)')
        l2c = Rearrange('b (c h w) -> b c h w', c=32, h=16, w=16)

        #build modules:
        self.encoder = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            BatchNorm2d(num_features=16),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1), 
            ReLU(),
            c2l,
            Linear(in_features=32*16*16, out_features=self.hidden),
            ReLU(),
            Linear(in_features=self.hidden, out_features=self.latent)
             )
        self.decoder = Sequential(
            Linear(self.latent, self.hidden),
            ReLU(),
            Linear(self.hidden, 32*16*16),
            ReLU(),
            l2c,
            ConvTranspose2d(32, 16, 4, 2, 1),
            BatchNorm2d(16),
            ReLU(),
            ConvTranspose2d(16, 1, 4, 2, 1)
        )
    def forward(self, x:Tensor)->Tensor:
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime

