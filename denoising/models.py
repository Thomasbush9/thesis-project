import torch as t
import numpy as np
import einops
from einops.layers.torch import Rearrange
from torch.nn import Conv2d, ConvTranspose2d, Sequential, ReLU, BatchNorm2d, Linear
from torch import Tensor, nn

# ==== Baseline AutoEncoder ====

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim_size:int, hidden_dim_size:int):
        """Creates Encoder and Decoder modules"""
        self.latent = latent_dim_size
        self.hidden = hidden_dim_size
        self.feature_shape = None
        #build modules:
        self.encoder_conv = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            BatchNorm2d(num_features=16),
            Conv2d(in_channels=16, out_channels=32),
        )
        self.encoderfc = Sequential(
            Linear(in_features=32, out_features=self.hidden),
            ReLU(),
            Linear(in_features=self.hidden, out_features=self.latent)
        )
        self.decoder_fc = Sequential(
            Linear(self.latent, self.hidden),
            ReLU(),
            Linear(self.hidden, 32*7*7),
            ReLU()
        )
        self.decoder_conv = Sequential(
            ConvTranspose2d(32, 16, 4, 2, 1),
            BatchNorm2d(16),
            ReLU(),
            ConvTranspose2d(16, 1, 4, 2, 1)
        )
    def forward(self, x:Tensor)->Tensor:

        xf = self.encoder_conv(x)
        self.feature_shape = xf.shape[1:] #(c, h w)
        xf = xf.view(xf.size(0), -1)
        z = self.encoderfc(xf)

        z_fc = self.decoder_fc(z)
        z_fc = z_fc.view(-1, *self.feature_shape)  
        x_prime = self.decoder_conv(z_fc)
        x_prime = self.decoder_conv(z)
        return x_prime
    
# === Variational Autoencoder === 