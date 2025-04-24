import torch as t
import numpy as np
import einops
from einops.layers.torch import Rearrange
from torch.nn import Conv2d, ConvTranspose2d, Sequential, ReLU, BatchNorm2d, Linear
from torch import Tensor, nn

#TODO: review and see if you have to init the weights 
# ==== Baseline AutoEncoder ====
# img = 1, 64, 64
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

# === Sparse AutoEncoder ===


# === VAE or UNET ===


# ==== CBDNet — Convolutional Blind Denoising Network ===
class CBDNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


# ==== Noise estimator network ====
        self.noise_estimator = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1),
            ReLU(),
        )
        # === Denoiser Network ===
        self.cnet_conv1 = Sequential(
            Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            ReLU(),
        )
        self.cnet_conv2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(), )
        self.cnet_conv3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            ReLU(),
        )

        self.cnent_transposed1 = Sequential(
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            ReLU(),

        )
        self.cnent_transposed2 = Sequential(
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            ReLU(),
        )
        self.encoder = Sequential(self.cnet_conv1, self.cnet_conv2, self.cnet_conv3)
        self.decoder = Sequential(self.cnent_transposed1, self.cnent_transposed2)

    def forward(self, x:Tensor) -> Tensor:
        """
        Perform forward pass with skip connections, returns:
        - reconstruncted image
        - noise map
        """
        noise_map = self.noise_estimator(x)
        out = t.cat([x, noise_map], dim=1)  
        acts = []
        # store activations for skipping connections 
        acts.append(x)
        for layer in self.encoder:
            out = layer(out)
            acts.append(out)
        for a, layer in zip(acts[::-1], self.decoder):
            out = layer(t.cat([out, a], dim=1))
        return out, noise_map
        

        


# def forward(self, x):
#     out = x
#     acts = []
#     for layer in self.encoder:
#         out = layer(out)
#         acts.append(out)

#     for a, layer in zip(acts[::-1], self.decoder):
#         out = layer(out, a)

#     return out
# === Residual Network === 
