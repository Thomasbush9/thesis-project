import torch as t
import numpy as np
import einops
from einops.layers.torch import Rearrange
from torch.nn import Conv2d, ConvTranspose2d, Sequential, ReLU, BatchNorm2d, Linear, Sigmoid, AvgPool2d, AdaptiveAvgPool2d
from torch import Tensor, nn
from utils import generatePatches

#TODO: review and see if you have to init the weights 
def weights_init(m):
    if isinstance(m, Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
# ==== Baseline AutoEncoder ====
# img = 1, 64, 64
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim_size:int, hidden_dim_size:int):
        """Creates Encoder and Decoder modules"""
        super().__init__()
        self.apply(weights_init)
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


# === VAE ===

#TODO: optimize network and define better loss (maybe just MSE)
# ==== CBDNet — Convolutional Blind Denoising Network ===
class CBDNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ==== Noise Estimator ====
        self.noise_estimator = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 1, kernel_size=3, padding=1),
            ReLU()
        )

        # === Encoder ===
        self.conv1 = nn.Sequential(
            Conv2d(2, 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool1 = AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU()
        )
        self.pool2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU()
        )

        # === Decoder ===
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv = nn.Sequential(
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU()
        )

        self.output_conv = Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        noise_map = self.noise_estimator(x)
        x_cat = t.cat([x, noise_map], dim=1)

        c1 = self.conv1(x_cat)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)

        up1 = self.up1(c3)
        up1 = up1 + c2  # skip connection
        up1 = self.up1_conv(up1)

        up2 = self.up2(up1)
        up2 = up2 + c1  # skip connection
        up2 = self.up2_conv(up2)

        out = self.output_conv(up2)
        final = out + x  # residual connection

        return final, noise_map

        

# === PRIDNet: pyramid real image denoising network ===
class AttentionUnit(nn.Module):
    def __init__(self, channels, reduction:int=4):
        super().__init__()

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channels, channels // reduction, bias=False),
            ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
            Sigmoid()
        )
    def forward(self, x:Tensor)->Tensor:
        b, c, w, h = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  x * y.expand_as(x)



class PRIDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cam = Sequential(*[
            Conv2d(in_channels=40 if i==0 else 64, out_channels=64, kernel_size=3, padding=1)
            ReLU()
            for i in range(4)
        ])
        self.attention_unit = AttentionUnit(channels=64, reduction=4)
        self.conv = Sequential(Conv2d(in_channels=64, out_channels=40, kernel_size=3, padding=1), ReLU())
        
        self.pyramid = None

        self.fusion = None

    def forward(self, x:Tensor)-> Tensor:
        x_prime = self.cam(x)
        x_prime = self.attention_unit.forward(x_prime)
        x_prime = self.conv(x_prime)
