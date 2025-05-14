#%%
import torch as t
import numpy as np
import einops
from einops.layers.torch import Rearrange
from torch.nn import Conv2d, ConvTranspose2d, Sequential, ReLU, BatchNorm2d, Linear, Sigmoid, AvgPool2d, AdaptiveAvgPool2d
from torch import Tensor, nn
from torch.nn.functional import interpolate, unfold
import torch.nn.functional as F
from utils import generatePatches
import math
from einops import rearrange
#%%
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
        self.apply(weights_init)
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
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )
        self.pool1 = AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU()
        )
        self.pool2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU()
        )

        # === Decoder ===
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv = nn.Sequential(
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
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
        final = x + t.tanh(out) * .1
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

def extract_sliding_local_patches(x: t.Tensor, kernel_size: int) -> t.Tensor:
    """
    Extract sliding local patches (unfold) using as_strided (MPS-friendly).
    Returns: Tensor of shape (B, C, kernel_size*kernel_size, H, W)
    """
    B, C, H, W = x.shape
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')  # [B, C, H+2p, W+2p]

    # Get sliding window view using as_strided
    # Shape: B x C x H x W x k x k
    x_strided = x.as_strided(
        size=(B, C, H, W, kernel_size, kernel_size),
        stride=(
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(2),
            x.stride(3),
        ),
    )

    # Rearrange to: B x C x (k*k) x H x W
    patches = rearrange(x_strided, "b c h w kh kw -> b c (kh kw) h w")
    return patches

class LocalContextAttention(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.attn_proj = nn.Sequential(
            Conv2d(in_channels, in_channels, 1),
            ReLU(),
            Conv2d(in_channels, in_channels, 1),
            Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_proj = self.attn_proj(x).unsqueeze(2)  # B x C x 1 x H x W

        patches = extract_sliding_local_patches(x, self.kernel_size)  # B x C x k^2 x H x W

        attn = t.softmax(patches * x_proj, dim=2)
        out = (attn * patches).sum(dim=2)  # B x C x H x W
        return out

class PyramidPooling(nn.Module):

    def __init__(self):
        super().__init__()
        self.pooling2 = AvgPool2d(kernel_size=2, stride=2, padding=0) # 32
        self.pooling3 = AvgPool2d(kernel_size=2, stride=4, padding=0) # 16

    def forward(self, x:Tensor):

        p1 = x
        p2 = self.pooling2(x)
        p3 = self.pooling3(x)
        return p1, p2, p3


class UNET(nn.Module):
    def __init__(self, input_size: int, in_channels: int):
        super().__init__()
        assert input_size >= 16 and (input_size & (input_size - 1)) == 0, "Input size must be a power of 2 and ≥ 16"

        base_channels = 64
        self.depth = int(math.log2(input_size)) - 4  # e.g., 64 → 2, 32 → 1, 16 → 0

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Conv layers for encoding (without pooling inside)
        for i in range(self.depth + 1):
            inc = in_channels if i == 0 else base_channels
            self.encoder.append(nn.Sequential(
                nn.Conv2d(inc, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

        # Decoder layers: upsample + conv
        for _ in range(self.depth):
            self.decoder.append(nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        out = x

        # Encoder: pool AFTER saving skip, not inside conv block
        for i in range(self.depth):
            out = self.encoder[i](out)
            skips.append(out)
            out = self.pool(out)

        # Final encoder conv (bottom of the U)
        out = self.encoder[-1](out)

        # Decoder
        for dec in self.decoder:
            out = self.upsample(out)
            skip = skips.pop(-1)
            if out.shape[-2:] != skip.shape[-2:]:
                out = t.nn.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            out = t.cat([out, skip], dim=1)
            out = dec(out)

        return self.final(out)

class FeatureFusions(nn.Module):

    def __init__(self, in_channels:int, reduction:int=4):
        super().__init__()

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2)
        self.conv3 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3)
        self.avg_pool = AdaptiveAvgPool2d(1)

        self.fc = Sequential(
            Linear(in_channels, in_channels // reduction, bias=False),
            ReLU(inplace=True),
            )

        # logit attention heads:
        self.alpha = Linear(in_channels//reduction, in_channels, bias=False)
        self.beta = Linear(in_channels//reduction, in_channels, bias=False)
        self.gamma = Linear(in_channels//reduction, in_channels, bias=False)



    def forward(self, x:Tensor)->Tensor:
        b, c, w, h = x.shape
        out = x
        u1 = self.conv1(x)
        u2 = self.conv2(x)
        u3 = self.conv3(x)

        tot = u1 + u2 + u3

        y = self.avg_pool(tot).view(b, c)

        s = self.fc(y)

        a = self.alpha(s)
        b_ = self.beta(s)
        g = self.gamma(s)

        weights = t.stack([a, b_, g], dim=1)
        weights = t.softmax(weights, dim=1)

        alpha, beta, gamma = weights.unbind(dim=1)
        assert alpha.shape == (b, c), 'ERROR shape'

        alpha = alpha.view((b, c, 1, 1))
        beta = beta.view((b, c, 1, 1))
        gamma = gamma.view((b, c, 1, 1))

        return u1 * alpha + u2 * beta + u3 * gamma


class PRIDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cam = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # self.attention_unit = AttentionUnit(channels=64, reduction=4)
        self.local_attention_unit = LocalContextAttention(in_channels=64, kernel_size=7)
        self.conv = Sequential(Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), ReLU())

        self.gen_ps = PyramidPooling()

        self.unet1 = UNET(input_size=64, in_channels=64)
        self.unet2 = UNET(input_size=32, in_channels=64)
        self.unet3 = UNET(input_size=16, in_channels=64)



        self.fusion = FeatureFusions(in_channels=64 * 4)  # = 256
        self.output = Conv2d(256, 64, kernel_size=1)
        self.final_out = Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x:Tensor)-> Tensor:
        x_prime = self.cam(x)
        x_prime = self.local_attention_unit.forward(x_prime)
        x_prime = self.conv(x_prime)
        # decide whether to add another cat

        p1, p2, p3 = self.gen_ps(x_prime)

        p1_prime, p2_prime, p3_prime = self.unet1(p1), self.unet2(p2), self.unet3(p3)

        p2_prime = interpolate(p2_prime, size=64, mode='bilinear', align_corners=False)
        p3_prime = interpolate(p3_prime, size=64, mode='bilinear', align_corners=False)

        # Step 6: Concatenate all scales + attention output
        x_concat = t.cat((p1_prime, p2_prime, p3_prime, x_prime), dim=1)  # → [B, 256, 64, 64]

        # Step 7: Fusion and output
        fused = self.fusion(x_concat)       # → [B, 256, 64, 64]
        out = self.output(fused)            # → [B, 64, 64, 64]
        out = self.final_out(out)           # → [B, 1, 64, 64] (final denoised grayscale image)

        return out




