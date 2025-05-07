import wandb
import torch as t
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from dataclasses import dataclass
from utils import loadVideoArray, generatePatches, reconstructFromPatches
from torch import Tensor
import numpy as np
import argparse
import torch.nn as nn
from models import AutoEncoder, CBDNet, PRIDNet
import torch
import einops
from torch.utils.data import random_split
from torchinfo import summary
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from piqa import SSIM
from typing import Literal

# ==== classes data and loss ===
class CustomDataset(Dataset):
    def __init__(self, data:Tensor):
        self.frames = data

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = self.frames[idx]
        return img

# new loss function
class SSIMLoss(SSIM):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, n_channels: int = 1, reduction: str = 'mean', **kwargs):
        super().__init__(window_size, sigma, n_channels, reduction, **kwargs)
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class AsymmetricLoss(nn.Module):
    def __init__(self, alpha: float = 0.3):
        """
        Implements the asymmetric loss from CBDNet.
        Penalizes underestimation more than overestimation.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: predicted noise map (B, C, H, W)
        target: ground-truth noise map (B, C, H, W)
        """
        diff = pred - target
        mask = (diff < 0).float()
        weights = self.alpha * (1 - mask) + (1 - self.alpha) * mask
        loss = weights * (diff ** 2)
        return loss.mean()

# === noise creation ===
def addPoissoinGaussiaNoise(img: torch.Tensor, sigma_s=0.01, sigma_c=0.005):
    """
    Adds heteroscedastic noise: variance depends on pixel intensity.
    Input and output should be in [0, 1] range.
    """
    variance = img * sigma_s**2 + sigma_c**2
    noise = torch.randn_like(img) * torch.sqrt(variance)
    return (img + noise).clamp(0, 1)

def applyInverseCrf(img: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return img ** gamma

def addRealisticNoise(img: torch.Tensor, sigma_s=0.12, sigma_c=0.03) -> torch.Tensor:
    """
    Realistic noise: L + n_s + n_c
    img: (B, C, H, W), float32 in [0,1]
    """
    # Inverse CRF (simulate raw sensor signal)
    linear_img = applyInverseCrf(img)

    # Signal-dependent noise (shot noise)
    noise_s = torch.randn_like(linear_img) * torch.sqrt(torch.clamp(linear_img * sigma_s**2, min=1e-6))

    # Constant noise (read noise)
    noise_c = torch.randn_like(linear_img) * sigma_c

    noisy_img = linear_img + noise_s + noise_c
    noisy_img = torch.clamp(noisy_img, 0, 1)

    # Forward CRF to get final noisy image
    return noisy_img ** (1 / 2.2)  # CRF simulation
def add_physical_noise(img, gamma=2.2, poisson_scale=5000, sigma_c=0.01):
    """
    Simulates realistic sensor noise:
    - inverse CRF
    - true Poisson noise
    - additive Gaussian noise
    - forward CRF
    """

    # 1. Inverse CRF
    img_lin = torch.clamp(img ** gamma, 0, 1)

    # 2. Poisson shot noise (must run on CPU if on MPS)
    device = img_lin.device
    img_cpu = img_lin.cpu()
    photon_img = torch.poisson(img_cpu * poisson_scale) / poisson_scale
    photon_img = photon_img.to(device)

    # 3. Gaussian read noise
    noise_c = torch.randn_like(img_lin) * sigma_c

    noisy_lin = torch.clamp(photon_img + noise_c, 0, 1)

    # 4. Forward CRF
    return torch.clamp(noisy_lin ** (1 / gamma), 0, 1)




def buildDatasetFromTensor(input:Tensor, dim:int, train_ratio:float=.8)->tuple:

    assert len(input.shape) == 3, 'Input tensor must be frame h w'
    patches_frames, original_shape, padding = generatePatches(input, dim)

    tot_patches = einops.rearrange(patches_frames, 'f p h w -> (f p ) 1 h w', h=dim, w=dim).type(torch.float)

    dataset = CustomDataset(tot_patches)

    # Split into train/test
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    test_len = total_len - train_len

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    return train_dataset, test_dataset, original_shape, padding


def getHoldoutData(testset:Dataset, num_data:int=40)->Tensor:
    """Ideally get a whole img not in training"""

    assert len(testset) % num_data==0, f'{num_data} makes a whole img'
    # extract imgs
    imgs = next(iter(DataLoader(testset, batch_size=num_data, shuffle=False)))
    return imgs
# === AutoEncoder Trainer ===
@dataclass
class AutoencoderArgs:
    trainset: Dataset
    testset: Dataset
    holdoutData: Tensor
    original_shape:tuple
    padding:tuple

    latent_dim_size: int = 20
    hidden_dim_size: int = 128

    # data / training
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "thesis_dss_CBDNet"
    wandb_name: str | None = None
    log_every_n_steps: int = 250


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs, device:Literal['cpu', 'mps']):
        self.args = args
        self.device = device
        self.trainset = args.trainset
        self.HOLDOUT_DATA = args.holdoutData
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.model = AutoEncoder(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(self.device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        self.step = 0
        self.loss_mse = nn.MSELoss()
        self.loss_ssim = SSIMLoss().to(self.device)

    def training_step(self, noisy:Tensor ,original: Tensor) -> Tensor:
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb if enabled.
        """
        pred = self.model.forward(noisy)
        pred1 = torch.clamp(pred, 0.0, 1.0)
        loss_ssim = self.loss_ssim(pred1, original)
        loss_mse = self.loss_mse(pred, original)
        loss = loss_mse + loss_ssim
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += original.shape[0]

        if self.args.use_wandb:
          wandb.log(dict(loss=loss,
                         SSIM=loss_ssim, MSE=loss_mse), step=self.step)


        return loss, loss_ssim, loss_mse
# TODO: make patches in right order
    @t.inference_mode()
    def log_samples(self) -> None:
        assert self.step > 0, "Call after training step."

        self.model.eval()
        output_patches = self.model(self.HOLDOUT_DATA.to(self.device)).cpu()
        original_patches = self.HOLDOUT_DATA.cpu()

        original_patches = original_patches.squeeze(1).unsqueeze(0)
        output_patches = output_patches.squeeze(1).unsqueeze(0)

        # Reconstruct full images
        reconstructed_img = reconstructFromPatches(output_patches, self.args.original_shape, self.args.padding)
        original_img = reconstructFromPatches(original_patches, self.args.original_shape, self.args.padding)

        # Combine vertically
        comparison = torch.cat([original_img, reconstructed_img], dim=1)  # shape: (1, H*2, W)

        # Make into grid image and send to WandB
        img_grid = make_grid(comparison, normalize=True, scale_each=True)
        if self.args.use_wandb:
            wandb.log({"reconstruction": wandb.Image(img_grid)}, step=self.step)
        else:
            plt.imshow(make_grid(img_grid).permute(1, 2, 0))

    def train(self) -> AutoEncoder:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)), ascii=True)

            for imgs in progress_bar:
                imgs = imgs.to(self.device)
                # noisy = imgs + 0.1 * torch.randn_like(imgs)
                # noisy = noisy.clamp(0, 1)
                # noisy = addPoissoinGaussiaNoise(imgs)
                noisy = addRealisticNoise(imgs, sigma_s=0.1, sigma_c=0.05)
                loss, loss_ssim, loss_mse = self.training_step(noisy,imgs)
                progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f}, MSE:{loss_mse:.4f}, SSIM:{loss_ssim:.4f} step={self.step:05d}")
                # log every 250 steps
                if self.step % self.args.log_every_n_steps == 0:
                  self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model
# ==== CBDNet Trainer ===

@dataclass
class CBDNetArgs:
    trainset: Dataset
    testset: Dataset
    holdoutData: Tensor
    original_shape:tuple
    padding:tuple

    # data / training
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "thesis_dss_autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250

class CBDNetTrainer():
    def __init__(self, args:CBDNetArgs, device:Literal['cpu', 'mps']) -> None:

        self.args = args
        self.device = device
        self.trainset = args.trainset
        self.HOLDOUT_DATA = args.holdoutData
        self.trainloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
        #define losses, optim, model:
        self.model = CBDNet().to(self.device)
        self.asym_loss = AsymmetricLoss()
        self.loss_rec = nn.MSELoss()
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=self.args.betas)
#TODO change loss with SSIM
    def training_step(self, imgs:Tensor, noisy_imgs:Tensor):
        '''Perform a single training step'''
        self.model.train()
        real_sigma = t.abs(imgs - noisy_imgs)
        pred_img, pred_sigma = self.model.forward(noisy_imgs)
        loss_ne = self.asym_loss(pred_sigma, real_sigma)
        loss = self.loss_rec(imgs, pred_img)
        loss.backward()
        # loss = .7 *loss_mse+  1.5 * loss_ne
        # loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += imgs.shape[0]

        if self.args.use_wandb:
          wandb.log(dict(loss=loss,
                         AsymLoss=loss_ne, MSE=loss), step=self.step)

        return loss, loss_ne

    @t.inference_mode()
    def log_samples(self) -> None:
        assert self.step > 0, "Call after training step."

        self.model.eval()
        output_patches = self.model(self.HOLDOUT_DATA.to(self.device))[0].detach().cpu()
        output_patches = output_patches.cpu()
        original_patches = self.HOLDOUT_DATA.cpu()

        original_patches = original_patches.squeeze(1).unsqueeze(0)
        output_patches = output_patches.squeeze(1).unsqueeze(0)

        # Reconstruct full images
        reconstructed_img = reconstructFromPatches(output_patches, self.args.original_shape, self.args.padding)
        original_img = reconstructFromPatches(original_patches, self.args.original_shape, self.args.padding)

        # Combine vertically
        comparison = torch.cat([original_img, reconstructed_img], dim=1)  # shape: (1, H*2, W)

        # Make into grid image and send to WandB
        img_grid = make_grid(comparison, normalize=True, scale_each=True)
        if self.args.use_wandb:
            wandb.log({"reconstruction": wandb.Image(img_grid)}, step=self.step)


    def train(self)->CBDNet:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)), ascii=True)

            for imgs in progress_bar:
                imgs = imgs.to(self.device)
                noisy = addRealisticNoise(imgs, sigma_s=0.08, sigma_c=0.02)
                loss, loss_ne = self.training_step(imgs, noisy)
                progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f},  NE:{loss_ne:.4f} step={self.step:05d}")
                # log every 250 steps
                if self.step % self.args.log_every_n_steps == 0:
                  self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model

@dataclass
class PRIDNetArgs:
    trainset: Dataset
    testset: Dataset
    holdoutData: Tensor
    original_shape:tuple
    padding:tuple

    # data / training
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "thesis_dss_autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250

class PRIDNetTrainer():
    def __init__(self, args:PRIDNetArgs, device:Literal['mps', 'cpu']):
        self.args = args
        self.device = device
        self.model = PRIDNet().to(self.device)
        self.trainset = args.trainset
        self.HOLDOUT_DATA = args.holdoutData
        self.trainloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
        self.loss_rec = nn.MSELoss()
        self.loss_ssim = SSIMLoss().to(self.device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=self.args.betas)

    def training_step(self, imgs:Tensor, noisy_img:Tensor):
        self.model.train()
        noisy_img = noisy_img
        pred = self.model.forward(noisy_img)
        loss_rec = self.loss_rec(imgs, pred)
        # loss_ssim = self.loss_ssim(imgs, pred)
        loss_ssim = 0
        loss = loss_rec + loss_ssim
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += imgs.shape[0]

        if self.args.use_wandb:
           wandb.log(dict(loss=loss, 
                          rec=loss_rec, ssim=loss_ssim), step=self.step)
        
        return loss, loss_rec, loss_ssim
    @t.inference_mode()
    def log_samples(self)->None:

        assert self.step > 0, "Call after training step."

        self.model.eval()
        output_patches = self.model(self.HOLDOUT_DATA.to(self.device)).cpu()
        original_patches = self.HOLDOUT_DATA.cpu()

        original_patches = original_patches.squeeze(1).unsqueeze(0)
        output_patches = output_patches.squeeze(1).unsqueeze(0)

        # Reconstruct full images
        reconstructed_img = reconstructFromPatches(output_patches, self.args.original_shape, self.args.padding)
        original_img = reconstructFromPatches(original_patches, self.args.original_shape, self.args.padding)

        # Combine vertically
        comparison = torch.cat([original_img, reconstructed_img], dim=1)  # shape: (1, H*2, W)

        # Make into grid image and send to WandB
        img_grid = make_grid(comparison, normalize=True, scale_each=True)
        if self.args.use_wandb:
            wandb.log({"reconstruction": wandb.Image(img_grid)}, step=self.step)
        else:
            plt.imshow(make_grid(img_grid).permute(1, 2, 0))

    def train(self)->PRIDNet:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)), ascii=True)

            for imgs in progress_bar:
                imgs = imgs.to(self.device)
                noisy = addRealisticNoise(imgs, sigma_s=0.08, sigma_c=0.02)
                noisy = noisy.to(self.device)
                loss, loss_rec, loss_ssim = self.training_step(imgs, noisy)
                progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f},  MSE:{loss_rec:.4f}, SSIM:{loss_ssim:.4f} step={self.step:05d}")
                # log every 250 steps
                if self.step % self.args.log_every_n_steps == 0:
                    self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="Denoising training")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--m", type=str, choices=["CBDN", "AE", "PR"])



    args = parser.parse_args()
    data_path = args.data_path

    # data_path = '/Users/thomasbush/Documents/DSS_Tilburg/data/keyframes/_2025-04-22 00:25:47.432414_keyframes.pth'


    data = torch.load(data_path, map_location='cpu', weights_only=False)
    keyframes = data['keyframes']
    keyframes = keyframes / 255.0  # shape: (frames, H, W)
    idx = data['keyframe_idx']

    train_dataset, test_dataset, orig_shape, padding = buildDatasetFromTensor(keyframes, dim=64)

    # args_trainer = AutoencoderArgs(trainset=train_dataset, testset=test_dataset, holdoutData=getHoldoutData(test_dataset), original_shape=orig_shape, padding=padding,
    #                                 use_wandb=False)
    args_trainer = PRIDNetArgs(trainset=train_dataset, testset=test_dataset, holdoutData=getHoldoutData(test_dataset), original_shape=orig_shape, padding=padding,
                                    use_wandb=False)

# === Start Trainign ===

    # trainer = AutoencoderTrainer(args_trainer, device='mps') if args.m == "AE"  else CBDNetTrainer(args_trainer, device='mps')
    trainer = PRIDNetTrainer(args_trainer, device='mps')
    autoencoder = trainer.train()
    # summary(autoencoder, (len(train_dataset),1, 64, 64), device='mps')








