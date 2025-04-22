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
from models import AutoEncoder
import torch
import einops
from torch.utils.data import random_split

class CustomDataset(Dataset):
    def __init__(self, data:Tensor):
        self.frames = data

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        img = self.frames[idx]  
        return img


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
    imgs = next(iter(DataLoader(test_dataset, batch_size=num_data, shuffle=False)))
    return imgs 

@dataclass
class AutoencoderArgs:
    trainset: Dataset
    testset: Dataset
    holdoutData: Tensor

    latent_dim_size: int = 16
    hidden_dim_size: int = 128

    # data / training
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "thesis_dss_autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs, device:str):
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
        self.loss = nn.MSELoss()

    def training_step(self, img: Tensor) -> Tensor:
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb if enabled.
        """
        pred = self.model.forward(img)
        loss = self.loss(pred, img)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += img.shape[0]

        if self.args.use_wandb:
          wandb.log(dict(loss=loss), step=self.step)

        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to weights & biases or displaying output.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(self.HOLDOUT_DATA.to(self.device).float())
        # if self.args.use_wandb:
        #     wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        # else:
        #     display_data(t.concat([HOLDOUT_DATA, output]), nrows=2, title="AE reconstructions")

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
                loss = self.training_step(imgs)
                progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f}, step={self.step:05d}")
                # log every 250 steps
                if self.step % self.args.log_every_n_steps == 0:
                  self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="Denoising training")
    parser.add_argument("--data_path", type=str)


    args = parser.parse_args()
    data_path = args.data_path

    data = torch.load(data_path, map_location='cpu', weights_only=False)
    keyframes = data['keyframes']
    idx = data['keyframe_idx']

    train_dataset, test_dataset, orig_shape, padding = buildDatasetFromTensor(keyframes, dim=64)
    args_trainer = AutoencoderArgs(trainset=train_dataset, testset=test_dataset, holdoutData=getHoldoutData(test_dataset))

# === Start Trainign ===
    trainer = AutoencoderTrainer(args_trainer, device='mps')
    autoencoder = trainer.train()



    
