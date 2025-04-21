import wandb
import torch as t
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from dataclasses import dataclass
from utils import loadVideoArray
from torch import Tensor
import numpy as np


# TODO create holdout data function 

class CustomDataset(Dataset):
    def __init__(self, data:np.ndarray):
        self.frames = Tensor(data)

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        img = self.frames[idx]  # shape: [H, W]
        return img


# create data loader
@dataclass
class AutoEncoderArgs:
    latent_dim_size: int = 5
    hidden_dim_size: int = 128

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "day5-autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250

class AutoEncoderTrainer:
    def __init__(self, args:AutoEncoderArgs):
        self.args = args



if __name__ == '__main__':
    print('Hello')
