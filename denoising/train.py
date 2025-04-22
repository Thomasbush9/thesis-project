import wandb
import torch as t
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from dataclasses import dataclass
from utils import loadVideoArray
from torch import Tensor
import numpy as np
import argparse
import torch.nn as nn
from models import AutoEncoder


# TODO create holdout data function,  train from keyframes, log if needed, save model if needed and model config 

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
class AutoencoderArgs:
    latent_dim_size: int = 5
    hidden_dim_size: int = 128

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 64
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
        self.trainset = get_dataset(args.dataset)
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
        # pass imgs to model

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
        output = self.model(HOLDOUT_DATA)
        if self.args.use_wandb:
            wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        else:
            display_data(t.concat([HOLDOUT_DATA, output]), nrows=2, title="AE reconstructions")

    def train(self) -> AutoEncoder:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        # YOUR CODE HERE - iterate over epochs, and train your model

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)), ascii=True)

            for imgs, _ in progress_bar:
                imgs = imgs.to(self.device)
                loss = self.training_step(imgs)
                progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f}, step={self.step:05d}")
                #pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.step=:06}")
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
    
