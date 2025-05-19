from denoising import train
import torch
import os
from pathlib import Path
from denoising.train import buildDatasetFromTensor
from denoising.train import PRIDLiteArgs, PRIDLiteTrainer, AutoencoderTrainer, AutoencoderArgs, CBDNetArgs, CBDNetTrainer
import argparse
from utils.utils import loadVideoArray
import yaml
from yaml.loader import SafeLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="trainer best models")
    parser.add_argument("--model", type=str, choices=["AE", "CBD", "PRID"])
    parser.add_argument("--data_path", type=str, required=False)


    args_user = parser.parse_args()
    model = args_user.model
    with open('best_param.yml', 'r') as f:
        yaml_data = list(yaml.load_all(f, Loader=SafeLoader))

    # === load video ===
#TODO insert keyframes to use for now.
    if args_user.data_path is not None:
        vid_path = args_user.data_path
        vid_arr = loadVideoArray(vid_path)
        keyframes = torch.Tensor(vid_arr)
    else:
        data_path = '/Users/thomasbush/Documents/Vault/DSS_Tilburg/data/keyframes/_2025-04-22 00:25:47.432414_keyframes.pth'




        data = torch.load(data_path, map_location='cpu', weights_only=False)
        keyframes = data['keyframes']
          # shape: (frames, H, W)
        idx = data['keyframe_idx']

    keyframes = keyframes / 255.0
    train_dataset, test_dataset, orig_shape, padding, holdout_data = buildDatasetFromTensor(keyframes, dim=64)


    trainers = {"PRID":PRIDLiteTrainer,
                "AE":AutoencoderTrainer,
                "CBD": CBDNetTrainer}
    arg_models = {"PRID":PRIDLiteArgs,
                  "AE":AutoencoderArgs,
                  "CBD":CBDNetArgs}

    arg = arg_models[model]
    trainer = trainers[model]
    device = 'mps'
    if arg == PRIDLiteArgs:
        best_p = yaml_data[0]
        lr = int(best_p['Lr'])
        batch_size = int(best_p['BatchSize'])
        model_args = PRIDLiteArgs(
            trainset=train_dataset,
            testset=test_dataset,
            holdoutData=holdout_data,
            original_shape=orig_shape,
            padding=padding,
            lr=lr,
            batch_size=batch_size,
            use_wandb=False,
            wandb_project="thesis_dss_pridnet",
            wandb_name=f"sweep_run_prid"
        )
    elif arg == AutoencoderArgs:
        best_p = yaml_data[1]
        lr = int(best_p['Lr'])
        batch_size = int(best_p["BatchSize"])
        hidden = int(best_p["Hidden"])
        latent = int(best_p["Latent"])
        model_args = AutoencoderArgs(
                    trainset=train_dataset,
                    testset=test_dataset,
                    holdoutData=holdout_data,
                    original_shape=orig_shape,
                    padding=padding,
                    latent_dim_size=latent,
                    hidden_dim_size=hidden,
                    lr=lr,
                    batch_size=batch_size,
                    use_wandb=False,
                    wandb_project="thesis_dss_autoencoder",
                    wandb_name=f"sweep_run_final"
                    )
    elif arg == CBDNetArgs:
        best_p = yaml_data[2]
        lr = int(best_p['Lr'])
        batch_size = int(best_p['BatchSize'])
        model_args = CBDNetArgs(
                    trainset=train_dataset,
                    testset=test_dataset,
                    holdoutData=holdout_data,
                    original_shape=orig_shape,
                    padding=padding,
                    lr=lr,
                    ssim_weights=0.11,
                    batch_size=batch_size,
                    use_wandb=False,
                    wandb_project="thesis-cbdnet",
                    wandb_name=f"sweep_run_cbd"

                )



    trainer_instance = trainer(args=model_args, device=device)
    if trainer == "AE" or "PRID":
        model, loss_f = trainer_instance.train()
    else:
        model = trainer_instance.train()

    torch.save(model.state_dict(), f"/Users/thomasbush/Documents/Vault/DSS_Tilburg/data/models/{args_user.model}model.pth")



