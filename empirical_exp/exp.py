import torch as t
import os
from pathlib import Path
from utils.utils import loadVideoArray, generatePatches, reconstructFromPatches
from denoising.models import PRIDLite
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from einops import rearrange

#TODO from video run models
#=== script that run models saved models on video ===
'''
/Users/thomasbush/Documents/Vault/Iurilli_lab/3d_tracking/data/multicam_video_2024-07-22T10_19_22_cropped_20250325101012/multicam_video_2024-07-22T10_19_22_mirror-left.avi.mp4
'''
def denoiseVideo(patches: t.Tensor, model, original_shape, padding, batch_size=32, device="mps"):
    model.to(device)
    model.eval()

    denoised_patches = []

    with t.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Denoising patches"):
            batch = patches[i:i+batch_size].to(device)
            out = model(batch)
            denoised_patches.append(out.cpu())

    denoised_patches = t.cat(denoised_patches, dim=0)  # shape: [N, 1, 64, 64]
    denoised_patches = denoised_patches.squeeze()
    denoised_patches = rearrange(denoised_patches, '(b p) h w -> b p h w', p=40)

    # Reconstruct the denoised video from patches
    denoised_video = reconstructFromPatches(
        denoised_patches,
        original_shape=original_shape,
        padding=padding)
    return denoised_video



def save_video_from_tensor(video_tensor: t.Tensor, save_path: str, fps: int = 30):
    """
    Save a grayscale video from a PyTorch tensor.
    Args:
        video_tensor: shape (frames, H, W), values in [0, 1] or [0, 255]
        save_path: output .mp4 file path
        fps: frames per second
    """
    video_np = video_tensor.cpu().numpy()

    # Rescale if needed
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).clip(0, 255)

    video_np = video_np.astype(np.uint8)

    frames, H, W = video_np.shape

    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H), isColor=False)

    for frame in video_np:
        out.write(frame)

    out.release()
    print(f"Video saved to {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parser to run model")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--model", type=str, choices=["CBD", "AE", "PRID"])

    args = parser.parse_args()
    vid_path = args.video_path
    model_c = args.model
    vid = loadVideoArray(vid_path)
    vid_tensor = t.Tensor(vid)
    vid_tensor = vid_tensor[:1000]
    print(vid_tensor.shape)
    vid_patches, original_shape, padding  = generatePatches(vid_tensor, dim=64)
    print(vid_patches.shape)
    vid_patches = rearrange(vid_patches, 'b p h w -> (b p) 1 h w', p=40)


    # === load model ===
    model_path = '/Users/thomasbush/Documents/Vault/DSS_Tilburg/data/models/best_prid_model.pth'
    model = PRIDLite()
    model.load_state_dict(t.load(model_path, map_location="cpu"))

    #=== run model ===
    denoised_video = denoiseVideo(
            patches=vid_patches,
            model=model,
            original_shape=original_shape,
            padding=padding,
            device="mps")
    output_dir = "/Users/thomasbush/Documents/Vault/DSS_Tilburg/data/denoised_videos"
    os.makedirs(output_dir, exist_ok=True)

    video_filename = Path(vid_path).stem + "_denoised.mp4"
    save_path = os.path.join(output_dir, video_filename)

    save_video_from_tensor(denoised_video, save_path)





