import numpy as np
import einops
from tqdm import tqdm
from pathlib import Path
import cv2
import torch as t
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch import Tensor
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as FT


# TODO: get dataset from patht to keyframes, train and test (holdout data)

def captureVideo(video_path:str)->np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # For grayscale
        frames.append(frame)
    cap.release()
    return  np.array(frames)

def loadVideoArray(video_path:str | list )->np.ndarray| dict:
    '''
    Returns dict of np.arrays per video (if one video it returns one array)
    '''

    if isinstance(video_path, str): #single video
        return captureVideo(video_path)


    if isinstance(video_path, list):

        return {video:captureVideo(video) for video in tqdm(video_path, desc='Loading videos')}



# === datasets from videos ===
class CustomDataset(Dataset):
    def __init__(self, data:np.ndarray):
        self.frames = Tensor(data)

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = self.frames[idx]  # shape: [H, W]
        return img

# === patches generation ===
def splitPadding(pad:int)-> tuple:
    if pad % 2== 0:
        pad = pad / 2
        return (int(pad), int(pad))
    else:
        pad1 = pad //2
        pad2 = pad //2 + (pad %2)
        return (int(pad1), int(pad2))

# TODO: optional: add non squargrid = make_grid(img0, nrow=ncols, padding=0, normalize=False)ed patches,  add option to flat it
def generatePatches(input:Tensor, dim:int)->Tensor:
    """given tensor (frames, h, w) and dim -> tensor (frames, (h\dim * w\dim), dim, dim) """

    assert len(input.shape) == 3, 'Input must be (frames, height, width)'
    n, h, w = input.shape
    original_shape = (n, h, w)
    # pad dims if needed:
    rh = h % dim
    padh = (dim - rh) if rh !=0 else 0
    rw = w % dim
    padw = (dim - rw) if rw !=0 else 0

    padw1, padw2 = splitPadding(padw)

    padh1, padh2 = splitPadding(padh)

    input = F.pad(input, pad=(padw1, padw2, padh1, padh2))

    input = input.unsqueeze(1)

    patches = input.unfold(2, dim, dim).unfold(3, dim, dim)
    patches = patches.contiguous().view(n, -1, dim, dim)

    return patches, original_shape, (padh1, padh2, padw1, padw2)

def reconstructFromPatches(patches: Tensor, original_shape: tuple, padding: tuple) -> Tensor:
    """Reconstructs original images from patches tensor
        patches: (frames, patches, dim, dim)
        original shape : (frames, h, w)
        padding: (ph1, ph2, pw1, pw2)

        returns
        reconstructed: (frames, h ,w)
    """
    n, num_patches, ph, pw = patches.shape
    if num_patches ==1:
        patches = patches.transpose(1,0)
        n, num_patches, ph, pw = patches.shape
    _, h_orig, w_orig = original_shape
    padh1, padh2, padw1, padw2 = padding

    H_pad = h_orig + padh1 + padh2
    W_pad = w_orig + padw1 + padw2


    nh = H_pad // ph
    nw = W_pad // pw

    assert nh * nw == num_patches, "Mismatch between number of patches and target shape"

    frames = patches.view(n, nh, nw, ph, pw)
    frames = frames.permute(0, 1, 3, 2, 4).contiguous()
    frames = frames.view(n, nh * ph, nw * pw)


    frames = frames[:, padh1:H_pad - padh2, padw1:W_pad - padw2]
    return frames


def plot_image_with_patches(patches_tensor, original_shape, patch_dim, padding, save_path: None|Path):
    """
    patches: tensor of shape (N, d, d) or (N, 1, d, d)
    original_shape: tuple (H, W)
    patch_dim: int
    padding: tuple (top, bottom, left, right)
    """
    H, W = original_shape
    pad_top, pad_bottom, pad_left, pad_right = padding

    # Total padded dimensions
    H_p = H + pad_top + pad_bottom
    W_p = W + pad_left + pad_right

    # Convert patches to tensor if needed
    if isinstance(patches_tensor, np.ndarray):
        patches_tensor = t.tensor(patches_tensor)

    # If patches have channels, remove it
    if patches_tensor.dim() == 4 and patches_tensor.shape[1] == 1:
        patches_tensor = patches_tensor.squeeze(1)

    # Reconstruct the image from patches
    num_patches_h = H_p // patch_dim
    num_patches_w = W_p // patch_dim
    assert patches_tensor.shape[0] == num_patches_h * num_patches_w

    patches_grid = patches_tensor.view(num_patches_h, num_patches_w, patch_dim, patch_dim)
    reconstructed = patches_grid.permute(0, 2, 1, 3).contiguous().view(H_p, W_p)

    # Remove the padding to recover the original image
    reconstructed = reconstructed[pad_top:pad_top+H, pad_left:pad_left+W]

    # Plot the image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(reconstructed, cmap='gray')
    # ax.set_title("Image with Patch Boundaries")

    for i in range(0, H_p, patch_dim):
        for j in range(0, W_p, patch_dim):
            # Draw rectangles in padded space
            y = i - pad_top
            x = j - pad_left
            rect = patches.Rectangle((x, y), patch_dim, patch_dim,
                                            linewidth=0.5, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path / "patches_overlay.svg", format="svg", bbox_inches="tight")
    plt.show()

def viewAsImage(img:Tensor, original_shape:tuple, padding:tuple, dim:int)->Tensor:
    assert len(original_shape) == 3, "Shape must be: patches, H, W"
    assert len(padding) == 4, "Padding must be 4 dims"
    _, h, w = original_shape
    p_top, p_bottom, p_left, p_right = padding

    ncols = int((w+ p_left + p_right)//dim)
    nrows = int((h + p_top + p_bottom) // dim)

    return ncols, nrows

def displayPatches(img:Tensor, ncols:int, nrows:int):
    assert img.shape[0] == ncols*nrows, 'patches must be same as dims'
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), squeeze=False)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            axs[i, j].imshow(FT.to_pil_image(img[idx]), cmap='gray')
            axs[i, j].axis('Off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
