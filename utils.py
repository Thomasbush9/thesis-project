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
    
def generatePatches(input:Tensor, dim:int)->Tensor:
    """given tensor (frames, h, w) and dim -> tensor (frames, (h\dim * w\dim), dim, dim) """
    
    assert len(input.shape) == 3, 'Input must be (frames, height, width)'
    n, h, w = input.shape
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

    return patches
