from utils import loadVideoArray
import torch 
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
import torchvision.transforms as T
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm 
from pathlib import Path
import argparse
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import from_numpy
from datetime import datetime

# TODO: make dataset universal, handle multiple videos (easy)
class CustomDataset(Dataset):
    def __init__(self, frame_array, transform=None, target_transform=None):
        self.frames = frame_array
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        img = self.frames[idx]  # shape: [H, W]
        
        # Convert grayscale -> RGB by repeating across 3 channels
        img_rgb = np.stack([img] * 3, axis=-1)  # shape: [H, W, 3]
        
        img_pil = Image.fromarray(img_rgb.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img_pil)
        return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="keyframe extractor",
        description="Generates keyframes and save them from video(s)"
    )
    parser.add_argument("--video_path", "-v", type=str)
    parser.add_argument("--gpu", type=bool)
    parser.add_argument("--method", "-m", type=str)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    video_path = args.video_path
    device = torch.device("mps" if args.gpu else "cpu")
    video_array = loadVideoArray(video_path)
    method = args.method
    save_path = Path(args.save_path)

    preprocess_custom = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),  # OR use T.Resize with aspect ratio preservation + padding
    T.Normalize(mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225])
    ])
# === features extraction ===
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights= weights)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)  
    print("Model Initialised, Model Summary:") 
    summary(model, (1,3, 224, 224), device=device)
    model.eval()  # important!
    
    
    img_dataset = CustomDataset(video_array, transform=preprocess)
    data_loader = DataLoader(img_dataset, batch_size=32, shuffle=True)
    features = []
    print("The Model is extracting the features...")
    with torch.no_grad():
        for img in tqdm(data_loader, desc=f'Extracting features '):
            img = img.to(device)
            features.append(model.forward(img))

    features = [img.to('cpu') for img in features]
    f = np.concatenate(features)
    features_np = f.mean(axis=(2, 3)) 

    print("PCA running ")
    pca = PCA(n_components = .9)
    features_pca = pca.fit_transform(features_np)

# === clustering ===
    keyframe_indices = None
    if method == "K":
        kmeans = KMeans(n_clusters=500, random_state=42)
        kmeans.fit(features_pca)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features_pca)
        keyframe_indices = sorted(closest_indices)
# === Saving keyframes ===
    if save_path:
        keyframes = video_array[keyframe_indices, ...]
        torch.save({
            'keyframe_idx':keyframe_indices,
            "keyframes": from_numpy(keyframes),
            "method": method
        }, save_path / f"_{datetime.now()}_keyframes.pth")


    


    