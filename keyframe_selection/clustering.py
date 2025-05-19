import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse
import os
from pathlib import Path

def fps(features, num_samples):
    selected = [0]
    for _ in tqdm(range(1, num_samples), desc="Generating keyframes (FPS)"):
        dists = cdist(features[selected], features, metric='euclidean')
        min_dists = dists.min(axis=0)
        next_idx = min_dists.argmax()
        selected.append(next_idx)
    return sorted(selected)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to .npy file with feature vectors")
    parser.add_argument('--output_dir', type=str, default='/Users/thomasbush/Documents/Vault/DSS_Tilburg/data/keyframes_output', help="Directory to save keyframe indices")
    parser.add_argument('--n_keyframes', type=int, default=500, help="Number of keyframes to select")
    args = parser.parse_args()

    data_path = args.data_path
    features = np.load(data_path)
    assert features.ndim == 2, f"Expected 2D array, got {features.shape}"

    # Reduce to 30D for faster FPS
    pca = PCA(n_components=30)
    reduced_f = pca.fit_transform(features)

    indices_fps = fps(reduced_f, args.n_keyframes)

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"keyframes_fps_{args.n_keyframes}.npy"
    np.save(out_path, np.array(indices_fps))

    print(f"Keyframe indices saved to: {out_path}")


