# Thesis Project: Improving Pose Tracking in Naturalistic Mouse Videos

This repository contains the code developed for my master's thesis  for the Master in Data Science and Society for Tilburg Univeristy (NL) on **methods to improve pose tracking of mice in naturalistic settings**.
The project is structured into two main research areas:

1. **Autoencoders for Denoising** – evaluating deep learning–based denoising models and their effect on downstream pose tracking.
2. **Keyframe Selection for Video Summarization** – selecting informative subsets of frames from long videos to reduce redundancy while preserving behavioral information.

The full thesis is available at in the manuscript folder.

---

## 📂 Repository Structure

### 🔹 Keyframe Selection
Scripts for extracting features from videos and selecting representative frames:
- `fps.py` – Farthest Point Sampling script
- `extraction.py` – extract deep features using **ResNet50**.
- `clustering.py` – perform **K-Means clustering** over extracted features to select *n* representative frames.

### 🔹 Denoising
Deep learning models and training utilities for denoising video frames:
- `models.py` – contains implementations of three denoising models:
  1. **Baseline Autoencoder**
  2. **CBDNet** – replication of [Convolutional Blind Denoising Network (CBDNet)](https://arxiv.org/abs/1807.04686)
  3. **PRIDNet** – replication of [Pyramid Real Image Denoising Network (PRIDNet)](https://arxiv.org/abs/1908.00273)

- `train.py` – training and evaluation loop for all models.
  - Supports **Weights & Biases (W&B) sweeps** for hyperparameter tuning (YAML configs included).
- `empirical_experiment/` – scripts to denoise complete videos and evaluate downstream **tracking performance**.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Thomasbush9/thesis-project.git
cd thesis-project
conda create -n thesis python=3.10
conda activate thesis
pip install -r requirements.txt

## Results

- **Denoising**: CBDNet outperforms the two other networks
- ** Keyframe selection**: K-means select more meaningful frames
```

## Contact

For questions, email me at <mailto:thomasbush52@gmail.com>.

