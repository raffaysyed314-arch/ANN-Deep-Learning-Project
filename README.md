# 2D Medical Image Classification: Brain Tumor Detection (BraTS)

## Project Overview
This repository contains the end-to-end deep learning pipeline for a 2D medical image classification task, developed for the Artificial Intelligence BS program (6th Semester) at Hazara University. The project investigates the efficacy of custom-built Convolutional Neural Networks (CNNs) versus enterprise-grade Transfer Learning models under severe data constraints.

## Dataset
* **Source:** Medical Segmentation Decathlon (BraTS Task01_BrainTumour).
* **Preprocessing:** 3D NIfTI volumes (`.nii.gz`) were parsed using `nibabel`. A center 2D slice (slice 75) was extracted from each volume, normalized, and converted to a standard RGB PNG format to fulfill the 2D classification requirement.
* **Splitting:** A strict stratified split (80/10/10) with a deterministic seed (42) was applied to ensure reproducible class distributions across PyTorch DataLoaders.

## Pipeline Architecture
The codebase is modular and divided into functional scripts located in the `scripts/` directory.

### Phase 1: Data Preparation
* `download_data.py`: Direct AWS retrieval of the BraTS dataset, bypassing authentication blockers.
* `preprocess_brats.py`: 3D-to-2D volume slicing and normalization.
* `create_dataloaders.py`: Stratified Train/Val/Test split and PyTorch DataLoader instantiation.

### Phase 2: Custom CNNs
* `models/custom_cnns.py`: Defines 3-layer, 4-layer, and 5-layer CNN architectures from scratch.
* `scripts/train_custom.py`: Training loop with learning curve generation.
* `scripts/evaluate_custom.py`: Test set evaluation generating classification reports and confusion matrices.
* *Finding:* Models exhibited high variance and plateaued near 60% accuracy due to data starvation.

### Phase 3: Transfer Learning
* `models/pretrained_cnns.py`: Imports ResNet18, MobileNetV2, and EfficientNet-B0. Backbones are frozen; classification heads are modified for binary output.
* `scripts/train_pretrained.py`: Fine-tuning loop targeting only the unfrozen classification layers.

### Phase 4: Explainable AI (XAI)
* `scripts/run_xai.py`: Implements `captum` LayerGradCam to visualize model attention. 
* *Finding:* The custom `CNN_5Layer` returned all-zero gradients (proof of failure to learn features), while `ResNet18` successfully generated heatmaps indicating learned attention.

## Setup and Execution
To replicate this environment:
1. Clone the repository.
2. Install dependencies: `pip install torch torchvision matplotlib pandas scikit-learn nibabel captum seaborn`
3. Execute the pipeline sequentially:
   ```bash
   python scripts/download_data.py
   python scripts/preprocess_brats.py
   python scripts/create_dataloaders.py
   python scripts/train_custom.py
   python scripts/evaluate_custom.py
   python scripts/train_pretrained.py
   python scripts/run_xai.py
