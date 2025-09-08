# DS216 Assignment 02: Image Classification & Segmentation 

This repository hosts the solution submitted for the MLDL-I: Assignment 02 Kaggle competition (DS216-MLDS2025), which challenged students to build classification and segmentation models using histopathology images

# Part 1: Principal Component Analysis (Scratch Implementation)

This section implements **Principal Component Analysis (PCA)** from scratch using both **Eigen decomposition** and **Singular Value Decomposition (SVD)**. The goal is to reduce dimensionality of histopathology images while retaining >90% variance.

## Steps

### 1. Resize Images
- Original images: **256 × 256 × 3**.  
- Resized to **64 × 64 × 3** using `skimage.transform.resize` with anti-aliasing.
<img width="795" height="434" alt="image" src="https://github.com/user-attachments/assets/f50394ef-4110-4b04-b752-d0e90502376e" />


### 2. PCA Implementation
- Implemented a custom `PCA` class with two methods:
  - **Eigen decomposition** of covariance matrix.
  - **Singular Value Decomposition (SVD)** (faster and numerically stable).
- Supports:
  - `fit()`: learn principal components.  
  - `transform()`: project data onto PCs.  
  - `inverse_transform()`: reconstruct data.  

### 3. Channel-wise PCA
- Split RGB channels: **Red, Green, Blue**.  
- Applied PCA separately to each channel.  
- Compared **Eigen vs. SVD** → SVD proved faster and more stable.  

### 4. Grayscale PCA
- Converted RGB images to grayscale.  
- Applied PCA independently.  

### 5. Explained Variance Analysis
- Computed cumulative variance explained by PCs.  
- Found that **~200 components retain ~95% variance**.  
- Reduced dataset from **909 × 64 × 64 × 3** → **909 × 15 × 15 × 3**.
<img width="1989" height="490" alt="image" src="https://github.com/user-attachments/assets/d1f94942-a2fa-46a2-a4e9-56eef9d8d839" />
 

### 6. Reconstruction
- Visualized reconstructed images after PCA.  
- Confirmed minimal quality loss with 225 components.  
<img width="795" height="434" alt="image" src="https://github.com/user-attachments/assets/d67d4691-54bc-424e-b0c8-2508477ce312" />


# Part 2 

## 1. Random Forest Classifier

Extracted flattened image patches resized by factor 1/4.

Applied 5×3 Repeated K-Fold cross-validation.

Tuned parameters: n_estimators=25, max_depth=5, max_features='sqrt'.

Achieved average validation accuracy ≈ 77.6% (baseline).

## 2. CNN from Scratch (Low & High Complexity)

Two CNN architectures: simple one vs. deeper variant with batch normalization and pooling.

Included data augmentation (rotation, shifts, shear, zoom, horizontal flip) with ImageDataGenerator.

Validation accuracies ranged ~76–84%.

## 3. Transfer Learning (ResNet18)

Loaded ResNet-18 pretrained on ImageNet.

Froze backbone, replaced final layer for binary output.

Fine-tuned with Adam optimizer and learning scheduler.

Achieved top performance: ~92.3% validation accuracy.

## 4. Semantic Segmentation (DeepLabV3+)

Utilized segmentation_models_pytorch with resnet34 encoder.

Loss = combined Dice + BCE.

Trained for 10 epochs, evaluated using Dice score.

Test output masks encoded via RLE for Kaggle-style submissions.
