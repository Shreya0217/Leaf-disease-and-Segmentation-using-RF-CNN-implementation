This repository hosts the solution submitted for the MLDL-I: Assignment 02 Kaggle competition (DS216-MLDS2025), which challenged students to build classification and segmentation models using histopathology images

🔹 Random Forest Classifier

Extracted flattened image patches resized by factor 1/4.

Applied 5×3 Repeated K-Fold cross-validation.

Tuned parameters: n_estimators=25, max_depth=5, max_features='sqrt'.

Achieved average validation accuracy ≈ 77.6% (baseline).

🔹 CNN from Scratch (Low & High Complexity)

Two CNN architectures: simple one vs. deeper variant with batch normalization and pooling.

Included data augmentation (rotation, shifts, shear, zoom, horizontal flip) with ImageDataGenerator.

Validation accuracies ranged ~76–84%.

🔹 Transfer Learning (ResNet18)

Loaded ResNet-18 pretrained on ImageNet.

Froze backbone, replaced final layer for binary output.

Fine-tuned with Adam optimizer and learning scheduler.

Achieved top performance: ~92.3% validation accuracy.

🔹 Semantic Segmentation (DeepLabV3+)

Utilized segmentation_models_pytorch with resnet34 encoder.

Loss = combined Dice + BCE.

Trained for 10 epochs, evaluated using Dice score.

Test output masks encoded via RLE for Kaggle-style submissions.
