# Facial Recognition with ArcFace Models

This repository implements a facial recognition system using ArcFace-based models, focusing on training and evaluation with identity classification and comparison metrics.

## Features

- **ArcFace Models**: Implementation of two ArcFace variants, `ResNetArcFace` and `ResNetDreamArcFace`, for robust facial recognition.
- **Training Pipeline**: Train models with custom loss functions, optimize using SGD, and log the training loss for comparison.
- **Evaluation Metrics**: Supports one-to-one and one-to-many facial identity comparison, calculating metrics such as genuine and impostor distances.
- **Dataset Handling**: Automatic dataset preparation, including identity-based image organization and splitting into train/test sets.
  
## Usage

1. **Training**  
   Run the training script using Hydra for easy configuration:
   ```bash
   python train.py
   ```
   The trained models are saved for future use.

2. **Testing**  
   Evaluate model performance using one-to-one and one-to-many comparisons:
   ```bash
   python test.py
   ```
   Results include precision metrics and visual plots.

## Dataset

- Utilizes the **CelebA-HQ** dataset. Images are organized by identity and split into train/test sets.

