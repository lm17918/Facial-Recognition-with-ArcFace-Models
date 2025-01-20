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

Here's a more detailed README for your project to guide users on how to prepare the dataset and run the necessary code:

---

# Dataset Preparation for CelebA-HQ Project

This project uses the **CelebA-HQ** dataset, which contains images of celebrity faces with various annotations. The dataset is split into different categories based on attributes, poses, and other face features.

## Dataset Information

For more information about the CelebA-HQ dataset, please refer to the [project website](https://github.com/switchablenorms/CelebAMask-HQ).

## Step-by-Step Guide

### 1. Download the Dataset

To use the dataset in this project, first, download and extract it using the following link:

- [CelebAMask-HQ Dataset on GitHub](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file)

Make sure to place the extracted dataset in a location you can easily access, as the paths to the files in the script need to be specified correctly.

### 2. File Structure

After extracting the dataset, you should have the following directory structure:

```
CelebAMask-HQ/
    ├── CelebAMask-HQ-attribute-anno.txt
    ├── CelebAMask-HQ-pose-anno.txt
    ├── images/
        ├── img_0001.jpg
        ├── img_0002.jpg
        └── ...
```

Ensure that both `CelebAMask-HQ-attribute-anno.txt` and `CelebAMask-HQ-pose-anno.txt` are in the root folder along with the `images` subfolder.

### 3. Preprocess the Dataset

Once the dataset is downloaded and extracted, the next step is to preprocess the annotations and split the data into train, validation, and test sets.

#### Running the Preprocessing Script

To preprocess the dataset, simply run the `preprocess_data.py` script as follows:

```bash
python preprocess_data.py
```

This script will:

- Load and merge the attribute annotations (`CelebAMask-HQ-attribute-anno.txt`) and pose annotations (`CelebAMask-HQ-pose-anno.txt`) based on the `Filename` column.
- Split the dataset into three sets: **train**, **validation**, and **test**.
- Save the final, preprocessed dataset as `preprocessed_CelebA.csv` in the current directory.

### 4. Dataset Split

The preprocessing script will automatically split the data as follows:

- 80% of the data will be assigned to the **train** set.
- 10% of the data will be assigned to the **validation** set.
- 10% of the data will be assigned to the **test** set.

This split ensures that you have data for training, validation, and testing models based on face attributes and poses.

### 5. Next Steps

Once the dataset is preprocessed and the `preprocessed_CelebA.csv` file is generated, you can move on to the next steps of your project, which may involve training a model, performing analysis, or any other task requiring the dataset.
