from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
import numpy as np


def dataset_setup(data_dir):
    # Define transforms for test dataset
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Path to your dataset
    data_dir = "./data/identity_dataset/test"

    # Load test dataset
    test_dataset = datasets.ImageFolder(data_dir, transforms_test)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )
    return test_dataset, test_dataloader


def model_setup(device, save_path):
    # Load pretrained ResNet18 model and replace the final fully connected layer
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 307)

    # Load model weights from saved checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Model loaded successfully.")

    # Move model to device
    model.to(device)
    return model


# Function to extract features from a single image
def extract_features(image, model, device):
    # Add batch dimension if the image tensor doesn't have it
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image = image.to(device)
        features = model(image)
        return features.cpu().numpy()


def calculate_metrics(impostor_distances, genuine_distances):
    # Calculate FAR and FRR for different thresholds
    thresholds = np.logspace(-9, 0, 50)
    far_values = []
    frr_values = []

    for threshold in thresholds:
        false_accepts = sum(1 for d in impostor_distances if d < threshold)
        far = false_accepts / len(impostor_distances)

        false_rejects = sum(1 for d in genuine_distances if d >= threshold)
        frr = false_rejects / len(genuine_distances)

        far_values.append(far)
        frr_values.append(frr)

    # Calculate Equal Error Rate (EER)
    eer_threshold = thresholds[
        np.nanargmin(np.abs(np.array(far_values) - np.array(frr_values)))
    ]
    eer = (
        far_values[np.nanargmin(np.abs(np.array(far_values) - np.array(frr_values)))]
        + frr_values[np.nanargmin(np.abs(np.array(far_values) - np.array(frr_values)))]
    ) / 2.0

    print(f"Equal Error Rate (EER): {eer:.4f} at threshold: {eer_threshold:.4e}")

    # Calculate ROC curve
    total_impostor_pairs = len(impostor_distances)
    total_genuine_pairs = len(genuine_distances)
    fpr, tpr, _ = roc_curve(
        [0] * total_impostor_pairs + [1] * total_genuine_pairs,
        impostor_distances + genuine_distances,
    )
    roc_auc = auc(fpr, tpr)
    return thresholds, far_values, frr_values, roc_auc, eer_threshold, eer
