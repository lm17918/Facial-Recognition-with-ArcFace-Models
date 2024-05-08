from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import tqdm


def dataset_setup(data_dir, model, device):
    # Define transforms for test dataset
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    test_dataset = datasets.ImageFolder(data_dir, transforms_test)

    # Create a new dataset with preprocessed images
    preprocessed_data = []
    i = 0
    for image, label in tqdm.tqdm(test_dataset):
        if i > 400:
            break
        i += 1
        preprocessed_image = _extract_features(image, model, device)
        preprocessed_data.append((preprocessed_image, label))

    # Create a DataLoader for the preprocessed dataset
    test_dataloader = torch.utils.data.DataLoader(
        preprocessed_data, batch_size=1, shuffle=False
    )

    return test_dataloader


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
def _extract_features(image, model, device):
    # Add batch dimension if the image tensor doesn't have it
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image = image.to(device)
        features = model(image)
        return features.cpu().numpy()


def calculate_metrics(thresholds, impostor_distances, genuine_distances):
    # Calculate FAR and FRR for different thresholds
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
    return far_values, frr_values, roc_auc, eer_threshold, eer


def one_to_many_comparison(test_dataloader):
    # Initialize lists to store genuine and impostor distances
    genuine_distances = []
    impostor_distances = []
    label_images = defaultdict(list)

    # Iterate through the dataset
    for images, labels in test_dataloader:
        for img, label in zip(images, labels):
            label_images[label.item()].append(img)

    all_labels = sorted(label_images.keys())

    for label in tqdm.tqdm(all_labels, leave=True, desc="General loop one to many"):
        len_imgs = len(label_images[label])
        for i in range(len_imgs):
            single_feature = label_images[label][i]
            similarity_list = []
            for j in tqdm.tqdm(range(len_imgs), leave=False, desc="Genuine distance"):
                if i != j:  # Skip comparing the image with itself
                    feature = label_images[label][j]
                    similarity_list.append(cosine_similarity(single_feature, feature))
            genuine_distances.append(np.average(similarity_list))

            for other_label in tqdm.tqdm(
                all_labels, leave=False, desc="Impostor distance"
            ):
                if other_label != label:
                    similarity_list = []
                    for img in label_images[other_label]:
                        feature = img
                        similarity_list.append(
                            cosine_similarity(single_feature, feature)
                        )
                    impostor_distances.append(np.average(similarity_list))

    return genuine_distances, impostor_distances


def one_to_one_comparison(test_dataloader):
    # Initialize lists to store genuine and impostor distances
    genuine_distances = []
    impostor_distances = []

    # Prepare label-wise image dictionary
    label_images = defaultdict(list)

    # Iterate through the dataset and group images by labels
    for images, labels in test_dataloader:
        for img, label in zip(images, labels):
            label_images[label.item()].append(img)

    # Get all unique labels
    all_labels = sorted(label_images.keys())

    # Compute distances for genuine pairs (within the same label)
    for label in tqdm.tqdm(all_labels, leave=True, desc="Genuine Distances one to one"):
        images = label_images[label]
        num_images = len(images)

        for i in range(num_images):
            for j in range(i + 1, num_images):  # Compare each pair (i, j) where i < j
                feature_i = images[i]
                feature_j = images[j]
                similarity = cosine_similarity(feature_i, feature_j)
                genuine_distances.append(similarity.item())

    # Compute distances for impostor pairs (across different labels)
    for i in tqdm.tqdm(
        range(len(all_labels)), leave=True, desc="Impostor Distances one to one"
    ):
        for j in range(i + 1, len(all_labels)):  # Compare each pair (i, j) where i < j
            label_i = all_labels[i]
            label_j = all_labels[j]

            images_i = label_images[label_i]
            images_j = label_images[label_j]

            for img_i in images_i:
                for img_j in images_j:
                    similarity = cosine_similarity(img_i, img_j)
                    impostor_distances.append(similarity.item())

    return genuine_distances, impostor_distances
