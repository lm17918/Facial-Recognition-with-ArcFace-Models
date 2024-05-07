import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import dataset_setup, extract_features, model_setup
from utils.utils_plots import plot_results

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./data/identity_dataset/test"
# Path to saved model
save_path = (
    "./weights/facial_identity_classification_transfer_learning_with_ResNet18.pth"
)


test_dataset, test_dataloader = dataset_setup(data_dir)
model = model_setup(device, save_path)
model.eval()


# Initialize lists to store genuine and impostor distances
genuine_distances = []
impostor_distances = []

# Iterate over the test dataset to create pairs
for i, (image, label) in enumerate(test_dataloader):
    if i % 2 == 0:
        # First image in the pair (genuine pair)
        image1 = image
    else:
        # Second image in the pair (genuine pair)
        image2 = image

        # Extract features for genuine pair
        genuine_features1 = extract_features(image1, model, device)
        genuine_features2 = extract_features(image2, model, device)
        genuine_distance = cosine_similarity(genuine_features1, genuine_features2)
        genuine_distances.append(genuine_distance.item())

        # Pair with a random different person's image (impostor pair)
        random_idx = np.random.randint(len(test_dataset))
        impostor_image, _ = test_dataset[random_idx]
        impostor_features = extract_features(impostor_image, model, device)
        impostor_distance = cosine_similarity(genuine_features1, impostor_features)
        impostor_distances.append(impostor_distance.item())
# Calculate FAR and FRR for different thresholds
thresholds = np.logspace(-7, 0, 100)
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

plot_results(
    thresholds, far_values, frr_values, roc_auc, genuine_distances, impostor_distances
)
