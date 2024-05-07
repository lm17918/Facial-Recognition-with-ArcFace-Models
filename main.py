import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import dataset_setup, extract_features, model_setup, calculate_metrics
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

thresholds, far_values, frr_values, roc_auc, eer_threshold, eer = calculate_metrics(
    impostor_distances, genuine_distances
)

plot_results(
    thresholds,
    far_values,
    frr_values,
    roc_auc,
    genuine_distances,
    impostor_distances,
    eer_threshold,
    eer,
)
