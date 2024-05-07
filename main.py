import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Load pretrained ResNet18 model and replace the final fully connected layer
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 307)

# Path to saved model
save_path = (
    "./weights/facial_identity_classification_transfer_learning_with_ResNet18.pth"
)

# Load model weights from saved checkpoint
model.load_state_dict(torch.load(save_path, map_location=device))
print("Model loaded successfully.")

# Move model to device
model.to(device)
model.eval()


# Function to extract features from a single image
def extract_features(image):
    # Add batch dimension if the image tensor doesn't have it
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image = image.to(device)
        features = model(image)
        return features.cpu().numpy()


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
        genuine_features1 = extract_features(image1)
        genuine_features2 = extract_features(image2)
        genuine_distance = cosine_similarity(genuine_features1, genuine_features2)
        genuine_distances.append(genuine_distance.item())

        # Pair with a random different person's image (impostor pair)
        random_idx = np.random.randint(len(test_dataset))
        impostor_image, _ = test_dataset[random_idx]
        impostor_features = extract_features(impostor_image)
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

# Plot FAR and FRR vs. Threshold
plt.figure(figsize=(12, 6))

# Plot FAR vs. distance threshold
plt.subplot(1, 2, 1)
plt.plot(thresholds, far_values, color="blue", label="FAR")
plt.xscale("log")
plt.xlabel("Distance Threshold")
plt.ylabel("False Acceptance Rate (FAR)")
plt.title("FAR vs. Distance Threshold")
plt.legend()

# Plot FRR vs. distance threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds, frr_values, color="red", label="FRR")
plt.xscale("log")
plt.xlabel("Distance Threshold")
plt.ylabel("False Rejection Rate (FRR)")
plt.title("FRR vs. Distance Threshold")
plt.legend()

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(
    frr_values,
    far_values,
    color="darkorange",
    lw=2,
    label=f"ROC curve (AUC = {roc_auc:.2f})",
)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Rejection Rate (FRR)")
plt.ylabel("False Acceptance Rate (FAR)")
plt.xscale("log")
plt.yscale("log")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
# Plot Distance distributions for System 1
plt.figure(figsize=(8, 6))

# Plot histogram for genuine distances
plt.hist(
    genuine_distances, bins=50, density=True, alpha=0.5, color="blue", label="Genuine"
)

# Plot histogram for impostor distances
plt.hist(
    impostor_distances, bins=50, density=True, alpha=0.5, color="red", label="Impostor"
)

# Set labels and title
plt.xlabel("Distance")
plt.ylabel("Percentage of Samples")
plt.title("Distance Distribution - System 1")

# Add legend
plt.legend()

# Show the plot
plt.show()
