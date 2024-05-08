import torch

from utils.utils import (
    dataset_setup,
    model_setup,
    calculate_metrics,
    one_to_one_comparison,
    one_to_many_comparison,
)
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

genuine_distances, impostor_distances = one_to_many_comparison(
    test_dataloader, model, device, test_dataset
)

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
