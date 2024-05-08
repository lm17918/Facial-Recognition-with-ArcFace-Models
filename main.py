import torch
import numpy as np
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
thresholds = np.logspace(-7, 0, 10)
model = model_setup(device, save_path)
model.eval()
test_dataset, test_dataloader = dataset_setup(data_dir, model, device)

one_to_many_genuine_distances, one_to_many_impostor_distances = one_to_many_comparison(
    test_dataloader
)

one_to_one_genuine_distances, one_to_one_impostor_distances = one_to_one_comparison(
    test_dataloader
)


(
    one_to_one_far_values,
    one_to_one_frr_values,
    one_to_one_roc_auc,
    one_to_one_eer_threshold,
    one_to_one_eer,
) = calculate_metrics(
    thresholds, one_to_one_impostor_distances, one_to_one_genuine_distances
)
(
    one_to_many_far_values,
    one_to_many_frr_values,
    one_to_many_roc_auc,
    one_to_many_eer_threshold,
    one_to_many_eer,
) = calculate_metrics(
    thresholds, one_to_many_impostor_distances, one_to_many_genuine_distances
)

plot_results(
    thresholds,
    one_to_one_far_values,
    one_to_one_frr_values,
    one_to_one_roc_auc,
    one_to_one_impostor_distances,
    one_to_one_genuine_distances,
    one_to_one_eer_threshold,
    one_to_one_eer,
    one_to_many_far_values,
    one_to_many_frr_values,
    one_to_many_roc_auc,
    one_to_many_impostor_distances,
    one_to_many_genuine_distances,
    one_to_many_eer_threshold,
    one_to_many_eer,
)
