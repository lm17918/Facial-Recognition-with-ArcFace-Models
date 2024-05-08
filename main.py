import numpy as np
import torch

from utils.utils import (
    calculate_metrics,
    dataset_setup,
    model_setup,
    one_to_many_comparison,
    one_to_one_comparison,
    OneToOneMetrics,
    OneToManyMetrics,
)
from utils.utils_plots import plot_results


def main():
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Paths and directories
    data_dir = "./data/identity_dataset/test"
    save_path = (
        "./weights/facial_identity_classification_transfer_learning_with_ResNet18.pth"
    )

    # Load model
    model = model_setup(device, save_path)
    model.eval()

    # Prepare test dataset and dataloader
    test_dataloader = dataset_setup(data_dir, model, device)

    # Perform one-to-many comparison
    one_to_many_genuine_distances, one_to_many_impostor_distances = (
        one_to_many_comparison(test_dataloader)
    )

    # Perform one-to-one comparison
    one_to_one_genuine_distances, one_to_one_impostor_distances = one_to_one_comparison(
        test_dataloader
    )

    # Calculate metrics for one-to-one comparison
    thresholds = np.logspace(-7, 0, 10)
    (
        one_to_one_far_values,
        one_to_one_frr_values,
        one_to_one_roc_auc,
        one_to_one_eer_threshold,
        one_to_one_eer,
    ) = calculate_metrics(
        thresholds, one_to_one_impostor_distances, one_to_one_genuine_distances
    )

    # Calculate metrics for one-to-many comparison
    (
        one_to_many_far_values,
        one_to_many_frr_values,
        one_to_many_roc_auc,
        one_to_many_eer_threshold,
        one_to_many_eer,
    ) = calculate_metrics(
        thresholds, one_to_many_impostor_distances, one_to_many_genuine_distances
    )

    one_to_one_metrics = OneToOneMetrics(
        one_to_one_far_values,
        one_to_one_frr_values,
        one_to_one_roc_auc,
        one_to_one_eer_threshold,
        one_to_one_eer,
        one_to_one_impostor_distances,
        one_to_one_genuine_distances,
    )

    # Calculate metrics for one-to-many comparison
    one_to_many_metrics = OneToManyMetrics(
        one_to_many_far_values,
        one_to_many_frr_values,
        one_to_many_roc_auc,
        one_to_many_eer_threshold,
        one_to_many_eer,
        one_to_many_impostor_distances,
        one_to_many_genuine_distances,
    )

    # Plot results
    plot_results(thresholds, one_to_one_metrics, one_to_many_metrics)


if __name__ == "__main__":
    main()
