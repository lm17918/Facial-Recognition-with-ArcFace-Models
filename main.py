import numpy as np
import torch
import hydra
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


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg):
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = model_setup(device, cfg.save_path)
    model.eval()

    # Prepare test dataset and dataloader
    test_dataloader = dataset_setup(cfg.data_dir, model, device)

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

    one_to_one_metrics = OneToOneMetrics(
        *calculate_metrics(
            thresholds, one_to_one_impostor_distances, one_to_one_genuine_distances
        ),
        one_to_one_impostor_distances,
        one_to_one_genuine_distances
    )

    # Calculate metrics for one-to-many comparison
    one_to_many_metrics = OneToManyMetrics(
        *calculate_metrics(
            thresholds, one_to_many_impostor_distances, one_to_many_genuine_distances
        ),
        one_to_many_impostor_distances,
        one_to_many_genuine_distances
    )

    # Plot results
    plot_results(thresholds, one_to_one_metrics, one_to_many_metrics)


if __name__ == "__main__":
    main()
