import matplotlib.pyplot as plt


def plot_far_vs_threshold(ax, thresholds, one_to_one_metrics, one_to_many_metrics):
    ax.plot(
        thresholds,
        one_to_one_metrics.far_values,
        color="blue",
        label="FAR (One-to-One)",
    )
    ax.plot(
        thresholds,
        one_to_many_metrics.far_values,
        color="green",
        linestyle="--",
        label="FAR (One-to-Many)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("False Acceptance Rate (FAR)")
    ax.set_title("FAR vs. Distance Threshold")
    ax.legend()


def plot_frr_vs_threshold(ax, thresholds, one_to_one_metrics, one_to_many_metrics):
    ax.plot(
        thresholds, one_to_one_metrics.frr_values, color="red", label="FRR (One-to-One)"
    )
    ax.plot(
        thresholds,
        one_to_many_metrics.frr_values,
        color="orange",
        linestyle="--",
        label="FRR (One-to-Many)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("FRR vs. Distance Threshold")
    ax.legend()


def plot_roc_curve(ax, one_to_one_metrics, one_to_many_metrics):
    ax.plot(
        one_to_one_metrics.frr_values,
        one_to_one_metrics.far_values,
        color="darkorange",
        lw=2,
        label=f"ROC curve (One-to-One AUC = {one_to_one_metrics.roc_auc:.2f})",
    )
    ax.plot(
        one_to_many_metrics.frr_values,
        one_to_many_metrics.far_values,
        color="green",
        lw=2,
        linestyle="--",
        label=f"ROC curve (One-to-Many AUC = {one_to_many_metrics.roc_auc:.2f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Rejection Rate (FRR)")
    ax.set_ylabel("False Acceptance Rate (FAR)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True)


def plot_distance_distribution(ax, genuine_distances, impostor_distances, title):
    ax.hist(
        genuine_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="blue",
        label="Genuine",
    )
    ax.hist(
        impostor_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="red",
        label="Impostor",
    )
    ax.set_xlabel("Distance")
    ax.set_ylabel("Percentage of Samples")
    ax.set_title(title)
    ax.legend()


def plot_eer_vs_threshold(ax, thresholds, one_to_one_metrics, one_to_many_metrics):
    ax.plot(
        thresholds,
        one_to_one_metrics.far_values,
        color="blue",
        label="FAR (One-to-One)",
    )
    ax.plot(
        thresholds, one_to_one_metrics.frr_values, color="red", label="FRR (One-to-One)"
    )
    ax.plot(
        [one_to_one_metrics.eer_threshold],
        [one_to_one_metrics.eer],
        marker="o",
        markersize=8,
        color="green",
        label=f"EER (One-to-One) = {one_to_one_metrics.eer:.2f}",
    )
    ax.axvline(
        x=one_to_one_metrics.eer_threshold,
        color="gray",
        linestyle="--",
        label=f"EER Threshold (One-to-One): {one_to_one_metrics.eer_threshold:.2f}",
    )
    ax.plot(
        thresholds,
        one_to_many_metrics.far_values,
        color="purple",
        linestyle="--",
        label="FAR (One-to-Many)",
    )
    ax.plot(
        thresholds,
        one_to_many_metrics.frr_values,
        color="orange",
        linestyle="--",
        label="FRR (One-to-Many)",
    )
    ax.plot(
        [one_to_many_metrics.eer_threshold],
        [one_to_many_metrics.eer],
        marker="o",
        markersize=8,
        color="brown",
        label=f"EER (One-to-Many) = {one_to_many_metrics.eer:.2f}",
    )
    ax.axvline(
        x=one_to_many_metrics.eer_threshold,
        color="black",
        linestyle="--",
        label=f"EER Threshold (One-to-Many): {one_to_many_metrics.eer_threshold:.2f}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("Error Rates")
    ax.set_title("EER vs. Distance Threshold")
    ax.legend()


def plot_results(thresholds, one_to_one_metrics, one_to_many_metrics):
    # Create a single figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))

    plot_far_vs_threshold(
        axs[0, 0], thresholds, one_to_one_metrics, one_to_many_metrics
    )
    plot_frr_vs_threshold(
        axs[0, 1], thresholds, one_to_one_metrics, one_to_many_metrics
    )
    plot_roc_curve(axs[1, 0], one_to_one_metrics, one_to_many_metrics)
    plot_distance_distribution(
        axs[1, 1],
        one_to_one_metrics.genuine_distances,
        one_to_one_metrics.impostor_distances,
        "Distance Distribution - One-to-One System",
    )
    plot_distance_distribution(
        axs[2, 1],
        one_to_many_metrics.genuine_distances,
        one_to_many_metrics.impostor_distances,
        "Distance Distribution - One-to-Many System",
    )
    plot_eer_vs_threshold(
        axs[2, 0], thresholds, one_to_one_metrics, one_to_many_metrics
    )

    # Adjust layout
    plt.tight_layout()

    # Show the combined plot
    plt.show()
