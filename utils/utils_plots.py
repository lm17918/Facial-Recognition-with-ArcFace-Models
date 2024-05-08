import matplotlib.pyplot as plt


def plot_results(
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
):
    # Create a single figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))

    # Plot FAR vs. distance threshold for both scenarios
    axs[0, 0].plot(
        thresholds, one_to_one_far_values, color="blue", label="FAR (One-to-One)"
    )
    axs[0, 0].plot(
        thresholds,
        one_to_many_far_values,
        color="green",
        linestyle="--",
        label="FAR (One-to-Many)",
    )
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_xlabel("Distance Threshold")
    axs[0, 0].set_ylabel("False Acceptance Rate (FAR)")
    axs[0, 0].set_title("FAR vs. Distance Threshold")
    axs[0, 0].legend()

    # Plot FRR vs. distance threshold for both scenarios
    axs[0, 1].plot(
        thresholds, one_to_one_frr_values, color="red", label="FRR (One-to-One)"
    )
    axs[0, 1].plot(
        thresholds,
        one_to_many_frr_values,
        color="orange",
        linestyle="--",
        label="FRR (One-to-Many)",
    )
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_xlabel("Distance Threshold")
    axs[0, 1].set_ylabel("False Rejection Rate (FRR)")
    axs[0, 1].set_title("FRR vs. Distance Threshold")
    axs[0, 1].legend()

    # Plot ROC curve for both scenarios
    axs[1, 0].plot(
        one_to_one_frr_values,
        one_to_one_far_values,
        color="darkorange",
        lw=2,
        label=f"ROC curve (One-to-One AUC = {one_to_one_roc_auc:.2f})",
    )
    axs[1, 0].plot(
        one_to_many_frr_values,
        one_to_many_far_values,
        color="green",
        lw=2,
        linestyle="--",
        label=f"ROC curve (One-to-Many AUC = {one_to_many_roc_auc:.2f})",
    )
    axs[1, 0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axs[1, 0].set_xlabel("False Rejection Rate (FRR)")
    axs[1, 0].set_ylabel("False Acceptance Rate (FAR)")
    # axs[1, 0].set_xscale("log")
    # axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axs[1, 0].legend(loc="lower right")
    axs[1, 0].grid(True)

    # Plot Distance distributions for System 1 (One-to-One)
    axs[1, 1].hist(
        one_to_one_genuine_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="blue",
        label="Genuine (One-to-One)",
    )
    axs[1, 1].hist(
        one_to_one_impostor_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="red",
        label="Impostor (One-to-One)",
    )
    axs[1, 1].set_xlabel("Distance")
    axs[1, 1].set_ylabel("Percentage of Samples")
    axs[1, 1].set_title("Distance Distribution - One-to-One System")
    axs[1, 1].legend()

    # Plot Distance distributions for System 2 (One-to-Many)
    axs[2, 1].hist(
        one_to_many_genuine_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="green",
        label="Genuine (One-to-Many)",
    )
    axs[2, 1].hist(
        one_to_many_impostor_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="orange",
        label="Impostor (One-to-Many)",
    )
    axs[2, 1].set_xlabel("Distance")
    axs[2, 1].set_ylabel("Percentage of Samples")
    axs[2, 1].set_title("Distance Distribution - One-to-Many System")
    axs[2, 1].legend()

    # Plot EER vs. distance threshold for both scenarios
    axs[2, 0].plot(
        thresholds, one_to_one_far_values, color="blue", label="FAR (One-to-One)"
    )
    axs[2, 0].plot(
        thresholds, one_to_one_frr_values, color="red", label="FRR (One-to-One)"
    )
    axs[2, 0].plot(
        [one_to_one_eer_threshold],
        [one_to_one_eer],
        marker="o",
        markersize=8,
        color="green",
        label=f"EER (One-to-One) = {one_to_one_eer:.2f}",
    )
    axs[2, 0].axvline(
        x=one_to_one_eer_threshold,
        color="gray",
        linestyle="--",
        label=f"EER Threshold (One-to-One): {one_to_one_eer_threshold:.2f}",
    )
    axs[2, 0].plot(
        thresholds,
        one_to_many_far_values,
        color="purple",
        linestyle="--",
        label="FAR (One-to-Many)",
    )
    axs[2, 0].plot(
        thresholds,
        one_to_many_frr_values,
        color="orange",
        linestyle="--",
        label="FRR (One-to-Many)",
    )
    axs[2, 0].plot(
        [one_to_many_eer_threshold],
        [one_to_many_eer],
        marker="o",
        markersize=8,
        color="brown",
        label=f"EER (One-to-Many) = {one_to_many_eer:.2f}",
    )
    axs[2, 0].axvline(
        x=one_to_many_eer_threshold,
        color="black",
        linestyle="--",
        label=f"EER Threshold (One-to-Many): {one_to_many_eer_threshold:.2f}",
    )
    axs[2, 0].set_xscale("log")
    axs[2, 0].set_xlabel("Distance Threshold")
    axs[2, 0].set_ylabel("Error Rates")
    axs[2, 0].set_title("EER vs. Distance Threshold")
    axs[2, 0].legend()

    # Adjust layout
    # plt.tight_layout()

    # Show the combined plot
    plt.show()
