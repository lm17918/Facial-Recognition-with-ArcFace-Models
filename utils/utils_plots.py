import matplotlib.pyplot as plt


def plot_results(
    thresholds, far_values, frr_values, roc_auc, genuine_distances, impostor_distances
):
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
        genuine_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="blue",
        label="Genuine",
    )

    # Plot histogram for impostor distances
    plt.hist(
        impostor_distances,
        bins=50,
        density=True,
        alpha=0.5,
        color="red",
        label="Impostor",
    )

    # Set labels and title
    plt.xlabel("Distance")
    plt.ylabel("Percentage of Samples")
    plt.title("Distance Distribution - System 1")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
