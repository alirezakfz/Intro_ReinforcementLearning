import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    """Plots the learning curve, combining epsilon and score on a single figure.

    Args:
        x (list): List of training steps.
        scores (list): List of scores at each training step.
        epsilons (list): List of epsilon values at each training step.
        filename (str): Filename to save the plot.
        lines (list, optional): List of additional lines to plot (default: None).
    """

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Create a 1x2 subplot grid

    # Plot epsilon on ax
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    # Calculate and plot running average score on ax2
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.set_xlabel('Training Steps')  # Add missing x-label for ax2
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    # Optional: Add additional lines (if provided)
    if lines is not None:
        for line in lines:
            ax.plot(line[0], line[1], label=line[2])
            ax2.plot(line[0], line[1], label=line[2])  # Plot lines on both axes

    # Customize plot title and legend (optional)
    plt.suptitle(f"Learning Curve - Epsilon and Score", fontsize=14)
    if lines is not None:
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to avoid memory leaks

if __name__ == "__main__":
    # Example usage (replace with your actual data)
    x = range(100)
    scores = [0.7, 0.75, 0.8, ...]
    epsilons = [0.1, 0.08, 0.05, ...]
    filename = "learning_curve.png"
    plot_learning_curve(x, scores, epsilons, filename)