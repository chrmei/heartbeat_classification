import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    normalize=False,
    class_names=None,
    figsize=(8, 6),
    cmap="Blues",
    title="Confusion Matrix",
):
    """
    Plot a confusion matrix with optional normalization.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    normalize : bool, default=False
        If True, normalize the confusion matrix and show absolute values in brackets
    class_names : list, optional
        List of class names for labeling
    figsize : tuple, default=(8, 6)
        Figure size
    cmap : str, default='Blues'
        Colormap for the heatmap
    title : str, default='Confusion Matrix'
        Title for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # Replace NaN with 0 (in case of division by zero)
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    if normalize:
        # Use normalized values for color mapping
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt="",
            cmap=cmap,
            cbar=True,
            ax=ax,
            square=True,
            linewidths=0.5,
        )

        # Add annotations: normalized values on top, absolute values in brackets below
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                normalized_val = cm_normalized[i, j]
                absolute_val = cm[i, j]
                # Format: normalized value on top, absolute value in brackets below
                text = f"{normalized_val:.2f}\n({absolute_val})"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
    else:
        # Just show absolute values
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=True,
            ax=ax,
            square=True,
            linewidths=0.5,
        )

    # Set labels
    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    return fig, ax


__all__ = ["plot_confusion_matrix"]
