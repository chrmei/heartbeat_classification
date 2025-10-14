"""
Confusion matrix visualization utilities.

Provides a clean, simple function to pretty print a confusion matrix
using seaborn/matplotlib.
"""

from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    matrix: Union[np.ndarray, Sequence[Sequence[int]]],
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    fmt: Optional[str] = None,
    colorbar: bool = True,
    title: Optional[str] = None,
    annot: bool = True,
    annot_fontsize: int = 10,
    xtick_rotation: int = 0,
    ytick_rotation: int = 0,
) -> plt.Figure:
    """
    Pretty-print a confusion matrix with seaborn/matplotlib.

    Parameters
    ----------
    matrix : array-like (n_classes, n_classes)
        Confusion matrix counts.
    class_names : list of str, optional
        Class labels for axes. If None, uses numeric indices.
    normalize : {'true', 'pred', 'all', None}, optional
        Apply normalization:
        - 'true': row-wise (per actual class)
        - 'pred': column-wise (per predicted class)
        - 'all': divide by total sum
        - None: no normalization
    figsize : tuple, default (8, 6)
        Figure size in inches.
    cmap : str, default 'Blues'
        Colormap for heatmap.
    fmt : str, optional
        String format for annotations. If None, inferred from normalization.
    colorbar : bool, default True
        Whether to draw colorbar.
    title : str, optional
        Figure title.
    annot : bool, default True
        Whether to annotate each cell.
    annot_fontsize : int, default 10
        Font size for annotations.
    xtick_rotation : int, default 0
        Rotation angle for x tick labels.
    ytick_rotation : int, default 0
        Rotation angle for y tick labels.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    cm = np.asarray(matrix, dtype=float)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    # Determine class names
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("class_names length must match matrix size")

    # Normalization
    if normalize is not None:
        if normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm = cm / row_sums
        elif normalize == "pred":
            col_sums = cm.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            cm = cm / col_sums
        elif normalize == "all":
            total = cm.sum()
            total = 1.0 if total == 0 else total
            cm = cm / total
        else:
            raise ValueError("normalize must be one of {'true','pred','all',None}")

    # Default format
    if fmt is None:
        fmt = ".2f" if normalize else "d"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar=colorbar,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": annot_fontsize},
        ax=ax,
    )

    # Labels and title
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title is None:
        title = "Confusion Matrix" + (" (normalized)" if normalize else "")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Ticks styling
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, va="center")

    # Layout
    fig.tight_layout()

    return fig


__all__ = ["plot_confusion_matrix"]


