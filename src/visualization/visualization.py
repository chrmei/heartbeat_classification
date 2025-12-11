"""
Heartbeat visualization utilities for ECG data analysis.

This module provides functions to visualize individual heartbeats from the MIT-BIH
and PTBDB datasets, where each row represents one heartbeat with 187 ECG signal samples.
"""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
import seaborn as sns
from typing import Union, List, Optional, Tuple, Any, Dict
import pandas as pd
import os


def plot_heartbeat(
    heartbeat_data: Union[np.ndarray, List[float], pd.Series],
    sample_rate: float = 360.0,
    title: Union[str, None] = None,
    figsize: Tuple[int, int] = (12, 6),
    color: str = "blue",
    linewidth: float = 1.5,
    grid: bool = True,
    show_peaks: bool = False,
    peak_threshold: float = 0.5,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a single heartbeat from ECG data.

    Parameters
    ----------
    heartbeat_data : array-like
        ECG signal data for one heartbeat. Should contain 187 samples for standard format.
    sample_rate : float, default=360.0
        Sampling rate of the ECG signal in Hz.
    title : str, default="ECG Heartbeat"
        Title for the plot.
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches.
    color : str, default="blue"
        Color of the ECG line.
    linewidth : float, default=1.5
        Width of the ECG line.
    grid : bool, default=True
        Whether to show grid.
    show_peaks : bool, default=False
        Whether to highlight R-peaks in the signal.
    peak_threshold : float, default=0.5
        Threshold for peak detection (fraction of max value).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> import numpy as np
    >>> from src.visualization.visualization import plot_heartbeat
    >>>
    >>> # Load a heartbeat from dataset
    >>> heartbeat = np.random.randn(187)  # Example data
    >>> fig = plot_heartbeat(heartbeat, title="Sample Heartbeat")
    >>> plt.show()
    """
    # Convert to numpy array if needed
    if isinstance(heartbeat_data, pd.Series):
        heartbeat_data = heartbeat_data.values
    elif isinstance(heartbeat_data, list):
        heartbeat_data = np.array(heartbeat_data)

    # Ensure we have the right number of samples
    if len(heartbeat_data) != 187:
        print(f"Warning: Expected 187 samples, got {len(heartbeat_data)}")

    # Create sample axis (0 to len(heartbeat_data))
    samples = np.arange(len(heartbeat_data))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ECG signal
    ax.plot(samples, heartbeat_data, color=color, linewidth=linewidth, label="ECG Signal")

    # Add peak detection if requested
    peaks = None
    if show_peaks:
        peaks = _detect_peaks(heartbeat_data, threshold=peak_threshold)
        if len(peaks) > 0:
            peak_samples = samples[peaks]
            peak_values = heartbeat_data[peaks]
            ax.scatter(
                peak_samples,
                peak_values,
                color="red",
                s=50,
                zorder=5,
                label=f"R-peaks ({len(peaks)})",
            )

    # Customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend with box
    if show_peaks and peaks is not None and len(peaks) > 0:
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    else:
        ax.legend(["ECG Signal"], frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Always show grid with better styling
    ax.grid(True, alpha=0.7, linestyle="-", linewidth=1.0, color="gray")
    ax.set_axisbelow(True)  # Put grid behind other elements

    # Set background colors
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Add complete box styling - make sure all spines are visible and styled
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def _detect_peaks(signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Simple peak detection algorithm for R-peak detection in ECG signals.

    Parameters
    ----------
    signal : array-like
        ECG signal data.
    threshold : float, default=0.5
        Threshold for peak detection (fraction of max value).

    Returns
    -------
    numpy.ndarray
        Indices of detected peaks.
    """
    # Find local maxima
    from scipy.signal import find_peaks

    # Normalize signal
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Find peaks above threshold
    peaks, _ = find_peaks(normalized_signal, height=threshold, distance=10)

    return peaks


def plot_multiple_heartbeats(
    data: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
    sample_rate: float = 360.0,
    title: Union[str, None] = None,
    figsize: Tuple[int, int] = (15, 10),
    color: str = "blue",
    linewidth: float = 1.5,
    grid: bool = True,
    show_peaks: bool = False,
    peak_threshold: float = 0.5,
    save_path: Optional[str] = None,
    dpi: int = 300,
    has_label: bool = True,
    max_plots_per_row: int = 3,
) -> plt.Figure:
    """
    Plot multiple heartbeats from ECG data in a subplot layout.

    Parameters
    ----------
    data : pandas.DataFrame, numpy.ndarray, or list of arrays
        ECG data where each row represents one heartbeat. For DataFrame,
        last column is treated as label if has_label=True.
    sample_rate : float, default=360.0
        Sampling rate of the ECG signal in Hz.
    title : str, default="Multiple ECG Heartbeats"
        Title for the entire figure.
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches.
    color : str, default="blue"
        Color of the ECG lines.
    linewidth : float, default=1.5
        Width of the ECG lines.
    grid : bool, default=True
        Whether to show grid on each subplot.
    show_peaks : bool, default=False
        Whether to highlight R-peaks in the signals.
    peak_threshold : float, default=0.5
        Threshold for peak detection (fraction of max value).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, default=300
        Resolution for saved figure.
    has_label : bool, default=True
        Whether the last column contains class labels (for DataFrame input).
    max_plots_per_row : int, default=3
        Maximum number of plots per row in the subplot layout.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object with subplots.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from src.visualization.visualization import plot_multiple_heartbeats
    >>>
    >>> # Load data from CSV
    >>> df = pd.read_csv('mitbih_train.csv', header=None)
    >>> fig = plot_multiple_heartbeats(df.head(6), title="Sample Heartbeats")
    >>> plt.show()
    """
    # Convert data to list of arrays
    heartbeat_list = []
    labels = []

    if isinstance(data, pd.DataFrame):
        for idx, row in data.iterrows():
            if has_label:
                heartbeat_data = row.iloc[:-1].values
                label = int(row.iloc[-1])
            else:
                heartbeat_data = row.values
                label = None
            heartbeat_list.append(heartbeat_data)
            labels.append(label)
    elif isinstance(data, np.ndarray):
        # Assume each row is a heartbeat
        for i in range(data.shape[0]):
            heartbeat_list.append(data[i])
            labels.append(None)
    elif isinstance(data, list):
        heartbeat_list = data
        labels = [None] * len(data)
    else:
        raise ValueError("Data must be a pandas DataFrame, numpy array, or list of arrays")

    n_heartbeats = len(heartbeat_list)
    if n_heartbeats == 0:
        raise ValueError("No heartbeats found in the data")

    # Calculate subplot layout
    n_cols = min(max_plots_per_row, n_heartbeats)
    n_rows = (n_heartbeats + n_cols - 1) // n_cols  # Ceiling division

    # Adjust figure size based on number of subplots
    fig_width = figsize[0]
    fig_height = figsize[1] * (n_rows / 2)  # Scale height with number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle single subplot case
    if n_heartbeats == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()

    # Plot each heartbeat
    for i, (heartbeat_data, label) in enumerate(zip(heartbeat_list, labels)):
        ax = axes[i]

        # Convert to numpy array if needed
        if isinstance(heartbeat_data, pd.Series):
            heartbeat_data = heartbeat_data.values
        elif isinstance(heartbeat_data, list):
            heartbeat_data = np.array(heartbeat_data)

        # Ensure we have the right number of samples
        if len(heartbeat_data) != 187:
            print(f"Warning: Heartbeat {i} expected 187 samples, " "got {len(heartbeat_data)}")

        # Create sample axis
        samples = np.arange(len(heartbeat_data))

        # Plot ECG signal
        ax.plot(
            samples,
            heartbeat_data,
            color=color,
            linewidth=linewidth,
            label="ECG Signal",
        )

        # Add peak detection if requested
        peaks = None
        if show_peaks:
            peaks = _detect_peaks(heartbeat_data, threshold=peak_threshold)
            if len(peaks) > 0:
                peak_samples = samples[peaks]
                peak_values = heartbeat_data[peaks]
                ax.scatter(
                    peak_samples,
                    peak_values,
                    color="red",
                    s=30,
                    zorder=5,
                    label=f"R-peaks ({len(peaks)})",
                )

        # Customize subplot
        subplot_title = f"Heartbeat {i+1}"
        if label is not None:
            subplot_title += f" (Class: {label})"

        ax.set_title(subplot_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude (mV)")

        # Add legend for this subplot
        if show_peaks and peaks is not None and len(peaks) > 0:
            ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=8)
        else:
            ax.legend(
                ["ECG Signal"],
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=8,
            )

        # Grid styling
        if grid:
            ax.grid(True, alpha=0.7, linestyle="-", linewidth=0.5, color="gray")
            ax.set_axisbelow(True)

        # Set background colors
        ax.set_facecolor("white")

        # Add complete box styling
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    # Hide unused subplots
    for i in range(n_heartbeats, len(axes)):
        axes[i].set_visible(False)

    # Set overall figure title
    if title is None:
        title = f"Multiple ECG Heartbeats ({n_heartbeats} samples)"

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("white")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def load_heartbeat_from_csv(
    csv_path: str, row_index: int = 0, has_label: bool = True
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Load a single heartbeat from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    row_index : int, default=0
        Index of the row to load.
    has_label : bool, default=True
        Whether the last column contains the class label.

    Returns
    -------
    tuple
        (heartbeat_data, label) where heartbeat_data is the ECG signal and
        label is the class label (None if has_label=False).
    """
    # Read CSV
    data = pd.read_csv(csv_path, header=None)

    # Get the specified row
    row = data.iloc[row_index]

    if has_label:
        # Last column is the label
        heartbeat_data = row.iloc[:-1].values
        label = int(row.iloc[-1])
    else:
        # All columns are signal data
        heartbeat_data = row.values
        label = None

    return heartbeat_data, label


def demo_heartbeat_visualization() -> None:
    """
    Demo function to showcase the visualization capabilities.
    """
    print("Heartbeat Visualization Demo")
    print("=" * 40)

    # Create sample data for demonstration
    np.random.seed(42)
    sample_heartbeats = []
    for i in range(6):
        # Generate synthetic ECG-like data
        t = np.linspace(0, 1, 187)
        heartbeat = np.sin(2 * np.pi * 1.2 * t) * np.exp(-t * 2) + 0.1 * np.random.randn(187)
        sample_heartbeats.append(heartbeat)

    # Create DataFrame with labels
    df = pd.DataFrame(sample_heartbeats)
    df["label"] = [0, 1, 0, 2, 1, 0]  # Sample labels

    print("1. Single heartbeat visualization:")
    plot_heartbeat(sample_heartbeats[0], title="Sample Single Heartbeat")
    plt.show()

    print("2. Multiple heartbeats visualization:")
    plot_multiple_heartbeats(df, title="Sample Multiple Heartbeats", show_peaks=True)
    plt.show()

    print("Demo completed!")


# Function to plot and save validation accuracy and validation loss over epochs from history
def plot_training_history(history: Any, save_dir: str, prefix: str) -> None:
    hist = history.history
    metrics = [m for m in hist.keys() if not m.startswith("val_")]

    # Create the output folder if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for m in metrics:
        plt.figure()
        plt.plot(hist[m], label=f"Train {m}")
        if f"val_{m}" in hist:
            plt.plot(hist[f"val_{m}"], label=f"Val {m}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(f"{m} over epochs")
        plt.legend()
        plt.grid(True)

        # Construct filename with prefix and filepath with directory and filename
        filename = f"{prefix}_{m}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the figure
        plt.savefig(filepath, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.show()


def save_cv_diagnostics(
    cv_df: pd.DataFrame, model_name: str, sampling_method: str, results_path: str
) -> None:
    """Create CV tradeoff, metric spread, and learning-curve plots.

    Checks Cross-Validation behaviour: stability or overfitting?
    3 Plots:
        Trade-off plot: Balanced Accuracy vs. F1-macro -> Pareto-frontier of models (accuracy vs generalization speed)
        Boxplot: Distribution of cross-fold F1 and balanced accuracy: stability and variance across parameter combinations
        Learning-Curve: Mean train F1 vs mean validation F1 across all parameter sets -> bias-variance: do training scores soar while validation lags?

    """
    base = results_path.replace(".csv", "")
    os.makedirs(os.path.dirname(base), exist_ok=True)

    # Tradeoff plot: Balanced Accuracy vs F1
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=cv_df,
        x="mean_test_bal_acc",
        y="mean_test_f1_macro",
        hue="mean_fit_time",
        palette="viridis",
        s=80,
        ax=ax,
    )
    ax.set_title(f"{model_name} ({sampling_method}) - CV Tradeoff")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("F1 Macro")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{base}_{model_name}_{sampling_method}_cv_tradeoff.png", dpi=250)
    plt.close(fig)

    # Cross-fold metric spread
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=cv_df[["mean_test_f1_macro", "mean_test_bal_acc"]], ax=ax)
    ax.set_title("Cross-Fold Metric Spread")
    ax.set_ylabel("Score")
    plt.tight_layout()
    fig.savefig(f"{base}_{model_name}_{sampling_method}_cv_spread.png", dpi=250)
    plt.close(fig)

    # Learning curve: train vs validation F1
    if "mean_train_f1_macro" in cv_df.columns and "mean_test_f1_macro" in cv_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        train_scores = cv_df["mean_train_f1_macro"].values
        val_scores = cv_df["mean_test_f1_macro"].values
        param_indices = np.arange(len(train_scores)) + 1
        ax.plot(param_indices, train_scores, marker="o", label="Train F1")
        ax.plot(param_indices, val_scores, marker="x", label="Validation F1")
        ax.set_title(
            f"Learning Curve - Training vs Validation F1 (per param set)\n{model_name} {sampling_method}"
        )
        ax.set_xlabel("Parameter combination")
        ax.set_ylabel("F1 Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{base}_{model_name}_{sampling_method}__learning_curve_train_val.png", dpi=300)
        plt.close(fig)


def save_overfit_diagnostic(
    cv_df: pd.DataFrame, model_name: str, sampling_method: str, results_path: str
) -> None:
    """visualize overfitting patterns.

    Scatter plot of mean-fit-time per paramter conbination vs train-validation F1-gap

    Shows how training-time complexity relates to overfitting. If points are
    mostly near y=0 model generalizes well. Large positive gaps = overfitting!

    """
    if "mean_train_f1_macro" not in cv_df or "mean_test_f1_macro" not in cv_df:
        return
    df = cv_df.copy()
    df["train_val_gap"] = df["mean_train_f1_macro"] - df["mean_test_f1_macro"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="mean_fit_time",
        y="train_val_gap",
        hue="mean_test_f1_macro",
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title(f"{model_name} - Overfitting Diagnostic")
    ax.set_xlabel("Mean Fit Time (s)")
    ax.set_ylabel("Train-Validation F1 Gap")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    base = results_path.replace(".csv", "")
    fig.savefig(f"{base}_{model_name}_{sampling_method}_overfit_diag.png", dpi=250)
    plt.close(fig)


def save_model_diagnostics(
    eval_results: Dict[str, Any], model_name: str, sampling_method: str, results_path: str
) -> None:
    """Confusion matrix and overfitting visualization.

    Confusion Matrix for final chosen, best model

    """
    cm = eval_results["test"]["confusion_matrix"]
    labels = eval_results["labels"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    path = results_path.replace(".csv", f"_{model_name}_{sampling_method}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve_binary(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
) -> None:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"{model_name}: No probability or decision function — skipping ROC.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()


def plot_roc_curve_multiclass(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
) -> None:
    # Binarize labels for one-vs-rest ROC
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"{model_name}: No probability or decision function — skipping ROC.")
        return

    plt.figure(figsize=(7, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkgreen", "red"])
    for i, color in zip(range(len(classes)), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC Curves — {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()


def save_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    sampling_method: str,
    results_path: str,
) -> None:
    try:
        if len(np.unique(y_test)) == 2:
            plot_roc_curve_binary(model, X_test, y_test, model_name)
        else:
            plot_roc_curve_multiclass(model, X_test, y_test, model_name)
        plt.savefig(
            results_path.replace(".csv", f"_{model_name}_{sampling_method}_roc_curve.png"), dpi=250
        )
        plt.close()
    except Exception as e:
        print(f"Skipping ROC curve for {model_name}: {e}")


if __name__ == "__main__":
    # %%
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x))
    plt.show()

    demo_heartbeat_visualization()
