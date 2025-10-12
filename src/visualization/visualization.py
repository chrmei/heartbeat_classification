"""
Heartbeat visualization utilities for ECG data analysis.

This module provides functions to visualize individual heartbeats from the MIT-BIH
and PTBDB datasets, where each row represents one heartbeat with 187 ECG signal samples.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Tuple
import pandas as pd


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
    dpi: int = 300
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
    ax.plot(samples, heartbeat_data, color=color, linewidth=linewidth, label='ECG Signal')
    
    # Add peak detection if requested
    peaks = None
    if show_peaks:
        peaks = _detect_peaks(heartbeat_data, threshold=peak_threshold)
        if len(peaks) > 0:
            peak_samples = samples[peaks]
            peak_values = heartbeat_data[peaks]
            ax.scatter(peak_samples, peak_values, color='red', s=50, zorder=5, 
                      label=f'R-peaks ({len(peaks)})')
    
    # Customize plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend with box
    if show_peaks and peaks is not None and len(peaks) > 0:
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    else:
        ax.legend(['ECG Signal'], frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    # Always show grid with better styling
    ax.grid(True, alpha=0.7, linestyle='-', linewidth=1.0, color='gray')
    ax.set_axisbelow(True)  # Put grid behind other elements
    
    # Set background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add complete box styling - make sure all spines are visible and styled
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
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
    max_plots_per_row: int = 3
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
            print(f"Warning: Heartbeat {i} expected 187 samples, got {len(heartbeat_data)}")
        
        # Create sample axis
        samples = np.arange(len(heartbeat_data))
        
        # Plot ECG signal
        ax.plot(samples, heartbeat_data, color=color, linewidth=linewidth, label='ECG Signal')
        
        # Add peak detection if requested
        peaks = None
        if show_peaks:
            peaks = _detect_peaks(heartbeat_data, threshold=peak_threshold)
            if len(peaks) > 0:
                peak_samples = samples[peaks]
                peak_values = heartbeat_data[peaks]
                ax.scatter(peak_samples, peak_values, color='red', s=30, zorder=5, 
                          label=f'R-peaks ({len(peaks)})')
        
        # Customize subplot
        subplot_title = f"Heartbeat {i+1}"
        if label is not None:
            subplot_title += f" (Class: {label})"
        
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude (mV)')
        
        # Add legend for this subplot
        if show_peaks and peaks is not None and len(peaks) > 0:
            ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=8)
        else:
            ax.legend(['ECG Signal'], frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=8)
        
        # Grid styling
        if grid:
            ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.5, color='gray')
            ax.set_axisbelow(True)
        
        # Set background colors
        ax.set_facecolor('white')
        
        # Add complete box styling
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
    
    # Hide unused subplots
    for i in range(n_heartbeats, len(axes)):
        axes[i].set_visible(False)
    
    # Set overall figure title
    if title is None:
        title = f"Multiple ECG Heartbeats ({n_heartbeats} samples)"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('white')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def load_heartbeat_from_csv(
    csv_path: str,
    row_index: int = 0,
    has_label: bool = True
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

def demo_heartbeat_visualization():
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
    df['label'] = [0, 1, 0, 2, 1, 0]  # Sample labels
    
    print("1. Single heartbeat visualization:")
    fig1 = plot_heartbeat(sample_heartbeats[0], title="Sample Single Heartbeat")
    plt.show()
    
    print("2. Multiple heartbeats visualization:")
    fig2 = plot_multiple_heartbeats(df, title="Sample Multiple Heartbeats", show_peaks=True)
    plt.show()
    
    print("Demo completed!")


if __name__ == "__main__":
    demo_heartbeat_visualization()
