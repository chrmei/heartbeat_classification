"""
Page 10: Result Summary
Expanded with visual comparisons and clinical implications
"""

import os
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from page_modules.styles import apply_matplotlib_style, COLORS, get_ecg_colors

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_10"

# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
    ext = image_path.suffix.lower()
    mime_types = {
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    mime = mime_types.get(ext, "image/png")
    b64 = get_image_base64(image_path)

    caption_html = (
        f'<p style="text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{caption}</p>'
        if caption
        else ""
    )

    return f"""
        <img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">
        {caption_html}
    """


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">📊 Results Summary</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ==========================================================================
    # KEY METRICS
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📊 Performance Comparison</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="MIT-BIH Accuracy",
            value="98.51%",
            delta="+5.11% vs benchmark",
            delta_color="normal",
        )

    with col2:
        st.metric(
            label="PTB Accuracy", value="98.42%", delta="+2.52% vs benchmark", delta_color="normal"
        )

    with col3:
        st.metric(
            label="MIT-BIH F1-Score",
            value="0.9236",
            delta="+0.08 vs baseline",
            delta_color="normal",
        )

    with col4:
        st.metric(label="Best Model", value="CNN8", delta="Transfer Learning", delta_color="off")

    st.markdown("---")

    # ==========================================================================
    # MODEL COMPARISON TABLE
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📋 Full Model Comparison</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Custom CSS for centered dataframe
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] table th,
        [data-testid="stDataFrame"] table td {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load results CSV file
    csv_path = str(IMAGES_DIR / "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")

        # Highlight best results
        def highlight_best(row):
            styles = [""] * len(row)
            if "CNN8" in str(row.get("Model", "")):
                styles = [f"background-color: rgba(45, 106, 79, 0.2)"] * len(row)
            return styles

        styled_df = df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, width="stretch", hide_index=True)
    else:
        # Create manual comparison table
        comparison_data = {
            "Model": ["XGBoost (Baseline)", "CNN8 (Deep Learning)", "Benchmark [3]"],
            "MIT-BIH Accuracy": ["97.2%", "98.51%", "93.4%"],
            "MIT-BIH F1": ["0.84", "0.92", "—"],
            "PTB Accuracy": ["97.8%", "98.42%", "95.9%"],
            "PTB F1": ["0.97", "0.98", "—"],
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("---")

    # ==========================================================================
    # VISUAL COMPARISON CHART
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📈 Performance Visualization</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    apply_matplotlib_style()

    tab_accuracy, tab_f1 = st.tabs(["Accuracy Comparison", "F1-Score by Class"])

    def render_citations():
        """Render citations section with horizontal separator."""
        st.markdown("---")
        # Page 10 doesn't have citations, so this is a placeholder
        pass

    with tab_accuracy:
        fig, ax = plt.subplots(figsize=(12, 5))

        models = ["Benchmark\n(Kachuee 2018)", "XGBoost\n(Baseline)", "CNN8\n(Deep Learning)"]
        mit_scores = [93.4, 97.2, 98.51]
        ptb_scores = [95.9, 97.8, 98.42]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, mit_scores, width, label="MIT-BIH", color=COLORS["clinical_blue"]
        )
        bars2 = ax.bar(x + width / 2, ptb_scores, width, label="PTB", color=COLORS["heart_red"])

        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(90, 100)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Add horizontal line for benchmark
        ax.axhline(
            y=93.4, color=COLORS["text_muted"], linestyle="--", alpha=0.5, label="MIT Benchmark"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        render_citations()

    with tab_f1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MIT-BIH per-class F1
        mit_classes = [
            "Class 0\n(Normal)",
            "Class 1\n(Supra.)",
            "Class 2\n(Ventr.)",
            "Class 3\n(Fusion)",
            "Class 4\n(Unknown)",
        ]
        mit_f1_scores = [0.9924, 0.8606, 0.9600, 0.8171, 0.9876]

        ecg_colors = get_ecg_colors()
        colors = [
            ecg_colors["class_0"],
            ecg_colors["class_1"],
            ecg_colors["class_2"],
            ecg_colors["class_3"],
            ecg_colors["class_4"],
        ]

        bars = axes[0].bar(
            mit_classes, mit_f1_scores, color=colors, edgecolor="white", linewidth=1.5
        )
        axes[0].set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        axes[0].set_title(
            "MIT-BIH: Per-Class F1-Score (CNN8)", fontsize=13, fontweight="bold", pad=15
        )
        axes[0].set_ylim(0.75, 1.05)
        axes[0].axhline(
            y=0.9236, color=COLORS["heart_red"], linestyle="--", linewidth=2, label="Macro Avg"
        )
        axes[0].legend(loc="lower right")

        for bar, score in zip(bars, mit_f1_scores):
            axes[0].annotate(
                f"{score:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # PTB per-class F1
        ptb_classes = ["Class 0\n(Normal)", "Class 1\n(MI)"]
        ptb_f1_scores = [0.97, 0.99]
        ptb_colors = [ecg_colors["normal"], ecg_colors["abnormal"]]

        bars = axes[1].bar(
            ptb_classes, ptb_f1_scores, color=ptb_colors, edgecolor="white", linewidth=1.5
        )
        axes[1].set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        axes[1].set_title(
            "PTB: Per-Class F1-Score (CNN8 Transfer)", fontsize=13, fontweight="bold", pad=15
        )
        axes[1].set_ylim(0.9, 1.05)

        for bar, score in zip(bars, ptb_f1_scores):
            axes[1].annotate(
                f"{score:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        render_citations()
