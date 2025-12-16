"""
Page 9: Deep Learning Models PTB - Transfer Learning
Same as Page 8 but as "transfer learning" on PTB
Todo by Julia
"""

import os
import sys
import base64
import random
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from page_modules.styles import COLORS

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.visualization import plot_heartbeat

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_9"
DATA_DIR = APP_DIR.parent / "data" / "original"
MODELS_DIR = APP_DIR.parent / "models"

# CNN8 Transfer Learning Model path (binary classification for PTB)
CNN8_TRANSFER_MODEL_PATH = MODELS_DIR / "PTB_04_02_dl_models" / "CNN8_TRANSFER" / "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_epoch_118_valloss_0.0471.keras"

# PTB Labels (binary classification)
PTB_LABELS_MAP = {0: "N", 1: "A"}
PTB_LABELS_TO_DESC = {"N": "Normal", "A": "Abnormal"}

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

    return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">{caption_html}'


@st.cache_data
def load_test_data(samples_per_class=20):
    """Load PTB test data with caching and stratified sampling."""
    import numpy as np

    # Load X and y separately
    X_test_path = str(IMAGES_DIR / "X_ptb_test.csv")
    y_test_path = str(IMAGES_DIR / "y_ptb_test.csv")

    X_test_full = pd.read_csv(X_test_path)
    y_test_full = pd.read_csv(y_test_path).values.flatten().astype(int)

    # Sample stratified - samples_per_class samples per class
    sampled_indices = []
    np.random.seed(42)  # For reproducibility

    for class_label in range(2):  # Classes 0-1 for PTB
        class_indices = np.where(y_test_full == class_label)[0]
        if len(class_indices) > samples_per_class:
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected = class_indices
        sampled_indices.extend(selected)

    sampled_indices = np.array(sampled_indices)
    X_test = X_test_full.iloc[sampled_indices]
    y_test = y_test_full[sampled_indices]

    return X_test, y_test, sampled_indices


@st.cache_resource
def load_model_joblib(model_path):
    """Load the model saved with joblib - avoids TensorFlow segmentation faults."""
    try:
        import joblib

        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_keras_model(model_path):
    """Load Keras model with caching.
    
    Uses standalone Keras 3 API for compatibility with models saved
    using Keras 3.x (which uses keras.src.models.functional internally).
    """
    try:
        import keras
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None


@st.cache_data
def load_ptb_test_data():
    """Load PTB test data with caching (combined normal + abnormal)."""
    try:
        # Load both PTB files
        df_normal = pd.read_csv(DATA_DIR / "ptbdb_normal.csv", header=None)
        df_abnormal = pd.read_csv(DATA_DIR / "ptbdb_abnormal.csv", header=None)
        
        # Add labels (0 for normal, 1 for abnormal)
        df_normal[187] = 0
        df_abnormal[187] = 1
        
        # Combine datasets
        df_combined = pd.concat([df_normal, df_abnormal], ignore_index=True)
        
        # Shuffle the data
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_test = df_combined.drop(187, axis=1)
        y_test = df_combined[187]
        
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading PTB test data: {e}")
        return None, None


def render_citations():
    """Render citations section with horizontal separator."""
    st.markdown("---")
    with st.expander("📚 Citations", expanded=False):
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']};">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[1]</strong> Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); 
                    <a href="https://doi.org/10.3389/fphys.2023.1246746" style="color: {COLORS['clinical_blue_light']};">doi: 10.3389/fphys.2023.1246746</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[2]</strong> ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee, S. Fazeli, M. Sarrafzadeh (2018); CoRR; 
                    <a href="https://doi.org/10.48550/arXiv.1805.00794" style="color: {COLORS['clinical_blue_light']};">doi: 10.48550/arXiv.1805.00794</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[3]</strong> <a href="https://www.datasci.com/solutions/cardiovascular/ecg-research" style="color: {COLORS['clinical_blue_light']};">https://www.datasci.com/solutions/cardiovascular/ecg-research</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[4]</strong> ECG-based heartbeat classification for arrhythmia detection: A survey; E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, D. Menotti (2015); Computer Methods and Programs in Biomedicine; 
                    <a href="https://doi.org/10.1016/j.cmpb.2015.12.008" style="color: {COLORS['clinical_blue_light']};">doi: 10.1016/j.cmpb.2015.12.008</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>[5]</strong> Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M. Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; 
                    <a href="https://doi.org/10.1016/j.compbiomed.2020.103726" style="color: {COLORS['clinical_blue_light']};">doi:10.1016/j.compbiomed.2020.103726</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🧠 Deep Learning Models - PTB Dataset</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Hero header for main section
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔄 Transfer Learning Approach</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    main_tab1, main_tab2, main_tab3 = st.tabs(
        [
            "Tested Configurations - CNN8 not trainable",
            "Optimization - Unfreeze Last Block",
            "🔮 Model Prediction",
        ]
    )

    with main_tab1:
        # -------------------------------------------------------
        # Section 1 — Top 3 Transfer Learning Models Overview (always visible)
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🏆 Top 3 Transfer Learning Configurations</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Three model cards in a row with equal height using flexbox
        st.markdown(
            f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div style="flex: 1; background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid {COLORS['success']};">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.5rem;">🥇 CNN8 + transfer2</h4>
                    <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                        <li>CNN8 with BatchNorm (all blocks frozen)</li>
                        <li>Transfer head: 2 dense layers + dropout (0.1)</li>
                        <li>F1 score: <strong style="color: {COLORS['success']};">0.8915</strong></li>
                    </ul>
                </div>
                <div style="flex: 1; background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid {COLORS['clinical_blue_light']};">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.5rem;">🥈 CNN8 + transfer7</h4>
                    <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                        <li>CNN8 with BatchNorm (all blocks frozen)</li>
                        <li>Transfer head: complex with BatchNorm + dropout (0.1)</li>
                        <li>F1 score: <strong style="color: {COLORS['clinical_blue_light']};">0.8879</strong></li>
                    </ul>
                </div>
                <div style="flex: 1; background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid {COLORS['warning']};">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.5rem;">🥉 CNN8 + transfer8</h4>
                    <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                        <li>CNN8 with BatchNorm (all blocks frozen)</li>
                        <li>Transfer head: same as transfer7 + dropout (0.2)</li>
                        <li>F1 score: <strong style="color: {COLORS['warning']};">0.8850</strong></li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------
        # Section 2 — Result Table
        # -------------------------------------------------------
        with st.expander("📋 Result Table", expanded=False):
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
            # Load CSV file
            csv_path = str(IMAGES_DIR / "dl_3.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep=";", index_col=0)

                # Find the F1 Test column (handle potential spacing issues)
                f1_col = None
                for col in df.columns:
                    if "F1" in col and "Test" in col:
                        f1_col = col
                        break

                if f1_col:
                    # Sort by F1 Test descending
                    df.sort_values(by=f1_col, ascending=False, inplace=True)
                    df.reset_index(drop=False, inplace=True)

                    # Highlight first row
                    HIGHLIGHT_INDICES = {0, 1, 2}

                    def highlight_specific(row):
                        if row.name in HIGHLIGHT_INDICES:
                            return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
                        return [""] * len(row)

                    styled_df = df.style.apply(highlight_specific, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("⚠️ Please add dl_3.csv to the page_modules/ directory")

        with st.expander("Model Architecture - Top 3 Models", expanded=False):
            tab1, tab2, tab3 = st.tabs(["CNN8 + transfer2", "CNN8 + transfer7", "CNN8 + transfer8"])

            # Map model choice to summary file
            summary_files = {
                "transfer2": "transfer2_summary.txt",
                "transfer7": "transfer7_summary.txt",
                "transfer8": "transfer8_summary.txt",
            }

            # CNN8 + transfer2 Tab
            with tab1:
                summary_file = summary_files["transfer2"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

            # CNN8 + transfer7 Tab
            with tab2:
                summary_file = summary_files["transfer7"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

            # CNN8 + transfer8 Tab
            with tab3:
                summary_file = summary_files["transfer8"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

        with st.expander("Accuracy & Loss Curves - Top 3 Models", expanded=False):
            tab1, tab2, tab3 = st.tabs(["CNN8 + transfer2", "CNN8 + transfer7", "CNN8 + transfer8"])

            # Map model choice to image files
            loss_images = {
                "transfer2": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_transfer2_sm_lrexpdec1e-3_earlystop_bs512_loss.png",
                "transfer7": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_transfer7_sm_lrexpdec1e-3_earlystop_bs128_loss.png",
                "transfer8": "cnn8_sm_lrexpdec5e-4_earlystop_bs512_epoch52_transfer8_sm_lrexpdec1e-3_earlystop_bs128_loss.png",
            }

            accuracy_images = {
                "transfer2": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_transfer2_sm_lrexpdec1e-3_earlystop_bs512_accuracy.png",
                "transfer7": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_transfer7_sm_lrexpdec1e-3_earlystop_bs128_accuracy.png",
                "transfer8": "cnn8_sm_lrexpdec5e-4_earlystop_bs512_epoch52_transfer8_sm_lrexpdec1e-3_earlystop_bs128_accuracy.png",
            }

            # CNN8 + transfer2 Tab
            with tab1:
                loss_file = loss_images["transfer2"]
                accuracy_file = accuracy_images["transfer2"]
                loss_path = str(IMAGES_DIR / loss_file)
                accuracy_path = str(IMAGES_DIR / accuracy_file)

                if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                    loss_img = get_image_html(Path(loss_path), "Loss curve", "")
                    accuracy_img = get_image_html(Path(accuracy_path), "Accuracy curve", "")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{loss_img}</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{accuracy_img}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("⚠️ Training history images not found for CNN8 + transfer2.")

            # CNN8 + transfer7 Tab
            with tab2:
                loss_file = loss_images["transfer7"]
                accuracy_file = accuracy_images["transfer7"]
                loss_path = str(IMAGES_DIR / loss_file)
                accuracy_path = str(IMAGES_DIR / accuracy_file)

                if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                    loss_img = get_image_html(Path(loss_path), "Loss curve", "")
                    accuracy_img = get_image_html(Path(accuracy_path), "Accuracy curve", "")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{loss_img}</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{accuracy_img}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("⚠️ Training history images not found for CNN8 + transfer7.")

            # CNN8 + transfer8 Tab
            with tab3:
                loss_file = loss_images["transfer8"]
                accuracy_file = accuracy_images["transfer8"]
                loss_path = str(IMAGES_DIR / loss_file)
                accuracy_path = str(IMAGES_DIR / accuracy_file)

                if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                    loss_img = get_image_html(Path(loss_path), "Loss curve", "")
                    accuracy_img = get_image_html(Path(accuracy_path), "Accuracy curve", "")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{loss_img}</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{accuracy_img}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("⚠️ Training history images not found for CNN8 + transfer8.")

        render_citations()

    with main_tab2:
        # -------------------------------------------------------
        # Section 1 — Best Model Overview (always visible)
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🏆 Step 1 – Best Model: CNN8 + transfer6</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']}; margin-bottom: 1rem;">
                <p style="margin: 0 0 0.5rem 0; color: {COLORS['text_primary']};">
                    <strong>CNN8 + transfer6:</strong> Transfer learning from MIT-trained CNN8
                </p>
                <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['text_primary']};">
                    <li>Base model: CNN8 from MIT dataset (architecture from [2] with dropout layers)</li>
                    <li>Transfer learning: Last residual block unfrozen</li>
                    <li>Transfer model: New classification part adapted for binary classification</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Training procedure and overall metrics in columns - using flexbox for equal height
        st.markdown(
            f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div style="flex: 1; background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid {COLORS['clinical_blue_light']};">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.75rem;">🔧 Transfer Learning Training Procedure</h4>
                    <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']};">
                        <li>Batch size: 128</li>
                        <li>Learning rate start: 0.001</li>
                        <li>Learning rate reduction: exponential decay</li>
                        <li>Last epoch: 118</li>
                    </ul>
                </div>
                <div style="flex: 1; background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid {COLORS['success']};">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.75rem;">📊 Average Performance on Test Data</h4>
                    <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']};">
                        <li>F1 score: <strong style="color: {COLORS['success']};">0.9805</strong></li>
                        <li>Accuracy: 0.9842</li>
                        <li>Precision: 0.9751</li>
                        <li>Recall: 0.9846</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # -------------------------------------------------------
        # Section 2 — Result Table
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">📋 Step 2 – All Model Results</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize session state for button
        if "page9_show_results" not in st.session_state:
            st.session_state["page9_show_results"] = False

        if st.button("📋 Show Result Table", key="page9_results_btn"):
            st.session_state["page9_show_results"] = True

        if st.session_state["page9_show_results"]:
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
            # Load CSV file
            csv_path = str(IMAGES_DIR / "dl_4.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep=";", index_col=0)

                # Find the F1 Test column (handle potential spacing issues)
                f1_col = None
                for col in df.columns:
                    if "F1" in col and "Test" in col:
                        f1_col = col
                        break

                if f1_col:
                    # Sort by F1 Test descending
                    df.sort_values(by=f1_col, ascending=False, inplace=True)
                    df.reset_index(drop=False, inplace=True)

                    # Highlight first row
                    HIGHLIGHT_INDICES = {0}

                    def highlight_specific(row):
                        if row.name in HIGHLIGHT_INDICES:
                            return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
                        return [""] * len(row)

                    styled_df = df.style.apply(highlight_specific, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("⚠️ Please add dl_4.csv to the page_modules/ directory")

        st.markdown("---")

        # -------------------------------------------------------
        # Section 3 — Classification Report
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">📊 Step 3 – Classification Report</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize session state for button
        if "page9_show_report" not in st.session_state:
            st.session_state["page9_show_report"] = False

        if st.button("📊 Generate Classification Report", key="page9_report_btn"):
            st.session_state["page9_show_report"] = True

        if st.session_state["page9_show_report"]:
            # Macro metrics with color gradients
            f1_macro = 0.9805
            prec_macro = 0.9751
            rec_macro = 0.9846

            def get_metric_gradient(value):
                """Generate gradient based on metric value."""
                if value >= 0.9:
                    return f"linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%)"
                elif value >= 0.8:
                    return (
                        f"linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%)"
                    )
                else:
                    return f"linear-gradient(135deg, {COLORS['warning']} 0%, #B8860B 100%)"

            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="background: {get_metric_gradient(f1_macro)}; 
                                padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                        <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{f1_macro:.4f}</div>
                        <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">F1-Macro</div>
                    </div>
                    <div style="background: {get_metric_gradient(prec_macro)}; 
                                padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                        <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{prec_macro:.4f}</div>
                        <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Precision-Macro</div>
                    </div>
                    <div style="background: {get_metric_gradient(rec_macro)}; 
                                padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                        <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{rec_macro:.4f}</div>
                        <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Recall-Macro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Per-class metrics dataframe
            report_df = pd.DataFrame(
                {
                    "precision": [0.9536, 0.9966],
                    "recall": [0.9913, 0.9814],
                    "f1-score": [0.9721, 0.9890],
                },
                index=["Class 0 (Normal)", "Class 1 (Abnormal)"],
            )
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------
        # Section 4 — Accuracy & Loss Curves
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">📈 Step 4 – Training History</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize session state for button
        if "page9_show_curves" not in st.session_state:
            st.session_state["page9_show_curves"] = False

        if st.button("📈 Show Accuracy & Loss Curves", key="page9_curves_btn"):
            st.session_state["page9_show_curves"] = True

        if st.session_state["page9_show_curves"]:
            loss_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_loss.png"
            accuracy_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_accuracy.png"
            loss_path = str(IMAGES_DIR / loss_file)
            accuracy_path = str(IMAGES_DIR / accuracy_file)

            if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                loss_img = get_image_html(Path(loss_path), "Loss curve", "")
                accuracy_img = get_image_html(Path(accuracy_path), "Accuracy curve", "")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f'<div style="text-align: center;">{loss_img}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div style="text-align: center;">{accuracy_img}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error("⚠️ Training history images not found for CNN8 + transfer6.")

        st.markdown("---")

        # -------------------------------------------------------
        # Section 5 — Confusion Matrix
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🧩 Step 5 – Confusion Matrix</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize session state for button
        if "page9_show_cm" not in st.session_state:
            st.session_state["page9_show_cm"] = False

        if st.button("🧩 Show Confusion Matrix", key="page9_cm_btn"):
            st.session_state["page9_show_cm"] = True

        if st.session_state["page9_show_cm"]:
            cm_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_epoch_118_valloss_0.0471_cm.png"
            cm_path = str(IMAGES_DIR / cm_file)

            if os.path.exists(cm_path):
                # Centered confusion matrix image at 80% width
                cm_img = get_image_html(Path(cm_path), "Confusion Matrix", "")
                st.markdown(
                    f'<div style="display: flex; justify-content: center; margin-bottom: 1.5rem;">'
                    f'<div style="width: 80%; max-width: 600px; text-align: center;">{cm_img}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Classification Performance and Misclassifications in one row
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                                    border-left: 3px solid {COLORS['success']}; height: 100%;">
                            <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.75rem;">✅ Classification Performance</h4>
                            <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                                <li>Class 0: 99% correct</li>
                                <li>Class 1: 98% correct</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                                    border-left: 3px solid {COLORS['warning']}; height: 100%;">
                            <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.75rem;">⚠️ Misclassifications</h4>
                            <ul style="margin: 0; padding-left: 1.25rem; color: {COLORS['text_primary']}; font-size: 0.9rem;">
                                <li>Class 1 → Class 0: <strong>2%</strong></li>
                                <li>Class 0 → Class 1: 1%</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Problematic Misclassifications in its own row
                st.markdown(
                    f"""
                    <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                                border-left: 3px solid {COLORS['heart_red']}; margin-top: 1rem;">
                        <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0; margin-bottom: 0.75rem;">🚨 Problematic Misclassifications</h4>
                        <p style="margin: 0; color: {COLORS['text_primary']};">
                            Class 1 predicted as class 0 → possibility of missing diagnoses
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    f"""
                ⚠️ Confusion matrix image not found: {cm_file}

                Please place the PNG file in the `page_modules/` directory.
                """
                )

        render_citations()

    with main_tab3:
        _render_model_prediction_tab()


def _render_model_prediction_tab():
    """Render the Model Prediction tab with live Keras model predictions."""
    PREFIX = "page9_live_"
    
    import matplotlib.pyplot as plt

    # Initialize session state
    if f"{PREFIX}model_loaded" not in st.session_state:
        st.session_state[f"{PREFIX}model_loaded"] = False
    if f"{PREFIX}model" not in st.session_state:
        st.session_state[f"{PREFIX}model"] = None
    if f"{PREFIX}X_test" not in st.session_state:
        st.session_state[f"{PREFIX}X_test"] = None
    if f"{PREFIX}y_test" not in st.session_state:
        st.session_state[f"{PREFIX}y_test"] = None
    if f"{PREFIX}normal_sample" not in st.session_state:
        st.session_state[f"{PREFIX}normal_sample"] = None
    if f"{PREFIX}normal_sample_idx" not in st.session_state:
        st.session_state[f"{PREFIX}normal_sample_idx"] = None
    if f"{PREFIX}abnormal_sample" not in st.session_state:
        st.session_state[f"{PREFIX}abnormal_sample"] = None
    if f"{PREFIX}abnormal_sample_idx" not in st.session_state:
        st.session_state[f"{PREFIX}abnormal_sample_idx"] = None

    # Hero header for Model Prediction
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔮 Model Prediction - CNN8 Transfer</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Description in styled container
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 1.25rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1.5rem;">
            <p style="margin: 0; opacity: 0.95;">
                Compare predictions for a <strong>normal heartbeat (Class 0)</strong> and an <strong>abnormal heartbeat (Class 1)</strong>
                using the CNN8 transfer learning model with live inference.
                Both samples can be randomly selected or chosen by occurrence number.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -------------------------------------------------------
    # Step 1 — Load Model + Data
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">📥 Step 1 – Load Test Data & Model</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state[f"{PREFIX}model_loaded"] and st.session_state[f"{PREFIX}model"] is not None:
        st.success("✅ CNN8 Transfer Model and data are already loaded.")
        st.write(f"**Dataset size:** {st.session_state[f'{PREFIX}X_test'].shape}")
        
        # Class distribution info
        class_counts = st.session_state[f"{PREFIX}y_test"].value_counts().sort_index()
        labels = [f"{PTB_LABELS_MAP[i]} ({PTB_LABELS_TO_DESC[PTB_LABELS_MAP[i]]})" for i in class_counts.index]
        colors = plt.cm.Set3(range(len(class_counts)))

        fig_pie, ax_pie = plt.subplots(figsize=(8, 4))
        ax_pie.pie(
            class_counts.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
        )
        ax_pie.set_title("Test Data Class Distribution", fontsize=12, pad=10)
        st.pyplot(fig_pie)
        plt.close()

        if st.button("🔄 Reload Model & Data", key=f"{PREFIX}reload_btn"):
            st.session_state[f"{PREFIX}model_loaded"] = False
            st.session_state[f"{PREFIX}model"] = None
            st.session_state[f"{PREFIX}X_test"] = None
            st.session_state[f"{PREFIX}y_test"] = None
            st.session_state[f"{PREFIX}normal_sample"] = None
            st.session_state[f"{PREFIX}abnormal_sample"] = None
            st.rerun()
    elif st.button("📥 Load Test Data & Model", key=f"{PREFIX}load_btn"):
        with st.spinner("Loading CNN8 Transfer model and test data..."):
            try:
                # Load test data
                X_test, y_test = load_ptb_test_data()
                
                if X_test is None or y_test is None:
                    st.error("Failed to load test data.")
                    return
                
                # Load Keras model
                model = load_keras_model(str(CNN8_TRANSFER_MODEL_PATH))
                
                if model is None:
                    st.error("Failed to load CNN8 Transfer model.")
                    return
                
                # Save to session state
                st.session_state[f"{PREFIX}X_test"] = X_test
                st.session_state[f"{PREFIX}y_test"] = y_test
                st.session_state[f"{PREFIX}model"] = model
                st.session_state[f"{PREFIX}model_loaded"] = True
                
                st.success("CNN8 Transfer Model & Data successfully loaded.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading model/data: {str(e)}")
                return

    st.markdown("---")

    # Check if model is loaded before proceeding
    if not st.session_state[f"{PREFIX}model_loaded"] or st.session_state[f"{PREFIX}model"] is None:
        st.info("⚠️ Please load the model and data first using the button above.")
        render_citations()
        return

    # -------------------------------------------------------
    # Step 2 — Select Samples
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">🎯 Step 2 – Select Samples</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get class indices
    class_0_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == 0
    ].index.tolist()
    class_1_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == 1
    ].index.tolist()
    max_n_normal = len(class_0_indices)
    max_n_abnormal = len(class_1_indices)

    # Initialize samples if not set
    if st.session_state[f"{PREFIX}normal_sample"] is None and max_n_normal > 0:
        normal_idx = class_0_indices[0]
        st.session_state[f"{PREFIX}normal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[normal_idx]
        st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

    if st.session_state[f"{PREFIX}abnormal_sample"] is None and max_n_abnormal > 0:
        abnormal_idx = class_1_indices[0]
        st.session_state[f"{PREFIX}abnormal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[abnormal_idx]
        st.session_state[f"{PREFIX}abnormal_sample_idx"] = abnormal_idx

    # Selection method
    selection_method = st.radio(
        "Selection method:",
        ["Random sample", "Nth occurrence"],
        key=f"{PREFIX}selection_method",
    )

    if selection_method == "Random sample":
        if st.button("🔮 Predict!", key=f"{PREFIX}predict_random_btn"):
            # Randomize both samples
            if max_n_normal > 0:
                random_pos_normal = random.randint(0, max_n_normal - 1)
                normal_idx = class_0_indices[random_pos_normal]
                st.session_state[f"{PREFIX}normal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[normal_idx]
                st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

            if max_n_abnormal > 0:
                random_pos_abnormal = random.randint(0, max_n_abnormal - 1)
                abnormal_idx = class_1_indices[random_pos_abnormal]
                st.session_state[f"{PREFIX}abnormal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[abnormal_idx]
                st.session_state[f"{PREFIX}abnormal_sample_idx"] = abnormal_idx
    else:  # Nth occurrence
        col1, col2 = st.columns(2)
        
        with col1:
            if max_n_normal > 0:
                n_occurrence_normal = st.number_input(
                    f"Normal (Class 0) - Nth occurrence (1 to {max_n_normal}):",
                    min_value=1,
                    max_value=max_n_normal,
                    value=1,
                    key=f"{PREFIX}n_occurrence_normal",
                )

        with col2:
            if max_n_abnormal > 0:
                n_occurrence_abnormal = st.number_input(
                    f"Abnormal (Class 1) - Nth occurrence (1 to {max_n_abnormal}):",
                    min_value=1,
                    max_value=max_n_abnormal,
                    value=1,
                    key=f"{PREFIX}n_occurrence_abnormal",
                )

        if st.button("🔮 Predict!", key=f"{PREFIX}get_samples_btn"):
            # Set normal sample
            if max_n_normal > 0 and n_occurrence_normal <= max_n_normal:
                normal_idx = class_0_indices[n_occurrence_normal - 1]
                st.session_state[f"{PREFIX}normal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[normal_idx]
                st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

            # Set abnormal sample
            if max_n_abnormal > 0 and n_occurrence_abnormal <= max_n_abnormal:
                abnormal_idx = class_1_indices[n_occurrence_abnormal - 1]
                st.session_state[f"{PREFIX}abnormal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[abnormal_idx]
                st.session_state[f"{PREFIX}abnormal_sample_idx"] = abnormal_idx

    # Get current samples from session state
    normal_sample = st.session_state[f"{PREFIX}normal_sample"]
    normal_idx = st.session_state[f"{PREFIX}normal_sample_idx"]
    abnormal_sample = st.session_state[f"{PREFIX}abnormal_sample"]
    abnormal_idx = st.session_state[f"{PREFIX}abnormal_sample_idx"]

    # Display predictions if both samples are available
    if normal_sample is not None and abnormal_sample is not None:
        st.markdown("---")

        # -------------------------------------------------------
        # Step 3 — ECG Visualization & Predictions
        # -------------------------------------------------------
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">📈 Step 3 – ECG Signal Visualization & Predictions</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model = st.session_state[f"{PREFIX}model"]

        # Prepare data for CNN model (needs reshaping for 1D CNN)
        # CNN expects shape: (batch_size, timesteps, features)
        normal_array = normal_sample.values.reshape(1, -1, 1)
        abnormal_array = abnormal_sample.values.reshape(1, -1, 1)

        # Make predictions
        with st.spinner("Running CNN8 Transfer inference..."):
            normal_probabilities = model.predict(normal_array, verbose=0)[0]
            abnormal_probabilities = model.predict(abnormal_array, verbose=0)[0]

        normal_prediction = int(np.argmax(normal_probabilities))
        abnormal_prediction = int(np.argmax(abnormal_probabilities))
        normal_true_label = 0
        abnormal_true_label = 1

        # Side-by-side ECG visualization
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_pred_str = PTB_LABELS_MAP[normal_prediction]
            fig1 = plot_heartbeat(
                normal_sample.values,
                title=f"Predicted: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]})",
                color="green" if normal_prediction == normal_true_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.write("**Abnormal Sample (Class 1)**")
            abnormal_pred_str = PTB_LABELS_MAP[abnormal_prediction]
            fig2 = plot_heartbeat(
                abnormal_sample.values,
                title=f"Predicted: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]})",
                color="green" if abnormal_prediction == abnormal_true_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig2)
            plt.close()

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_true_str = PTB_LABELS_MAP[normal_true_label]
            normal_pred_str = PTB_LABELS_MAP[normal_prediction]

            st.markdown("**True Label:**")
            st.info(f"Class {normal_true_label}: {normal_true_str} ({PTB_LABELS_TO_DESC[normal_true_str]})")

            st.markdown("**Predicted Label:**")
            if normal_prediction == normal_true_label:
                st.success(f"Class {normal_prediction}: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]}) ✓")
            else:
                st.error(f"Class {normal_prediction}: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]}) ✗")

            st.write(f"**Sample Index:** {normal_idx}")

        with col2:
            st.write("**Abnormal Sample (Class 1)**")
            abnormal_true_str = PTB_LABELS_MAP[abnormal_true_label]
            abnormal_pred_str = PTB_LABELS_MAP[abnormal_prediction]

            st.markdown("**True Label:**")
            st.info(f"Class {abnormal_true_label}: {abnormal_true_str} ({PTB_LABELS_TO_DESC[abnormal_true_str]})")

            st.markdown("**Predicted Label:**")
            if abnormal_prediction == abnormal_true_label:
                st.success(f"Class {abnormal_prediction}: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]}) ✓")
            else:
                st.error(f"Class {abnormal_prediction}: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]}) ✗")

            st.write(f"**Sample Index:** {abnormal_idx}")

        # Display probabilities tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Normal Sample Probabilities**")
            normal_prob_df = pd.DataFrame(
                {
                    "Class": [PTB_LABELS_MAP[i] for i in range(2)],
                    "Description": [PTB_LABELS_TO_DESC[PTB_LABELS_MAP[i]] for i in range(2)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in normal_probabilities],
                    "_sort": normal_probabilities,
                }
            )
            normal_prob_df = normal_prob_df.sort_values("_sort", ascending=False)
            normal_prob_df = normal_prob_df.drop(columns=["_sort"])
            st.dataframe(normal_prob_df, use_container_width=True)

        with col2:
            st.markdown("**📊 Abnormal Sample Probabilities**")
            abnormal_prob_df = pd.DataFrame(
                {
                    "Class": [PTB_LABELS_MAP[i] for i in range(2)],
                    "Description": [PTB_LABELS_TO_DESC[PTB_LABELS_MAP[i]] for i in range(2)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in abnormal_probabilities],
                    "_sort": abnormal_probabilities,
                }
            )
            abnormal_prob_df = abnormal_prob_df.sort_values("_sort", ascending=False)
            abnormal_prob_df = abnormal_prob_df.drop(columns=["_sort"])
            st.dataframe(abnormal_prob_df, use_container_width=True)

    render_citations()
