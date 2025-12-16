"""
Page 9: Deep Learning Models PTB - Transfer Learning
Same as Page 8 but as "transfer learning" on PTB
Todo by Julia
"""

import os
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from page_modules.styles import COLORS

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_9"

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
    """Render the Model Prediction tab"""
    # Hero header for Model Predictions
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔮 Model Predictions</div>'
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
                Explore how the transfer learning model performs on individual test samples from the PTB dataset.
                Select a sample to see the ECG signal and prediction results for binary classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load test data and precomputed predictions
    X_test_path = str(IMAGES_DIR / "X_ptb_test.csv")
    y_test_path = str(IMAGES_DIR / "y_ptb_test.csv")
    predictions_path = str(IMAGES_DIR / "precomputed_predictions_transfer.csv")

    if (
        os.path.exists(X_test_path)
        and os.path.exists(y_test_path)
        and os.path.exists(predictions_path)
    ):
        # Load test data using cached function (20 samples per class)
        X_test, y_test, sampled_indices = load_test_data(samples_per_class=20)

        # Load precomputed predictions
        predictions_df = pd.read_csv(predictions_path)

        # Create tabs for each class
        tab0, tab1 = st.tabs(["Class 0", "Class 1"])

        tabs = [tab0, tab1]
        class_names = {0: "Normal", 1: "Abnormal"}

        for class_idx, tab in enumerate(tabs):
            with tab:
                # Filter samples by class from the sampled_indices
                class_mask = y_test == class_idx
                class_positions = [i for i, mask in enumerate(class_mask) if mask]
                class_original_indices = [sampled_indices[i] for i in class_positions]

                if len(class_original_indices) == 0:
                    st.warning(f"No samples found for Class {class_idx}")
                    continue

                # Sample selector - show original index from full dataset
                sample_idx = st.selectbox(
                    "Select sample:",
                    class_original_indices,
                    key=f"sample_class_{class_idx}",
                )

                # Get the selected sample using position in X_test
                position_in_sampled = list(sampled_indices).index(sample_idx)
                X_sample = X_test.iloc[position_in_sampled].values
                y_sample = y_test[position_in_sampled]

                # Display ECG signal
                st.markdown("**ECG Signal:**")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(X_sample, linewidth=0.8)
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Get prediction for this sample
                pred_row = predictions_df[predictions_df["sample_index"] == sample_idx]

                if not pred_row.empty:
                    predicted_class = int(pred_row["predicted_label"].values[0])
                    prediction_probs = pred_row[["prob_class_0", "prob_class_1"]].values[0]

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**True Label:**")
                        st.info(f"Class {y_sample}: {class_names[y_sample]}")

                    with col2:
                        st.markdown("**Predicted Label:**")
                        if predicted_class == y_sample:
                            st.success(f"Class {predicted_class}: {class_names[predicted_class]} ✓")
                        else:
                            st.error(f"Class {predicted_class}: {class_names[predicted_class]} ✗")

                    # Show prediction probabilities
                    st.markdown("**Prediction Probabilities:**")

                    # Custom CSS for centered table
                    st.markdown(
                        """
                        <style>
                        [data-testid="stTable"] table th,
                        [data-testid="stTable"] table td {
                            text-align: center !important;
                        }
                        [data-testid="stTable"] thead tr th:first-child,
                        [data-testid="stTable"] tbody tr th:first-child {
                            display: none;
                        }
                        </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    prob_df = pd.DataFrame(
                        {
                            "Class 0": [f"{prediction_probs[0]:.4f}"],
                            "Class 1": [f"{prediction_probs[1]:.4f}"],
                        },
                        index=[""],
                    )
                    st.table(prob_df)
                else:
                    st.error(f"No prediction found for sample {sample_idx}")
    else:
        st.error(
            """
        ⚠️ Required files not found.

        Please add the following files to page_modules/:
        - X_ptb_test.csv
        - y_ptb_test.csv
        - precomputed_predictions_transfer.csv
        """
        )

    render_citations()
