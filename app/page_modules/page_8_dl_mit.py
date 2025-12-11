"""
Page 8: Results on Deep Learning Models MIT Dataset
Description, Classification Report, Confusion Matrix, Live-Prediction
Accuracy / Loss Curves
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
IMAGES_DIR = APP_DIR / "images" / "page_8"

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


@st.cache_data
def load_test_data(test_data_path, samples_per_class=20):
    """Load test data with caching and stratified sampling."""
    import numpy as np

    test_df = pd.read_csv(test_data_path)
    X_test_full = test_df.iloc[:, :-1]
    y_true_full = test_df.iloc[:, -1].values.astype(int)

    # Sample stratified - samples_per_class samples per class
    sampled_indices = []
    np.random.seed(42)  # For reproducibility

    for class_label in range(5):  # Classes 0-4
        class_indices = np.where(y_true_full == class_label)[0]
        if len(class_indices) > samples_per_class:
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected = class_indices
        sampled_indices.extend(selected)

    sampled_indices = np.array(sampled_indices)
    X_test = X_test_full.iloc[sampled_indices]
    y_true = y_true_full[sampled_indices]

    return X_test, y_true, sampled_indices


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
                    <strong>[3]</strong> ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee, S. Fazeli, M. Sarrafzadeh (2018); CoRR; 
                    <a href="https://doi.org/10.48550/arXiv.1805.00794" style="color: {COLORS['clinical_blue_light']};">doi: 10.48550/arXiv.1805.00794</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[4]</strong> <a href="https://www.datasci.com/solutions/cardiovascular/ecg-research" style="color: {COLORS['clinical_blue_light']};">https://www.datasci.com/solutions/cardiovascular/ecg-research</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[5]</strong> Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); 
                    <a href="https://doi.org/10.3389/fphys.2023.1246746" style="color: {COLORS['clinical_blue_light']};">doi: 10.3389/fphys.2023.1246746</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[6]</strong> Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M. Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; 
                    <a href="https://doi.org/10.1016/j.compbiomed.2020.103726" style="color: {COLORS['clinical_blue_light']};">doi:10.1016/j.compbiomed.2020.103726</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>[7]</strong> ECG-based heartbeat classification for arrhythmia detection: A survey; E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, D. Menotti (2015); Computer Methods and Programs in Biomedicine; 
                    <a href="https://doi.org/10.1016/j.cmpb.2015.12.008" style="color: {COLORS['clinical_blue_light']};">doi: 10.1016/j.cmpb.2015.12.008</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🧠 Deep Learning Models - MIT Dataset</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Hero header for main section
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔍 Find Best DL Model Option</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    main_tab1, main_tab2, main_tab3 = st.tabs(
        ["Tested Models", "Optimization - CNN7 and CNN8", "🔮 Model Prediction"]
    )

    with main_tab1:
        with st.expander("Result Table", expanded=False):
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
            csv_path = str(IMAGES_DIR / "dl_1.csv")
            df = pd.read_csv(csv_path, sep=";", index_col=0)

            # Sort by F1 Test descending
            df.sort_values(by="F1 Test", ascending=False, inplace=True)
            df.reset_index(drop=False, inplace=True)

            # Highlight first 3 rows (top 3)
            HIGHLIGHT_INDICES = {0, 1, 2}

            def highlight_specific(row):
                if row.name in HIGHLIGHT_INDICES:
                    return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
                return [""] * len(row)

            styled_df = df.style.apply(highlight_specific, axis=1)
            st.dataframe(styled_df, width="stretch")

        with st.expander("Best DL Options - Top 3 Models", expanded=False):
            st.write(
                """
                1. **CNN7**:
                    * Model architecture from [3] with added batch normalization layers
                    * Five residual blocks, followed by fully connected layers
                    * Batch normalization layers after each convolutional layer
                    * F1 score on test data: **0.9117**
                2. **CNN8**:
                    * Model architecture from [3] with added dropout layers
                    * Five residual blocks, followed by fully connected layers
                    * Dropout layers at the end of each residual block (0.1)
                    * F1 score on test data: **0.8996**
                3. **CNN1**:
                    * Model architecture inspired by lessons with batch normalization and dropout layers
                    * 3 convolutional blocks followed by dense layers
                    * F1 score on test data: **0.8834**
                """
            )

        with st.expander("Model Architecture - Top 3 Models", expanded=False):
            tab1, tab2, tab3 = st.tabs(["CNN7", "CNN8", "CNN1"])

            # Map model choice to summary file
            summary_files = {
                "CNN7": "cnn7_summary.txt",
                "CNN8": "cnn8_summary.txt",
                "CNN1": "cnn1_summary.txt",
            }

            # CNN7 Tab
            with tab1:
                summary_file = summary_files["CNN7"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

            # CNN8 Tab
            with tab2:
                summary_file = summary_files["CNN8"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

            # CNN1 Tab
            with tab3:
                summary_file = summary_files["CNN1"]
                summary_path = str(IMAGES_DIR / summary_file)

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary_text = f.read()
                    st.code(summary_text, language="text")
                else:
                    st.error(f"⚠️ Model summary file not found: {summary_file}")

        with st.expander("Accuracy & Loss Curves - Top 3 Models", expanded=False):
            tab1, tab2, tab3 = st.tabs(["CNN7", "CNN8", "CNN1"])

            # Map model choice to image files
            loss_images = {
                "CNN7": "cnn7_sm_lrexpdec5e-4_earlystop_bs512_loss.png",
                "CNN8": "cnn8_sm_lrexpdec5e-4_earlystop_bs512_loss.png",
                "CNN1": "cnn1_sm_lrexpdec5e-4_earlystop_bs512_loss.png",
            }

            accuracy_images = {
                "CNN7": "cnn7_sm_lrexpdec5e-4_earlystop_bs512_accuracy.png",
                "CNN8": "cnn8_sm_lrexpdec5e-4_earlystop_bs512_accuracy.png",
                "CNN1": "cnn1_sm_lrexpdec5e-4_earlystop_bs512_accuracy.png",
            }

            # CNN7 Tab
            with tab1:
                loss_file = loss_images["CNN7"]
                accuracy_file = accuracy_images["CNN7"]
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
                    st.error("⚠️ Training history images not found for CNN7.")

            # CNN8 Tab
            with tab2:
                loss_file = loss_images["CNN8"]
                accuracy_file = accuracy_images["CNN8"]
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
                    st.error("⚠️ Training history images not found for CNN8.")

            # CNN1 Tab
            with tab3:
                loss_file = loss_images["CNN1"]
                accuracy_file = accuracy_images["CNN1"]
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
                    st.error("⚠️ Training history images not found for CNN1.")

        render_citations()

    with main_tab2:
        with st.expander("Result Table", expanded=False):
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
            csv_path = str(IMAGES_DIR / "dl_2.csv")
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
                st.dataframe(styled_df, width="stretch")
            else:
                st.dataframe(df, width="stretch")

        with st.expander("Best DL Option", expanded=False):
            st.write(
                """
            1. **CNN8**:
            * Model architecture from [2] with added dropout layers
            * Five residual blocks, followed by fully connected layers
            * Dropout layers at the end of each residual block (0.1)
            """
            )

            # Training procedure and overall metrics in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Training Procedure:**")
                st.write(
                    """
                - Batch size: 512
                - Learning rate start: 0.001
                - Learning rate reduction: exponential decay
                - Last epoch: 52
                """
                )

            with col2:
                st.markdown("**Average Performance on Test Data:**")
                st.write(
                    """
                - F1 score: **0.9236**
                - Accuracy: 0.9851
                - Precision: 0.9062
                - Recall: 0.9424
                """
                )

            st.markdown("---")

            # Per-class metrics in tabs with smaller text
            st.markdown("**Per-Class Metrics**")
            tab1, tab2, tab3 = st.tabs(["F1 Score", "Precision", "Recall"])

            # Custom CSS for centered table and hide index column
            st.markdown(
                """
                <style>
                [data-testid="stTable"] table {
                    width: 100%;
                }
                [data-testid="stTable"] th, [data-testid="stTable"] td {
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

            with tab1:
                metrics_df = pd.DataFrame(
                    {
                        "Class 0": [0.9924],
                        "Class 1": [0.8606],
                        "Class 2": [0.9600],
                        "Class 3": [0.8171],
                        "Class 4": [0.9876],
                    },
                    index=[""],
                )
                st.table(metrics_df)

            with tab2:
                metrics_df = pd.DataFrame(
                    {
                        "Class 0": [0.9946],
                        "Class 1": [0.8393],
                        "Class 2": [0.9580],
                        "Class 3": [0.7606],
                        "Class 4": [0.9816],
                    },
                    index=[""],
                )
                st.table(metrics_df)

            with tab3:
                metrics_df = pd.DataFrame(
                    {
                        "Class 0": [0.9902],
                        "Class 1": [0.8831],
                        "Class 2": [0.9620],
                        "Class 3": [0.8827],
                        "Class 4": [0.9938],
                    },
                    index=[""],
                )
                st.table(metrics_df)

        with st.expander("Confusion Matrix - Best DL Option", expanded=False):
            # Path to confusion matrix image
            cm_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_cm.png"
            cm_path = str(IMAGES_DIR / cm_file)

            if os.path.exists(cm_path):
                # Image on left, text on right
                cm_img = get_image_html(Path(cm_path), "Confusion Matrix", "")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(
                        f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{cm_img}</div>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        """
                    **Classification Performance:**
                    - Class 0 and class 4: in 99% of cases is the prediction correct
                    - Class 2: in 96% of cases is the prediction correct
                    - Class 3: in 90% of cases is the prediction correct
                    - Class 1: in 88% of cases is the prediction correct
                    """
                    )

                    st.markdown(
                        """
                    **Misclassifications:**
                    - Class 1 is predicted in **11%** of cases as class 0
                    - Class 3 is predicted in 5% of cases as class 0
                    - Class 3 is predicted in 5% of cases as class 2
                    - Class 3 is predicted in 5% of cases as class 1
                    - Class 2 is predicted in 2% of cases as class 3
                    """
                    )

                    st.markdown(
                        """
                    **Problematic Misclassifications:**
                    - Class 1-4 as class 0 -> possibility of missing diagnoses
                    """
                    )
            else:
                st.error(
                    f"""
                ⚠️ Confusion matrix image not found: {cm_file}

                Please place the PNG file in the `page_modules/` directory.
                """
                )

        with st.expander("Accuracy & Loss Curves - Best DL Option", expanded=False):
            tab1 = st.tabs(["CNN8"])

            # Map model choice to image files
            loss_images = {"CNN8": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_loss.png"}
            accuracy_images = {"CNN8": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_accuracy.png"}

            # CNN8 Tab
            with tab1[0]:
                loss_file = loss_images["CNN8"]
                accuracy_file = accuracy_images["CNN8"]
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
                    st.error("⚠️ Training history images not found for CNN8.")

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
                Explore how the CNN8 model performs on individual test samples from each class.
                Select a sample from each class to see the ECG signal and prediction results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load test data and precomputed predictions
    DATA_DIR = APP_DIR.parent / "data" / "original"
    test_data_path = str(DATA_DIR / "mitbih_test.csv")
    predictions_path = str(IMAGES_DIR / "precomputed_predictions_mit.csv")

    if os.path.exists(test_data_path) and os.path.exists(predictions_path):
        # Load test data using cached function (20 samples per class)
        X_test, y_true, sampled_indices = load_test_data(test_data_path, samples_per_class=20)

        # Load precomputed predictions
        predictions_df = pd.read_csv(predictions_path)

        # Create tabs for each class
        tab0, tab1, tab2, tab3, tab4 = st.tabs(
            ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
        )

        tabs = [tab0, tab1, tab2, tab3, tab4]
        class_names = {
            0: "Normal",
            1: "Supraventricular",
            2: "Ventricular",
            3: "Fusion",
            4: "Unknown",
        }

        for class_idx, tab in enumerate(tabs):
            with tab:
                # Filter samples by class from the sampled_indices
                class_mask = y_true == class_idx
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
                y_sample = y_true[position_in_sampled]

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
                    prediction_probs = pred_row[
                        [
                            "prob_class_0",
                            "prob_class_1",
                            "prob_class_2",
                            "prob_class_3",
                            "prob_class_4",
                        ]
                    ].values[0]

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
                            "Class 2": [f"{prediction_probs[2]:.4f}"],
                            "Class 3": [f"{prediction_probs[3]:.4f}"],
                            "Class 4": [f"{prediction_probs[4]:.4f}"],
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
        """
        )

    render_citations()
