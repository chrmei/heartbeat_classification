"""
Page 9: Deep Learning Models PTB - Transfer Learning
Same as Page 8 but as "transfer learning" on PTB
Todo by Julia
"""

import os

import pandas as pd
import streamlit as st


@st.cache_data
def load_test_data(samples_per_class=20):
    """Load PTB test data with caching and stratified sampling."""
    import numpy as np

    # Load X and y separately
    X_test_path = os.path.join(
        os.path.dirname(__file__), "..", "images", "page_9", "X_ptb_test.csv"
    )
    y_test_path = os.path.join(
        os.path.dirname(__file__), "..", "images", "page_9", "y_ptb_test.csv"
    )

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


def render():
    st.title("Deep Learning Models - PTB Dataset (Transfer Learning)")
    st.markdown("---")

    st.header("Transfer Learning Approach")

    st.subheader("Tested Transfer Learning Configurations - CNN8 not trainable")
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
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "images", "page_9", "dl_3.csv"
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=";", index_col=0)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ Please add dl_3.csv to the page_modules/ directory")

    with st.expander("Best Transfer Learning Options - Top 3 Models", expanded=False):
        st.write("""
            1. **CNN8 + transfer2**:
                * CNN model architecture from [2] with added batch normalization layers
                * here: all five residual blocks not trainable
                * transfer 2: two dense layers followed by dropout layer (0.1)
                * F1 score on test data: **0.8915**
            2. **CNN8 + transfer7**:
                * CNN model architecture from [2] with added batch normalization layers
                * here: all five residual blocks not trainable
                * transfer 7: more complex than transfer 2 with dense layers, batch normalization, and dropout layers (0.1)
                * F1 score on test data: **0.8879**
            3. **CNN8 + transfer8**:
                * CNN model architecture from [2] with added batch normalization layers
                * here: all five residual blocks not trainable
                * transfer 8: same architecture as transfer 7 but with different dropout rates (0.2)
                * F1 score on test data: **0.8850**
            """)

    with st.expander("Accuracy & Loss Curves - Top 3 Models", expanded=False):
        tab1, tab2, tab3 = st.tabs(
            ["CNN8 + transfer2", "CNN8 + transfer7", "CNN8 + transfer8"]
        )

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
            loss_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", loss_file
            )
            accuracy_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", accuracy_file
            )

            if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(loss_path, use_container_width=True)
                with col2:
                    st.image(accuracy_path, use_container_width=True)
            else:
                st.error("⚠️ Training history images not found for CNN8 + transfer2.")

        # CNN8 + transfer7 Tab
        with tab2:
            loss_file = loss_images["transfer7"]
            accuracy_file = accuracy_images["transfer7"]
            loss_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", loss_file
            )
            accuracy_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", accuracy_file
            )

            if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(loss_path, use_container_width=True)
                with col2:
                    st.image(accuracy_path, use_container_width=True)
            else:
                st.error("⚠️ Training history images not found for CNN8 + transfer7.")

        # CNN8 + transfer8 Tab
        with tab3:
            loss_file = loss_images["transfer8"]
            accuracy_file = accuracy_images["transfer8"]
            loss_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", loss_file
            )
            accuracy_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", accuracy_file
            )

            if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(loss_path, use_container_width=True)
                with col2:
                    st.image(accuracy_path, use_container_width=True)
            else:
                st.error("⚠️ Training history images not found for CNN8 + transfer8.")

    with st.expander("Model Architecture - Top 3 Models", expanded=False):
        tab1, tab2, tab3 = st.tabs(
            ["CNN8 + transfer2", "CNN8 + transfer7", "CNN8 + transfer8"]
        )

        # Map model choice to summary file
        summary_files = {
            "transfer2": "transfer2_summary.txt",
            "transfer7": "transfer7_summary.txt",
            "transfer8": "transfer8_summary.txt",
        }

        # CNN8 + transfer2 Tab
        with tab1:
            summary_file = summary_files["transfer2"]
            summary_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", summary_file
            )

            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary_text = f.read()
                st.code(summary_text, language="text")
            else:
                st.error(f"⚠️ Model summary file not found: {summary_file}")

        # CNN8 + transfer7 Tab
        with tab2:
            summary_file = summary_files["transfer7"]
            summary_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", summary_file
            )

            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary_text = f.read()
                st.code(summary_text, language="text")
            else:
                st.error(f"⚠️ Model summary file not found: {summary_file}")

        # CNN8 + transfer8 Tab
        with tab3:
            summary_file = summary_files["transfer8"]
            summary_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", summary_file
            )

            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary_text = f.read()
                st.code(summary_text, language="text")
            else:
                st.error(f"⚠️ Model summary file not found: {summary_file}")

    st.subheader(
        "Optimization of Transfer Learning Configuration - Unfreeze Last Residual Block of CNN8"
    )
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
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "images", "page_9", "dl_4.csv"
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=";", index_col=0)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ Please add dl_4.csv to the page_modules/ directory")

    with st.expander("Best Transfer Learning Option", expanded=False):
        st.write("""
        **CNN8 + Transfer6:**
        * Base model: CNN8 from MIT dataset (architecture from [2] with dropout layers)
        * Transfer learning: Last residual block unfrozen
        * Transfer model: New classification part adapted for binary classification, added after pretrained CNN8
        """)

        # Training procedure and overall metrics in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Transfer Learning Training Procedure:**")
            st.write("""
            - Batch size: 128
            - Learning rate start: 0.001
            - Learning rate reduction: exponential decay
            - Last epoch: 118
            """)

        with col2:
            st.markdown("**Average Performance on Test Data:**")
            st.write("""
            - F1 score: **0.9805**
            - Accuracy: 0.9842
            - Precision: 0.9751
            - Recall: 0.9846
            """)

        st.markdown("---")

        # Per-class metrics in tabs
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
                    "Class 0": [0.9721],
                    "Class 1": [0.9890],
                },
                index=[""],
            )
            st.table(metrics_df)

        with tab2:
            metrics_df = pd.DataFrame(
                {
                    "Class 0": [0.9536],
                    "Class 1": [0.9966],
                },
                index=[""],
            )
            st.table(metrics_df)

        with tab3:
            metrics_df = pd.DataFrame(
                {
                    "Class 0": [0.9913],
                    "Class 1": [0.9814],
                },
                index=[""],
            )
            st.table(metrics_df)

    with st.expander("Confusion Matrix - Best Transfer Learning Model", expanded=False):
        # Path to confusion matrix image
        cm_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_epoch_118_valloss_0.0471_cm.png"
        cm_path = os.path.join(
            os.path.dirname(__file__), "..", "images", "page_9", cm_file
        )

        if os.path.exists(cm_path):
            # Image on left, text on right
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(cm_path, use_container_width=True)

            with col2:
                st.markdown("""
                **Classification Performance:**
                - Class 0: in 99% of cases is the prediction correct
                - Class 1: in 98% of cases is the prediction correct
                """)

                st.markdown("""
                **Misclassifications:**
                - Class 1 is predicted in **2%** of cases as class 0
                - Class 0 is predicted in 1% of cases as class 1
                """)

                st.markdown("""
                **Problematic Misclassifications:**
                - Class 1 as class 0 -> possibility of missing diagnoses
                """)
        else:
            st.warning(f"""
            ⚠️ Confusion matrix image not found: {cm_file}

            Please place the PNG file in the `page_modules/` directory.
            """)

    with st.expander(
        "Accuracy & Loss Curves - Best Transfer Learning Option", expanded=False
    ):
        tab1 = st.tabs(["CNN8 + transfer6"])

        # Map model choice to image files
        loss_images = {
            "CNN8 + transfer6": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_loss.png"
        }
        accuracy_images = {
            "CNN8 + transfer6": "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_accuracy.png"
        }

        # CNN8 + transfer6 Tab
        with tab1[0]:
            loss_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_loss.png"
            accuracy_file = "cnn8_sm_lrexpdec1e-3_earlystop_bs512_epoch52_lastresblockunfrozen_transfer6_sm_lrexpdec1e-3_earlystop_bs128_accuracy.png"

            loss_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", loss_file
            )
            accuracy_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_9", accuracy_file
            )

            if os.path.exists(loss_path) and os.path.exists(accuracy_path):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(loss_path, use_container_width=True)
                with col2:
                    st.image(accuracy_path, use_container_width=True)
            else:
                st.error("⚠️ Training history images not found for CNN8 + transfer6.")

    st.header("Live Predictions")

    st.markdown("""
    Explore how the transfer learning model performs on individual test samples from the PTB dataset.
    Select a sample to see the ECG signal and prediction results for binary classification.
    """)

    # Load test data and precomputed predictions
    X_test_path = os.path.join(
        os.path.dirname(__file__), "..", "images", "page_9", "X_ptb_test.csv"
    )
    y_test_path = os.path.join(
        os.path.dirname(__file__), "..", "images", "page_9", "y_ptb_test.csv"
    )
    predictions_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "images",
        "page_9",
        "precomputed_predictions_transfer.csv",
    )

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
                    prediction_probs = pred_row[
                        ["prob_class_0", "prob_class_1"]
                    ].values[0]

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**True Label:**")
                        st.info(f"Class {y_sample}: {class_names[y_sample]}")

                    with col2:
                        st.markdown("**Predicted Label:**")
                        if predicted_class == y_sample:
                            st.success(
                                f"Class {predicted_class}: {class_names[predicted_class]} ✓"
                            )
                        else:
                            st.error(
                                f"Class {predicted_class}: {class_names[predicted_class]} ✗"
                            )

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
        st.error("""
        ⚠️ Required files not found.

        Please add the following files to page_modules/:
        - X_ptb_test.csv
        - y_ptb_test.csv
        - precomputed_predictions_transfer.csv
        """)

    st.subheader("Citations")
    with st.expander("", expanded=False):
        st.write("""
            [1] Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); doi: 10.3389/fphys.2023.1246746

            [2] ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee,  S. Fazeli, M. Sarrafzadeh (2018); CoRR; doi: 10.48550/arXiv.1805.00794

            [3] https://www.datasci.com/solutions/cardiovascular/ecg-research

            [4] ECG-based heartbeat classification for arrhythmia detection: A survey;  E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, D. Menotti (2015); Computer Methods and Programs in Biomedicine; doi: 10.1016/j.cmpb.2015.12.008

            [5] Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M, Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; doi:10.1016/j.compbiomed.2020.103726
            """)
