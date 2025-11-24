"""
Page 8: Results on Deep Learning Models MIT Dataset
Description, Classification Report, Confusion Matrix, Live-Prediction
Accuracy / Loss Curves
Todo by Julia
"""

import os

import pandas as pd
import streamlit as st


def render():
    st.title("Deep Learning Models - MIT Dataset")
    st.markdown("---")

    st.header("Find Best DL Model Option")

    st.subheader("Tested Models")
    with st.expander("Result Table", expanded=False):
        # Load CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "dl_1.csv")
        df = pd.read_csv(csv_path, sep=";", index_col=0)
        st.dataframe(df)

    with st.expander("Best DL Options - Top 3 Models", expanded=False):
        st.write("""
            1. **CNN7**:
                * Model architecture from [2] with added batch normalization layers
                * Five residual blocks, followed by fully connected layers
                * Batch normalization layers after each convolutional layer
                * F1 score on test data: **0.9117**
            2. **CNN8**:
                * Model architecture from [2] with added dropout layers
                * Five residual blocks, followed by fully connected layers
                * Dropout layers at the end of each residual block (0.1)
                * F1 score on test data: **0.8996**
            3. **CNN1**:
                * Model architecture inspired by lessons
                * 3 convolutional blocks followed by dense layers
                * F1 score on test data: **0.8834**
            """)

    with st.expander("Model Architecture - Top 3 Models", expanded=False):
        model_choice = st.selectbox("Select model:", ["CNN7", "CNN8", "CNN1"])

        # Map model choice to summary file
        summary_files = {
            "CNN7": "cnn7_summary.txt",
            "CNN8": "cnn8_summary.txt",
            "CNN1": "cnn1_summary.txt",
        }

        summary_file = summary_files[model_choice]
        summary_path = os.path.join(os.path.dirname(__file__), summary_file)

        # Check if summary file exists
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_text = f.read()
            # Display in a code block with horizontal scrolling
            st.code(summary_text, language="text")
        else:
            st.error(f"""
            ⚠️ Model summary file not found: {summary_file}
            """)

    with st.expander("Accuracy & Loss Curves - Top 3 Models", expanded=False):
        model_choice_history = st.selectbox(
            "Select model to view training history:",
            ["CNN7", "CNN8", "CNN1"],
            key="history_selector",
        )

        # Map model choice to image files (adjust paths as needed)
        loss_images = {
            "CNN7": "cnn7_sm_lrexpdec5e-4_earlystop_bs512_epoch_loss.png",
            "CNN8": "cnn8_sm_lrexpdec5e-4_earlystop_bs512_loss.png",
            "CNN1": "cnn1_sm_lrexpdec5e-4_earlystop_bs512_loss.png",
        }

        accuracy_images = {
            "CNN7": "cnn7_sm_lrexpdec5e-4_earlystop_bs512_epoch_accuracy.png",
            "CNN8": "cnn8_sm_lrexpdecpaper1e-3_earlystop_bs512_accuracy.png",
            "CNN1": "cnn1_sm_lrexpdec5e-4_earlystop_bs512_accuracy.png",
        }

        loss_file = loss_images[model_choice_history]
        accuracy_file = accuracy_images[model_choice_history]

        loss_path = os.path.join(os.path.dirname(__file__), loss_file)
        accuracy_path = os.path.join(os.path.dirname(__file__), accuracy_file)

        # Check if files exist and display them
        if os.path.exists(loss_path) and os.path.exists(accuracy_path):
            # Display images side by side: Loss left, Accuracy right
            col1, col2 = st.columns(2)

            with col1:
                st.image(loss_path, use_container_width=True)

            with col2:
                st.image(accuracy_path, use_container_width=True)
        elif os.path.exists(loss_path):
            st.image(loss_path, use_container_width=True)
            st.warning(f"Accuracy curve not found: {accuracy_file}")
        elif os.path.exists(accuracy_path):
            st.image(accuracy_path, use_container_width=True)
            st.warning(f"Loss curve not found: {loss_file}")
        else:
            st.error(f"""
            ⚠️ Training history images not found for {model_choice_history}.

            Expected files:
            - {loss_file}
            - {accuracy_file}

            Please place the PNG files in the `page_modules/` directory.
            """)

    st.subheader("Optimization of Models with Architecture from [2] - CNN7 and CNN8")

    st.header("Live Predictions")

    # TODO by Julia: Add live prediction functionality
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load test data
    - User can choose:
      * Random sample
      * Specific row number
      * Filter by class
    - Display ECG signal visualization
    - Show model prediction
    - Show prediction probabilities for all classes
    - Load and use best CNN model (CNN8)
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
