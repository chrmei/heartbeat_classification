"""
Page 7: DL intro
Description
Todo by Julia
"""

import streamlit as st


def render():
    st.title("7: Deep Learning Models")
    st.markdown("---")

    st.header("Introduction")

    with st.expander("Workflow", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**MIT Dataset:**")
            st.write(
                """
                - test different DL model types (inspiration from lessons, [2] and [5])
                    * goal: find best option for arrhythmia classification
                - optimization of hyperparameters and training procedure
                    * goal: outperform models presented in publications for arrhythmia classification
            """
            )

        with col2:
            st.markdown("**PTB Dataset:**")
            st.write(
                """
            - transfer learning: use pretrained MIT model (best option found) and retrain for PTB classification task
                * goal: generate DL model for MI detection and benefit from MIT modeling phase
            - optimization of hyperparameters and training procedure
                * goal: outperform models presented in publications for MI detection
            """
            )

        tab1, tab2 = st.tabs(["[2]", "[5]"])

        with tab1:
            import os

            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_7",
                "2018.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=400)
            else:
                st.error("⚠️ image not found")

        with tab2:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_7",
                "2020.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=400)
            else:
                st.error("⚠️ image not found")

    with st.expander("Tested DL models for MIT dataset", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Dense Neural Network (DNN) [1]:**")
            st.write(
                """
                - DNN architecture inspired by lessons
            """
            )

        with col2:
            st.markdown("**Convolutional Neural Network (CNN) [1]:**")
            st.write(
                """
                - CNN architectures presented in [2, 5] rebuilt
            """
            )

        with col3:
            st.markdown("**Long Short-Term Memory (LSTM) [1]:**")
            st.write(
                """
                - LSTM architectures presented in [5] rebuilt
            """
            )

        tab1, tab2, tab3 = st.tabs(["CNN [2]", "CNN [5]", "LSTM [5]"])

        with tab1:
            import os

            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_7",
                "cnn_2018.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.error("⚠️ CNN [2] image not found")

        with tab2:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_7",
                "cnn4_2020.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=400)
            else:
                st.error("⚠️ CNN [5] image not found")

        with tab3:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_7",
                "lstm_2020.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.error("⚠️ LSTM [5] image not found")

    with st.expander("Training Setup", expanded=False):
        st.write(
            """
        - oversampling technique for all DL training procedures: **SMOTE**
        - training procedure: **minimization of loss function** -> class sensitive
        - **early stopping** when training loss did not further decreased to prevent overfitting
        """
        )

    with st.expander("Citations", expanded=False):
        st.write(
            """
            [1] Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); doi: 10.3389/fphys.2023.1246746

            [2] ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee,  S. Fazeli, M. Sarrafzadeh (2018); CoRR; doi: 10.48550/arXiv.1805.00794

            [3] https://www.datasci.com/solutions/cardiovascular/ecg-research

            [4] ECG-based heartbeat classification for arrhythmia detection: A survey;  E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, D. Menotti (2015); Computer Methods and Programs in Biomedicine; doi: 10.1016/j.cmpb.2015.12.008

            [5] Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M, Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; doi:10.1016/j.compbiomed.2020.103726
            """
        )
