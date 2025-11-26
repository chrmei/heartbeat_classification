"""
Page 7: DL intro
Description
Todo by Julia
"""

import streamlit as st


def render():
    st.title("Deep Learning Models")
    st.markdown("---")

    st.header("Introduction")

    st.subheader("Workflow")
    with st.expander("", expanded=False):
        st.write("""
            - test different DL model types -> goal: find best option for MIT dataset
            - optimization of hyperparameters and training procedure for MIT dataset
            - transfer learning for PTB dataset: use pretrained model and retrain for PTB classification task
            """)

    st.subheader("Tested DL models for MIT dataset")
    with st.expander("", expanded=False):
        st.write("""
            - Dense Neural Network (DNN) [1]
                * architecture inspired by lessons
            - Convolutional Neural Network (CNN) [1]
                * CNN architectures used in [2, 5] rebuilt
            - Long Short-Term Memory (LSTM)
                * tested because of [1, 5]
            """)

    st.subheader("Training Setup")
    with st.expander("", expanded=False):
        st.write("""
        - oversampling technique for all DL training procedures: **SMOTE**
        - training procedure: **minimization of loss function** -> class sensitive
        - **early stopping** when training loss did not further decreased to prevent overfitting
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
