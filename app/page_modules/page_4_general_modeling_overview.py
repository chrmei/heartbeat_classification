"""
Page 4: Baseline Models - presentation of trained models and their results
"""

import streamlit as st
from page_modules.styles import COLORS


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🧪 General Modeling Overview</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Modeling approach summary in a styled box
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1.5rem;">
            <h4 style="color: white; margin-top: 0;">📋 Modeling Approach</h4>
            <ul style="opacity: 0.95; margin-bottom: 0;">
                <li>Test different models and find best option and appropriate oversampling technique</li>
                <li>Hyperparameter tuning strategy:
                    <ul style="margin-top: 0.25rem;">
                        <li><strong>MIT dataset:</strong> RandomizedSearch → GridSearch for top 3 models</li>
                        <li><strong>PTB dataset:</strong> LazyClassifier → GridSearch for best identified models</li>
                    </ul>
                </li>
                <li><strong>F1 Score</strong> as main metric due to class imbalance</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Baseline Models", "Sampling Methods", "Deep Learning Models" ])

    # -------------------------------------------------------------
    # TAB 1 – BASELINE MODELS
    # -------------------------------------------------------------
    with tab1:

        st.markdown(
            '<div class="hero-container" style="padding: 1.5rem;">'
            '<div class="hero-title" style="font-size: 1.8rem;">🧮 Tested Baseline Models</div>'
            '</div>',
            unsafe_allow_html=True
        )

        models = [
            ("🧮 Logistic Regression (LR)", "Fast and simple; good for workflow validation."),
            (
                "📐 Linear Discriminant Analysis (LDA)",
                "Historically used for arrhythmia classification [5].",
            ),
            (
                "👥 K-Nearest Neighbours (KNN)",
                "Non-parametric; used in earlier arrhythmia studies [5].",
            ),
            ("🌳 Decision Tree (DT)", "Interpretable; useful for feature-importance insights."),
            ("🌲 Random Forest (RF)", "Ensemble of trees; strong baseline for tabular data."),
            (
                "🧱 Support Vector Machine (SVM)",
                "Well-established model for arrhythmia classification [5].",
            ),
            (
                "⚡ Extreme Gradient Boosting (XGB)",
                "High-performance ensemble; robust to imbalance.",
            ),
            (
                "🧠 Artificial Neural Network (ANN)",
                "Simple feed-forward DL architecture used in arrhythmia classification [5].",
            ),
        ]

        cols = st.columns(3)

        for i, (title, desc) in enumerate(models):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_lighter']} 0%, #A8DADC 100%);
                                padding: 12px; border-radius: 8px; color: {COLORS['clinical_blue']};
                                box-shadow: 0 4px 15px rgba(168, 218, 220, 0.4); margin-bottom: 12px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px; color: {COLORS['text_secondary']};">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



    with tab2:
        st.markdown(
            '<div class="hero-container" style="padding: 1.5rem;">'
            '<div class="hero-title" style="font-size: 1.8rem;">⚖️ Sampling Variants</div>'
            '</div>',
            unsafe_allow_html=True
        )

        sampling_variants = [
            ("📦 Raw Training Data", "Unmodified dataset used as baseline reference."),
            (
                "🔁 RandomOversampler",
                "Random duplication of minority-class samples to balance the distribution.",
            ),
            (
                "🧪 SMOTE",
                "Generates synthetic samples for underrepresented classes based on nearest neighbors.",
            ),
            (
                "🎯 ADASYN",
                "Extension of SMOTE; focuses synthetic sample generation on harder-to-learn minority samples.",
            ),
            (
                "🧹 SMOTETomek",
                "SMOTE oversampling combined with Tomek-links removal for cleaner class boundaries.",
            ),
            ("🧼 SMOTEENN", "SMOTE oversampling combined with Edited Nearest Neighbor cleaning."),
        ]

        cols = st.columns(3)

        for i, (title, desc) in enumerate(sampling_variants):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_lighter']} 0%, #A8DADC 100%);
                                padding: 12px; border-radius: 8px; color: {COLORS['clinical_blue']};
                                box-shadow: 0 4px 15px rgba(168, 218, 220, 0.4); margin-bottom: 12px; min-height: 150px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px; color: {COLORS['text_secondary']};">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # -------------------------------------------------------------
    # TAB 2 – DEEP LEARNING MODELS
    # -------------------------------------------------------------
    with tab3:

        st.markdown(
            '<div class="hero-container" style="padding: 1.5rem;">'
            '<div class="hero-title" style="font-size: 1.8rem;">🧠 Tested Deep Learning Models</div>'
            '</div>',
            unsafe_allow_html=True
        )

        dl_models = [
            (
                "🧠 Dense Neural Networks (DNN)",
                "Fully connected layers; classical deep-learning baseline for ECG analysis [5].",
            ),
            (
                "📡 Convolutional Neural Networks (CNN)",
                "State-of-the-art for ECG classification; leverage spatial & temporal structure [3]. "
                "Automatically learn features from raw signals (no handcrafted features).",
            ),
            (
                "🏗️ Rebuilt CNN Architecture (as in [3, 6])",
                "Custom reconstruction of published CNN topology; replicates experimental setup and benchmarks.",
            ),
            (
                "🔁 Long Short-Term Memory (LSTM)",
                "Sequence model capturing long-range temporal dependencies; used in ECG classification [5, 6].",
            ),
        ]

        # Create 2x2 grid layout
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)

        # First row: first 2 models
        for i in range(2):
            with row1_cols[i]:
                title, desc = dl_models[i]
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_lighter']} 0%, #A8DADC 100%);
                                padding: 12px; border-radius: 8px; color: {COLORS['clinical_blue']};
                                box-shadow: 0 4px 15px rgba(168, 218, 220, 0.4); margin-bottom: 12px; min-height: 140px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px; color: {COLORS['text_secondary']};">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Second row: last 2 models
        for i in range(2):
            with row2_cols[i]:
                title, desc = dl_models[i + 2]
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_lighter']} 0%, #A8DADC 100%);
                                padding: 12px; border-radius: 8px; color: {COLORS['clinical_blue']};
                                box-shadow: 0 4px 15px rgba(168, 218, 220, 0.4); margin-bottom: 12px; min-height: 140px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px; color: {COLORS['text_secondary']};">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
                    <strong>[5]</strong> Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); 
                    <a href="https://doi.org/10.3389/fphys.2023.1246746" style="color: {COLORS['clinical_blue_light']};">doi: 10.3389/fphys.2023.1246746</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>[6]</strong> Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M. Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; 
                    <a href="https://doi.org/10.1016/j.compbiomed.2020.103726" style="color: {COLORS['clinical_blue_light']};">doi:10.1016/j.compbiomed.2020.103726</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
