"""
Page 4: Baseline Models - presentation of trained models and their results
"""

import streamlit as st


def render():
    st.title("General Modeling Overview")

    st.write(
        """
        - Test different models and find best option and appropriate oversampling technique.
        - Then: Hyperparameter tuning:
        - For MIT dataset: 
            - 1. RandomizedSearch to identify promising models.
            - 2. GridSearch for the "best" 3 models.
        - For PTB dataset: LazyClassifier and GridSearch for the best identified models. model.
        """
    )

    tab1, tab2, tab3 = st.tabs(["Baseline Models", "Deep Learning Models", "Sampling Methods"])

    # -------------------------------------------------------------
    # TAB 1 – BASELINE MODELS
    # -------------------------------------------------------------
    with tab1:

        st.subheader("Tested Baseline Models")

        models = [
            ("🧮 Logistic Regression (LR)", "Fast and simple; good for workflow validation."),
            (
                "📐 Linear Discriminant Analysis (LDA)",
                "Historically used for arrhythmia classification [1].",
            ),
            (
                "👥 K-Nearest Neighbours (KNN)",
                "Non-parametric; used in earlier arrhythmia studies [1].",
            ),
            ("🌳 Decision Tree (DT)", "Interpretable; useful for feature-importance insights."),
            ("🌲 Random Forest (RF)", "Ensemble of trees; strong baseline for tabular data."),
            (
                "🧱 Support Vector Machine (SVM)",
                "Well-established model for arrhythmia classification [1].",
            ),
            (
                "⚡ Extreme Gradient Boosting (XGB)",
                "High-performance ensemble; robust to imbalance.",
            ),
            (
                "🧠 Artificial Neural Network (ANN)",
                "Simple feed-forward DL architecture used in arrhythmia classification [1].",
            ),
        ]

        cols = st.columns(3)

        for i, (title, desc) in enumerate(models):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #444;padding:12px;border-radius:8px;
                                margin-bottom:12px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px;color:#ccc;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------
    # TAB 2 – DEEP LEARNING MODELS
    # -------------------------------------------------------------
    with tab2:

        st.subheader("Tested Deep Learning Models")

        dl_models = [
            (
                "🧠 Dense Neural Networks (DNN)",
                "Fully connected layers; classical deep-learning baseline for ECG analysis [1].",
            ),
            (
                "📡 Convolutional Neural Networks (CNN)",
                "State-of-the-art for ECG classification; leverage spatial & temporal structure [2]. "
                "Automatically learn features from raw signals (no handcrafted features).",
            ),
            (
                "🏗️ Rebuilt CNN Architecture (as in [2, 5])",
                "Custom reconstruction of published CNN topology; replicates experimental setup and benchmarks.",
            ),
            (
                "🔁 Long Short-Term Memory (LSTM)",
                "Sequence model capturing long-range temporal dependencies; used in ECG classification [1, 5].",
            ),
        ]

        cols = st.columns(3)

        for i, (title, desc) in enumerate(dl_models):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #444;padding:12px;border-radius:8px;
                                margin-bottom:12px;min-height:140px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px;color:#ccc;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab3:
        st.subheader("Sampling Variants")

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
                    <div style="border:1px solid #444;padding:12px;border-radius:8px;
                                margin-bottom:12px;min-height:150px;">
                        <strong>{title}</strong><br>
                        <span style="font-size:14px;color:#ccc;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
