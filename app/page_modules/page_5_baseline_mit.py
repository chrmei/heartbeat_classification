"""
Page 5: Results baseline models MIT
Include "prediction" on live examples (user can choose randomly or a certain row and select a class)
Classification Report, Confusion Matrix (interactive)
Todo by Christian
"""

import streamlit as st


def render():
    st.title("Baseline Models Results - MIT Dataset")
    st.markdown("---")

    st.header("Model Performance")

    # TODO by Christian: Add classification report
    st.subheader("Classification Report")
    st.write(
        """
    **TODO by Christian:**
    - Display classification report with metrics:
      * Accuracy: 0.9823
      * F1: 0.9096
      * Precision: 0.9223
      * Recall: 0.8981
    - Per-class metrics (Precision, Recall, F1-Scores) for all 5 classes
    - Make it interactive/visual
    """
    )

    st.header("Confusion Matrix")

    # TODO by Christian: Add interactive confusion matrix
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Display interactive confusion matrix
    - Show misclassification patterns
    - Highlight largest misclassification: true class 1, predicted class 0
    - Explain clinical significance
    """
    )

    st.header("Live Predictions")

    # TODO by Christian: Add live prediction functionality
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Load test data
    - User can choose:
      * Random sample
      * Specific row number
      * Filter by class
    - Display ECG signal visualization
    - Show model prediction
    - Show prediction probabilities for all classes
    - Allow user to select different models to compare
    """
    )

    st.header("Best Model: XGBoost")

    # TODO by Christian: Add best model details
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Show best parameters identified using GridSearch
    - Display model performance metrics
    - Explain why XGBoost was selected as best model
    """
    )

    st.header("Model Loading")

    # TODO by Christian: Add model loading functionality
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Load baseline models from saved files
    - Models should be loaded from: models/MIT_02_01_baseline_models_randomized_search_no_sampling/
    - Models should be loaded from: models/MIT_02_02_baseline_models_randomized_search_sampling/
    - Display model metadata
    """
    )

    st.header("Oversampling Technique")

    # TODO by Christian: Add oversampling information
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Explain SMOTE as appropriate oversampling technique
    - Generation of new synthetic samples for underrepresented classes
    - No generation of duplicates
    """
    )
