"""
Page 6: Results baseline models PTB
Same as Page 5 but on PTB
Todo by Christian
"""

import streamlit as st


def render():
    st.title("Baseline Models Results - PTB Dataset")
    st.markdown("---")
    
    st.header("Model Performance")
    
    # TODO by Christian: Add classification report
    st.subheader("Classification Report")
    st.write("""
    **TODO by Christian:**
    - Display classification report with metrics:
      * Accuracy: 0.9739
      * F1: 0.9730
      * Precision: 0.9739
      * Recall: 0.9721
    - Per-class metrics (Precision, Recall, F1-Scores) for both classes
    - Make it interactive/visual
    """)
    
    st.header("Confusion Matrix")
    
    # TODO by Christian: Add interactive confusion matrix
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Display interactive confusion matrix
    - Show misclassification patterns
    - Highlight largest misclassification: true class 0, predicted class 1
    - Explain clinical significance (not extremely critical but important to check)
    """)
    
    st.header("Live Predictions")
    
    # TODO by Christian: Add live prediction functionality
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Load PTB test data
    - User can choose:
      * Random sample
      * Specific row number
      * Filter by class (normal/abnormal)
    - Display ECG signal visualization
    - Show model prediction
    - Show prediction probabilities for both classes
    - Allow user to select different models to compare
    """)
    
    st.header("Best Model: XGBoost")
    
    # TODO by Christian: Add best model details
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Show best parameters identified using GridSearch
    - Display model performance metrics
    - Explain LazyClassifier selection process
    - Explain why XGBoost was selected as best model
    """)

