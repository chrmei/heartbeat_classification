"""
Page 7: Results on Deep Learning Models MIT Dataset
Description, Classification Report, Confusion Matrix, Live-Prediction
Accuracy / Loss Curves
Todo by Julia
"""

import streamlit as st


def render():
    st.title("Deep Learning Models - MIT Dataset")
    st.markdown("---")
    
    st.header("Model Description")
    
    # TODO by Julia: Add DL model description
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Describe tested DL models: DNN, CNN, LSTM
    - Explain CNN architecture (inspired by [2])
    - Added dropout and batch normalization layers
    - Show architecture details/table
    - Explain training procedure (SMOTE, early stopping, loss function)
    """)
    
    st.header("Model Performance")
    
    # TODO by Julia: Add classification report
    st.subheader("Classification Report")
    st.write("""
    **TODO by Julia:**
    - Display classification report with metrics:
      * Accuracy: 0.9851
      * F1 Score: 0.9236
      * Precision: 0.9062
      * Recall: 0.9424
    - Per-class metrics (Precision, Recall, F1-Scores) for all 5 classes
    - Make it interactive/visual
    """)
    
    st.header("Confusion Matrix")
    
    # TODO by Julia: Add confusion matrix
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Display confusion matrix
    - Show misclassification patterns
    - Highlight largest misclassification: true class 1, predicted class 0
    - Explain clinical significance
    """)
    
    st.header("Training History")
    
    # TODO by Julia: Add accuracy and loss curves
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Display accuracy curves (training vs validation)
    - Display loss curves (training vs validation)
    - Show training history plots
    - Explain early stopping and convergence
    """)
    
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
    
    st.header("Best Model: CNN8")
    
    # TODO by Julia: Add best model details
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Show optimized training procedure (batch size, learning rate schedules)
    - Display model architecture details
    - Explain why CNN8 was selected as best model
    """)

