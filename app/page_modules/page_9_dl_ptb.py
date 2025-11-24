"""
Page 9: Deep Learning Models PTB - Transfer Learning
Same as Page 8 but as "transfer learning" on PTB
Todo by Julia
"""

import streamlit as st


def render():
    st.title("Deep Learning Models - PTB Dataset (Transfer Learning)")
    st.markdown("---")

    st.header("Transfer Learning Description")

    # TODO by Julia: Add transfer learning description
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Explain transfer learning approach
    - Retrain model found for MIT dataset
    - Best option: freeze majority of MIT model, with small last part unfrozen
    - Add classification model adapted to binary classification problem with dropout layers
    - Show architecture details
    """)

    st.header("Model Performance")

    # TODO by Julia: Add classification report
    st.subheader("Classification Report")
    st.write("""
    **TODO by Julia:**
    - Display classification report with metrics:
      * Accuracy: 0.9842
      * F1 Score: 0.9805
      * Precision: 0.9751
      * Recall: 0.9864
    - Per-class metrics (Precision, Recall, F1-Scores) for both classes
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
    - Explain clinical significance (number of misclassifications is very small)
    """)

    st.header("Training History")

    # TODO by Julia: Add accuracy and loss curves
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Display accuracy curves (training vs validation)
    - Display loss curves (training vs validation)
    - Show training history plots for transfer learning
    - Explain training procedure
    """)

    st.header("Live Predictions")

    # TODO by Julia: Add live prediction functionality
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load PTB test data
    - User can choose:
      * Random sample
      * Specific row number
      * Filter by class (normal/abnormal)
    - Display ECG signal visualization
    - Show model prediction
    - Show prediction probabilities for both classes
    - Load and use transfer learning model (CNN8 + transfer6)
    """)

    st.header("Best Model: CNN8 + Transfer")

    # TODO by Julia: Add best model details
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Show transfer learning configuration
    - Display model architecture details
    - Explain why this approach was selected
    - Compare with baseline results
    """)
