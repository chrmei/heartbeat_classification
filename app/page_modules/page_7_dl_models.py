"""
Page 7: DL intro
Description
Todo by Julia
"""

import streamlit as st


def render():
    st.title("Deep Learning Models")
    st.markdown("---")

    st.header("Model Description")

    # TODO by Julia: Add DL model description
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    #- Describe tested DL models: DNN, CNN, LSTM
    #- Explain CNN architecture (inspired by [2])
    #- Added dropout and batch normalization layers
    #- Show architecture details/table
    #- Explain training procedure (SMOTE, early stopping, loss function)
    - oversampling technique: SMOTE
    - training procedure: minimization of loss function -> class sensitive
    - early stopping when training loss did not further decreased
    """)
