"""
Page 12: SHAP Analysis on PTB
Same as Page 10 but on PTB
Todo by Julia
"""

import streamlit as st


def render():
    st.title("SHAP Analysis - PTB")
    st.markdown("---")

    st.header("SHAP Overview")

    # TODO by Julia: Add SHAP explanation
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Explain SHAP (SHapley Additive exPlanations)
    - Purpose: identify important ECG signal parts
    - How SHAP values help understand model decisions
    """)

    st.header("Best Model: CNN8 + Transfer Learning")

    # TODO by Julia: Add model information
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Display information about CNN8 + transfer6 model
    - Explain why this model was selected for SHAP analysis
    """)

    st.header("Class 0: Normal")

    # TODO by Julia: Add Class 0 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 0
    - 6 out of 20 most important features before timestep 20
    - 12 out of 20 most important features distributed between 20 to 40
    - Very early and early timesteps have biggest importance
    - Overlay SHAP values and ECG signal for 3 examples
    - First and second R peaks are very important
    - Timesteps around feature 30 are important
    - High SHAP values distributed between two R peaks
    """)

    st.header("Class 1: Abnormal")

    # TODO by Julia: Add Class 1 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 1
    - Overlay SHAP values and ECG signal for 3 examples
    - Region of second R peak and beginning of signal have strong influence
    - High SHAP values distributed between beginning and second R peak
    - Small wave before second R peak contributed to decision towards class 1
    - Some examples show two regions of high SHAP values (around timestep 20 and 50)
    """)

    st.header("Misclassification Analysis")

    # TODO by Julia: Add misclassification analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for misclassifications
    - Largest misclassification: true class 1, predicted class 0
    - Beginning of ECG signal pushed classification strongly towards class 0
    - Features around timestep 35 and 90 pushed classification towards class 0
    - Second R peak around timestep 95 pushed classification decision towards class 0
    - Evidence for class 1 was very underrepresented
    - Explain why misclassification occurred
    """)

    st.header("SHAP Graphics Location")

    # TODO by Julia: Add information about SHAP graphics location
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - SHAP graphics should be loaded from: reports/interpretability/SHAP_PTB/
    - Load pre-created SHAP visualizations
    - Display them in an organized manner
    """)
