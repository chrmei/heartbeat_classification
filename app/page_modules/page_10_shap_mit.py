"""
Page 10: SHAP Analysis on MIT - best models
Loading of SHAP graphics (pre-created)
Todo by Julia
"""

import streamlit as st


def render():
    st.title("SHAP Analysis - MIT Dataset")
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
    
    st.header("Best Model: CNN8")
    
    # TODO by Julia: Add model information
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Display information about CNN8 model
    - Explain why this model was selected for SHAP analysis
    """)
    
    st.header("Class 0: Normal (N)")
    
    # TODO by Julia: Add Class 0 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 0
    - 11 out of 20 most important features before timestep 20
    - 6 out of 20 most important features between timestep 90 and 110
    - Very early timesteps and timesteps around middle of signal are important
    - Overlay SHAP values and ECG signal for 3 examples
    - R peak around feature 70-90 is of very high importance
    """)
    
    st.header("Class 1: Atrial Premature (S)")
    
    # TODO by Julia: Add Class 1 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 1
    - 15 out of 20 most important features before timestep 20
    - 3 out of 20 most important features between timestep 90 and 110
    - Very early timesteps are important
    - Overlay SHAP values and ECG signal for 3 examples
    - 4 important peaks: feature 0-10 (R peak), 25-50 (T wave), 75-110 (R peak), peak after second R peak (T wave)
    """)
    
    st.header("Class 2: Premature Ventricular Contraction (V)")
    
    # TODO by Julia: Add Class 2 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 2
    - 12 out of 20 most important features before timestep 20
    - 7 out of 20 most important features between timestep 20 and 30
    - Very early and early timesteps are important
    - Overlay SHAP values and ECG signal for 3 examples
    - Very early features important: normally position of R peak, but no R peak present
    """)
    
    st.header("Class 3: Fusion Ventricular and Normal (F)")
    
    # TODO by Julia: Add Class 3 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 3
    - 8 out of 20 most important features before timestep 20
    - 3 out of 20 most important features between timestep 60 and 70
    - 5 out of 20 most important features between timestep 90 and 110
    - Overlay SHAP values and ECG signal for 3 examples
    - R peak important, but distribution of high SHAP values is spread and complex
    - Model seems to have exploited zero-padded values (R-R interval)
    """)
    
    st.header("Class 4: Not Classifiable/Fusion of Paced and Normal (Q)")
    
    # TODO by Julia: Add Class 4 SHAP analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for Class 4
    - 7 out of 20 most important features before timestep 20
    - 10 out of 20 most important features between timestep 90 and 110
    - Very early timesteps and timesteps around middle of signal are important
    - Overlay SHAP values and ECG signal for 3 examples
    - High importance: second R peak (around feature 100) - different location than class 0
    """)
    
    st.header("Misclassification Analysis")
    
    # TODO by Julia: Add misclassification analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - Load and display SHAP graphics for misclassifications
    - Largest misclassification: true class 1, predicted class 0
    - Features around timesteps 5, 30, and 50 pushed classification towards class 0
    - Features around timestep 100 pushed classification towards class 0
    - Explain why misclassification occurred
    """)
    
    st.header("SHAP Graphics Location")
    
    # TODO by Julia: Add information about SHAP graphics location
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Julia:**
    - SHAP graphics should be loaded from: reports/interpretability/SHAP_MIT/
    - Load pre-created SHAP visualizations
    - Display them in an organized manner
    """)

