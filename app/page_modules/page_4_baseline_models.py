"""
Page 4: Baseline Models - presentation of trained models and their results
Todo by Christian
"""

import streamlit as st


def render():
    st.title("Baseline Models")
    st.markdown("---")
    
    st.header("Modeling Workflow")
    
    # TODO by Christian: Add modeling workflow overview
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Baseline models: test baseline models and find best option and appropriate oversampling technique
    - Using RandomizedSearch, GridSearch (MIT, PTB) and LazyClassifier (PTB)
    - Overview of the workflow
    """)
    
    st.header("Tested Baseline Models")
    
    # TODO by Christian: Add table of tested baseline models
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Create table showing all tested baseline models:
      * Logistic Regression (LR)
      * Linear Discriminant (LDA)
      * K nearest neighbours (KNN)
      * Decision Tree (DT)
      * Random Forest (RF)
      * Support Vector Machine (SVM)
      * Extreme Gradient Boosting (XGB)
      * Artificial Neural Network (ANN)
    - Include descriptions/characteristics for each model
    """)
    
    st.header("Model Loading")
    
    # TODO by Christian: Add model loading functionality
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Load baseline models from saved files
    - Models should be loaded from: models/MIT_02_01_baseline_models_randomized_search_no_sampling/
    - Models should be loaded from: models/MIT_02_02_baseline_models_randomized_search_sampling/
    - Display model metadata
    """)
    
    st.header("Oversampling Technique")
    
    # TODO by Christian: Add oversampling information
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Christian:**
    - Explain SMOTE as appropriate oversampling technique
    - Generation of new synthetic samples for underrepresented classes
    - No generation of duplicates
    """)

