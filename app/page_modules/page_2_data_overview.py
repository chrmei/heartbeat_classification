"""
Page 2: Presentation of the data (volume, architecture, etc.)
Data analysis using DataVisualization figures
Description and justification of the pre-processing carried out
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Data Overview")
    st.markdown("---")
    
    st.header("Dataset Information")
    
    # TODO by Kiki: Add dataset information
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Supervised problem: labeled data
    - Classification problem: arrhythmia 5 classes, MI 2 classes
    - 188 columns -> time points, final column: classification result
    - Each row represents 1.2 heartbeats
    - Show example rows from MIT and PTB
    """)
    
    st.header("MIT-BIH Dataset")
    
    # TODO by Kiki: Add MIT dataset details
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - 5 categories (0: normal N, 1: atrial premature S, 2: premature ventricular contraction V, 
      3: fusion ventricular and normal F, 4: not classifiable/fusion of paced and normal Q)
    - 109446 samples in total
    - Numerical, normalized and preprocessed data
    - No missing values, no duplicates
    - Imbalanced class distribution (especially classes 3 and 1 underrepresented)
    - Load data and create visualizations
    """)
    
    st.header("PTB Dataset")
    
    # TODO by Kiki: Add PTB dataset details
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - 2 categories (0: normal, 1: abnormal)
    - 14552 samples in total (14545 after removing duplicates)
    - Numerical, normalized and preprocessed data
    - No missing values
    - Imbalanced class distribution
    - Load data and create visualizations
    """)
    
    st.header("Data Visualization")
    
    # TODO by Kiki: Add data visualization figures
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Create and display data visualization figures
    - Class distribution plots
    - Sample ECG signal visualizations
    - Data statistics and summaries
    """)
    
    st.header("Pre-Processing Description")
    
    # TODO by Kiki: Add pre-processing justification
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Describe pre-processing steps
    - Justify pre-processing choices
    - Show pre-processed data examples
    """)

