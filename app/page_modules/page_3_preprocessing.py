"""
Page 3: Pre-Processing discussion around extreme RR-Distances
First MIT, then PTB
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Pre-Processing: RR-Distance Analysis")
    st.markdown("---")
    
    st.header("RR Distance Overview")
    
    # TODO by Kiki: Add RR distance explanation
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Explain RR distance: duration of heartbeat
    - Each row shows data of 1.2R, padded with zeros
    - How to extract R by identifying index at start of zero padding
    """)
    
    st.header("MIT-BIH: Extreme RR-Distance Analysis")
    
    # TODO by Kiki: Add MIT RR distance analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Calculate R for each heartbeat of each class
    - Identify extreme values
    - Analyze corresponding ECG data
    - Class 0: ECG data of example lower extreme values look NOT normal
    - Class 0: For upper extreme values ECG data seem okay
    - For other classes: hard to say if extreme values are abnormal (could be due to disease)
    - Class 3/4: high amount of extreme values for R
    - Show visualizations and plots
    """)
    
    st.header("PTB: Extreme RR-Distance Analysis")
    
    # TODO by Kiki: Add PTB RR distance analysis
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Calculate R for each heartbeat of each class
    - Identify extreme values
    - Analyze corresponding ECG data
    - Class 0: Some ECG data of example lower extreme values look NOT normal
    - For other class: hard to say if extreme values are abnormal (could be due to disease)
    - Show visualizations and plots
    """)
    
    st.header("Important Findings")
    
    # TODO by Kiki: Add important findings
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Important information for modeling phase: check if performance increases when deleting extreme values for class 0
    - Important information for outlook: Medical experts should check classification of samples belonging to extreme values
    """)

