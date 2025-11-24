"""
Page 1: Presentation of the subject, the problem and the issues
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Introduction")
    st.markdown("---")
    
    st.header("Subject, Problem and Issues")
    
    # TODO by Kiki: Add content about:
    # - The heart and blood circulation
    # - Heart diseases as leading cause of death
    # - ECG as standard tool for diagnosis
    # - Manual analysis challenges (time-consuming, error-prone)
    # - Need for automated systems
    
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Add introduction text about heart function and ECG
    - Explain the problem: manual ECG analysis challenges
    - Present the motivation for automated classification systems
    """)
    
    st.header("Project Goals")
    
    # TODO by Kiki: Add project goals
    st.subheader("Content Placeholder")
    st.write("""
    **TODO by Kiki:**
    - Arrhythmia Classification: 5 types of heartbeats using MIT-BIH dataset
    - Myocardial Infarction Detection: Healthy vs abnormal using PTB dataset
    """)

