"""
Page 1: Presentation of the subject, the problem and the issues
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("1: Introduction")
    st.markdown("---")

    st.header("Context, Problem and Motivation")

    st.subheader("Context")
    st.markdown(
        """
        **Global Health Crisis**  
        **Heart diseases are one of the leading causes of mortality worldwide.**
        - Heart maintains blood circulation throughout the body, supplying organs with oxygen and nutrients.  
        - Detecting abnormalities early is essential to prevent severe outcomes.
            - **Arrhythmias:** irregular heartbeat  
            - **Myocardial infarction (MI):** heart attack, occurring when blood flow to the heart is blocked
        """
    )

    st.markdown(
        """
        **Electrocardiograms (ECGs)**
        - Gold standard for non-invasive heart monitoring.  
        - Record the electrical activity of the heart to identify irregularities.
        - An ECG signal captures the electrical patterns of each heartbeat through characteristic components
            - **P wave**: atrial contraction
            - **QRS complex**: ventricular contraction
            - **T wave**: ventricular relaxation
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.image("app/images/page_1/human_heart.svg", caption="Heart anatomy [1]", width=320)
    with c2:
        st.markdown(
            "<div style='height:95px'></div>", unsafe_allow_html=True
        )  # Spacer for alignment
        st.image("app/images/page_1/ECG_wave.jpg", caption="ECG waveform [2]", width=320)

    st.subheader("Problem")
    st.markdown(
        """
        Manual ECG interpretation is **time-consuming**, prone to **human error**, especially when dealing with large volumes of patient data.
        """
    )

    st.subheader("Motivation & Solution")
    st.markdown(
        """
        Developing an **automated Deep Learning heartbeat classification systems**, which serves as a crucial decision-support tool for medical professionals, enabling faster, more consistent, and scalable preliminary screening.
        """
    )

    st.divider()
    st.header("Project Goals")

    st.markdown(
        """
        This project focuses on automating two clinically relevant ECG classification tasks:
        """
    )
    st.subheader("1. Arrhythmia Classification (MIT-BIH Dataset – 5 Classes)")
    st.markdown(
        "Categorizing heartbeats into 5 distinct types (e.g., Normal, Ventricular, Fusion)."
    )

    st.subheader("2. Myocardial Infarction Detection (PTB Dataset – Binary)")
    st.markdown(
        """
        Distinguishing between healthy heartbeats and those indicating a heart attack (MI) using a **transfer-learning** approach.  
        """
    )

    st.markdown(
        """
        **Primary Objective:** To build a model that outperforms the 2018 benchmark study by Kachuee et al.[3] (Target Accuracy: >93.4% for MIT, >95.9% for PTB).
        """
    )

    st.divider()

    # --- Why Automation Matters ---
    st.header("Why Automation Matters")

    st.markdown(
        """
        - Cardiovascular diseases are the leading global cause of death.  
        - Early detection of arrhythmias and myocardial infarction is crucial for preventing severe outcomes.  
        - ECGs are widely available but challenging to analyze efficiently at scale.  
        - Automated systems can standardize interpretation, reduce diagnostic delays, and enhance patient outcomes.  
        """
    )

    st.markdown(
        """
        **References of Images**  
        [1] Wikipedia. Heart: https://en.wikipedia.org/wiki/Heart#/media/  
        [2] Pham BT, Le PT, Tai TC, Hsu YC, Li YH, Wang JC. Electrocardiogram Heartbeat Classification for Arrhythmias and Myocardial Infarction. Sensors (Basel). 2023 Mar 9;23(6):2993. doi: 10.3390/s23062993. PMID: 36991703; PMCID: PMC10051525.
        """
    )


if __name__ == "__main__":
    render()
