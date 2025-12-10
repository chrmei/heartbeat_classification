"""
Page 1: Presentation of the subject, the problem and the issues
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("1: Introduction")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📋 Context, Problem & Motivation",
        "🎯 Project Goals",
        "⚡ Why Automate this?"
    ])

    # --- Tab 1: Context, Problem & Motivation ---
    with tab1:
        with st.expander("🌍 Context: Global Health Crisis & ECGs", expanded=True):
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

        with st.expander("⚠️ Problem", expanded=True):
            st.markdown(
                """
                Manual ECG interpretation is **time-consuming**, prone to **human error**, especially when dealing with large volumes of patient data.
                """
            )

        with st.expander("💡 Motivation & Solution", expanded=True):
            st.markdown(
                """
                Developing an **automated Deep Learning heartbeat classification systems**, which serves as a crucial decision-support tool for medical professionals, enabling faster, more consistent, and scalable preliminary screening.
                """
            )

    # --- Tab 2: Project Goals ---
    with tab2:
        st.markdown(
            """
            This project focuses on automating two clinically relevant ECG classification tasks:
            """
        )

        with st.expander("🫀 Arrhythmia Classification (MIT-BIH Dataset – 5 Classes)", expanded=True):
            st.markdown(
                "Categorizing heartbeats into 5 distinct types (e.g., Normal, Ventricular, Fusion)."
            )

        with st.expander("❤️‍🩹 Myocardial Infarction Detection (PTB Dataset – Binary)", expanded=True):
            st.markdown(
                """
                Distinguishing between healthy heartbeats and those indicating a heart attack (MI) using a **transfer-learning** approach.  
                """
            )

        with st.expander("🏆 Primary Objective", expanded=True):
            st.markdown(
                """
                To build a model that outperforms the 2018 benchmark study by Kachuee et al.[3]  
                
                **Target Accuracy:**
                - MIT-BIH: >93.4%
                - PTB: >95.9%
                """
            )

    # --- Tab 3: Why Automate this? ---
    with tab3:
        with st.expander("🔑 Key Reasons for Automation", expanded=True):
            st.markdown(
                """
                - **Leading cause of death:** Cardiovascular diseases are the leading global cause of death.  
                - **Early detection is critical:** Early detection of arrhythmias and myocardial infarction is crucial for preventing severe outcomes.  
                - **Scalability challenge:** ECGs are widely available but challenging to analyze efficiently at scale.  
                - **Better outcomes:** Automated systems can standardize interpretation, reduce diagnostic delays, and enhance patient outcomes.  
                """
            )

    st.markdown("---")

    with st.expander("📚 Citations", expanded=False):
        st.write(
        """
        [1] Wikipedia. Heart: https://en.wikipedia.org/wiki/Heart#/media/  

        [2] Electrocardiogram Heartbeat Classification for Arrhythmias and Myocardial Infarction; Pham BT, Le PT, Tai TC, Hsu YC, Li YH, Wang JC (2023). Sensors (Basel). 2023 Mar 9;23(6):2993. doi: 10.3390/s23062993. PMID: 36991703; PMCID: PMC10051525.

        [3] ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee, S. Fazeli, M. Sarrafzadeh (2018); CoRR; doi: 10.48550/arXiv.1805.00794
        """
        )


if __name__ == "__main__":
    render()
