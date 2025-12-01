"""
Page 3: Pre-Processing discussion around extreme RR-Distances
First MIT, then PTB
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("3: Pre-Processing: RR-Distance Analysis")
    st.markdown("---")

    st.markdown(
    """
    To better understand signal quality and dataset consistency before training, we performed an analysis of the **RR-distance**.  
    """
    )

    st.header("RR Distance Overview")

    st.markdown(
    """
    - **Definition:**  
        - The RR-distance represents the time interval between two consecutive R-peaks (i.e., the duration of one full heartbeat cycle).  
    
    - **Extraction:**  
        - Each row in the dataset corresponds to approximately 1.2 heartbeats.  
        - Because signal lengths vary, the unused tail of each row was padded with zeros.  
        - Therefore, the index of the first zero-padding point allows us to estimate the RR-distance for each sample.  
    
    - **Relevance:**   
        - Extremely short or long RR-intervals may indicate:  
            - physiological abnormalities (certain arrhythmias or conduction issues)  
            - potential annotation inconsistencies or mislabeled samples  
    """
    )

    st.divider()

    st.header("MIT-BIH: Extreme RR-Distance Analysis")
    st.markdown(
    """
    - The MIT-BIH dataset contains five heartbeat categories.  
    - After computing RR-distances for all classes, we analyzed samples with unusually short or long R-R intervals.
    """
    )

    st.image(
        "app/images/page_3/MIT_r.png",
        caption=" MIT RR-distances distribution per class with extreme values highlighted",
        width=600,
    )

    st.subheader("Key Observations")
    st.markdown(
        """
    - **Class 0 (Normal):**  
        - **Lower Extremes:** Samples with very short durations often displayed distorted waveforms that **did not look normal**.  
        → This suggests potential misclassification or labeling noise.  
        - **Upper Extremes:** Longer durations generally appeared morphologically normal.  
        """
    )
    st.image(
        "app/images/page_3/mit_c0_low-extreme.png",
        caption="MIT example plots of Class 0 heartbeats with extremely short RR-distances",
        width=1200,
    )
    st.image(
        "app/images/page_3/mit_c0_up-extreme.png",
        caption="MIT example plots of Class 0 heartbeats with extremely long RR-distances",
        width=1200,
    )

    st.markdown(
        """
    - **Classes 3 and 4:**  
        - These classes show a **high number of extreme RR values**, which is expected due to the pathological nature of these arrhythmias.  
    - **Classes 1 and 2:**  
        - Extreme examples are difficult to interpret visually because arrhythmias naturally distort timing.  
    """
    )

    st.subheader("Interpretation")
    st.markdown(
        """
    The extreme low-RR samples in **Class 0** suggest that the “normal” category may contain mislabeled beats.
    These anomalies are important because they can negatively affect model training.
    """
    )

    st.divider()

    st.header("PTB: Extreme RR-Distance Analysis")
    st.markdown(
    """
    - The PTB dataset (normal vs. MI) shows an asymmetric extreme-value pattern in its R-R distance distribution.
    """
    )

    st.image(
        "app/images/page_3/PTB_r.png",
        caption="PTB RR-distances distribution per class with extreme values highlighted",
        width=600,
    )

    st.subheader("Key Observations")
    st.markdown(
        """
    - **Class 0 (Normal):**  
        - Displays **only lower extreme values** (around 50 – 60).  
        - Several low-RR samples show abnormal ECG morphology, similar to what was observed in MIT-BIH.
        - This raises the possibility of **annotation inconsistencies** within the Normal class.
    """
    )

    st.image(
        "app/images/page_3/ptb_c0_low-extreme.png",
        caption="PTB example plots of Class 0 heartbeats with extremely short RR-distances",
        width=1200,
    )

    st.markdown(
        """
    - **Class 1 (Abnormal / MI):**
        - Shows **only upper extreme values**, with beats reaching 140 – 150 ms.
        - These high-RR values are difficult to interpret visually because MI pathology can naturally broaden the waveform.
    """
    )

    st.subheader("Interpretation")
    st.markdown(
        """
    For the Normal class, lower extreme values may still represent mislabeled or poor-quality recordings.
    """
    )

    st.divider()

    st.header("Important Findings")

    st.subheader("Modeling Impact")
    st.markdown(
        """
    Removing the suspicious Class-0 samples with extremely low RR-distances **did not improve performance metrics**. Because of this, and to avoid reducing dataset size, we **kept all samples** in the modeling pipeline.  

    """
    )

    
