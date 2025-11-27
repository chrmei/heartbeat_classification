"""
Page 12: Conclusion
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Conclusion")
    st.markdown("---")

    st.header("Project Summary")

    st.markdown("""
    - **Mission Accomplished**: We successfully automated both classification tasks.  
    - **Strong Performance**: Our deep learning models outperformed benchmark results on both datasets:  
        - **MIT-BIH (Arrhythmia)**: Achieved **98.51%** accuracy (vs. 93.4% benchmark [3]).  
        - **PTB (MI)**: Achieved **98.42%** accuracy (vs. 95.9% benchmark [3]).  
    - **Interpretability**: SHAP analysis confirmed that the model focuses on physiologically meaningful features (e.g., R-peaks) rather than noise.  
    
    Although the overall accuracy is high, the remaining misclassifications are clinically relevant and highlight the need for improved data quality and expert review.  
    """)
    
    st.header("Results Linking to Business / Clinical Use Case")
    
    st.markdown("""
    - **Potential clinical workflow integration:**. 
    While our project is a research prototype, the underlying approach could support early stages of an ECG triage workflow:  
        1.	The automated model performs an initial screening of raw ECGs.  
        2.	It flags "Abnormal" or “high-risk” patterns for closer inspection.  
        3.	Clinicians review these flagged samples and confirm or dismiss the model’s suggestion. 
    - **Value proposition:**
    Such a system could **reduce routine screening workload, support early detection** of subtle abnormalities, and **accelerate decision-making** in time-critical situations.  
    """)
    
    st.header("Critical Consideration")

    st.markdown("""
    - **Decision support, not diagnosis:**  
    The model acts as a “second pair of eyes,” not a replacement for clinical experts.  
    - **Human-in-the-loop:**  
    Expert validation is essential for patient safety—especially for edge cases or mislabeled samples.  
    """)
    
    st.header("Criticism and Outlook")
    
    st.markdown("""
   - **Data Quality:**  
    Class imbalance remains the key bottleneck. Future work should include collecting real clinical data for underrepresented classes (MIT Classes 1 & 3), rather than relying solely on synthetic augmentation.
    - **False Negatives:**  
    A small number of true arrhythmias were predicted as “Normal.” Reducing these errors is the highest clinical priority.  
    - **Transfer Learning:**  
    The strong PTB performance suggests pre-trained models can compensate for limited dataset size.  
    - **Next Steps:**  
    Developing a real-time ECG ingestion pipeline to allow clinicians to upload raw ECGs directly into the dashboard.  

    """)
    
    st.header("Citations")

    st.write(
        """
        [1] Wikipedia. Heart: https://en.wikipedia.org/wiki/Heart#/media/  
        [2] Pham BT, Le PT, Tai TC, Hsu YC, Li YH, Wang JC. Electrocardiogram Heartbeat Classification for Arrhythmias and Myocardial Infarction. Sensors (Basel). 2023 Mar 9;23(6):2993. doi: 10.3390/s23062993. PMID: 36991703; PMCID: PMC10051525.  
        [3] M. Kachuee, S. Fazeli and M. Sarrafzadeh, "ECG Heartbeat Classification: A Deep Transferable Representation," in 2018 IEEE International Conference on Healthcare Informatics (ICHI), New York City, NY, USA, 2018, pp. 443-444, doi: 10.1109/ICHI.2018.00092.  
        [4] https://www.datasci.com/solutions/cardiovascular/ecg-research  
        [5] Ansari Y, Mourad O, Qaraqe K, Serpedin E. Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017-2023. Front Physiol. 2023 Sep 15;14:1246746. doi: 10.3389/fphys.2023.1246746. PMID: 37791347; PMCID: PMC10542398.  
        [6] Luz EJ, Schwartz WR, Cámara-Chávez G, Menotti D. ECG-based heartbeat classification for arrhythmia detection: A survey. Comput Methods Programs Biomed. 2016 Apr;127:144-64. doi: 10.1016/j.cmpb.2015.12.008. Epub 2015 Dec 30. PMID: 26775139.  
        [7] Murat F, Yildirim O, Talo M, Baloglu UB, Demir Y, Acharya UR. Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review. Comput Biol Med. 2020 May;120:103726. doi: 10.1016/j.compbiomed.2020.103726. Epub 2020 Apr 8. PMID: 32421643.  
        [8] Rawi, Atiaf A., Murtada Kalafalla Albashir and Awadallah M. Ahmed. “Classification and Detection of ECG Arrhythmia and Myocardial Infarction Using Deep Learning: A Review.” Webology (2022): n. pag.  
        [9] Velagapudi Swapna Sindhu, Kavuri Jaya Lakshmi, Ameya Sanjanita Tangellamudi, K. Ghousiya Begum. A novel deep neural network heartbeats classifier for heart health monitoring. International Journal of Intelligent Networks, Volume 4, 2023, Pages 1-10, ISSN 2666-6030.  
        [10] Xiao, Rong & Yang, Meicheng & Ma, Caiyun & Zhao, Lina & Li, Jianqing & Liu, Chengyu. (2024). Interpretable XGBoost-SHAP Model for Arrhythmic Heartbeat Classification. 10.22489/CinC.2024.186.  
        """
    )

