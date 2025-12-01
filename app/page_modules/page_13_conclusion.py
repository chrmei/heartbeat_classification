"""
Page 12: Conclusion
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Conclusion")
    st.markdown("---")

    st.header("Project Summary")

    st.markdown(
        """
    - **Mission Accomplished**: We successfully automated both classification tasks.  
    - **Strong Performance**: Our deep learning models outperformed benchmark results on both datasets:  
        - **MIT-BIH (Arrhythmia)**: Achieved **98.51%** accuracy (vs. 93.4% benchmark [3]).  
        - **PTB (MI)**: Achieved **98.42%** accuracy (vs. 95.9% benchmark [3]).  
    - **Interpretability**: SHAP analysis confirmed that the model focuses on physiologically meaningful features (e.g., R-peaks) rather than noise.  
    
    Although the overall accuracy is high, the remaining misclassifications are clinically relevant and highlight the need for improved data quality and expert review.  
    """
    )

    st.header("Criticism and Outlook")

    st.markdown(
        """
    - **Data Quality:**  
        - Class imbalance remains the key bottleneck of the current dataset.  
        - The dataset should be ideally **reviewed by medical expert**.  
        - Future work should include collecting **additional real clinical data"" for underrepresented classes (e.g., MIT Classes 1 & 3) rather than relying solely on synthetic augmentation.  
    
    - **False Negatives:**  
        - A small number of true arrhythmias were incorrectly predicted as “Normal”.  
        - In clinical contexts, **reducing false negatives is the highest priority**, as missed abnormalities can delay critical interventions.    
    
    - **Transfer Learning:**  
        - The strong performance on PTB dataset suggests that **pre-trained models** can help compensate for limited dataset size.   
        - This opens potential for using transfer learning to improve generalization when clinical data is scarce.  
    """
    )

    st.header("Results Linking to Clinical Use Case")

    st.markdown(
    """
    - **Potential Clinical Workflow Integration**
        - Although our project is a research prototype, the underlying approach could support the **early stages of an ECG triage workflow**:  
            1. **Automated model performs an initial screening** of incoming ECG signals.  
            2. The system **flags “Abnormal” or high-risk patterns** for closer inspection.  
            3. **Clinicians review** these flagged samples and confirm or dismiss the model’s suggestions.  
    """
    )

    st.markdown(
    """
    - **Value Proposition**  
        - Reduces routine manual screening workload.  
        - Supports **early detection** of subtle abnormalities.  
        - **Accelerates clinical decision-making**, especially in time-critical situations.  
        - Future extensions could enable clinicians to upload raw ECGs directly into the dashboard, or integrate DL models into ECG devices for real-time analysis.  
    """
    )

    st.header("Critical Considerations")

    st.markdown(
    """
    - **Decision Support, Not Diagnosis**  
        - The model should act as a **'second pair of eyes'** and a **decision-support tool**, not a replacement for qualified medical experts.  

    - **Human-in-the-Loop Requirement**. 
        - Expert validation remains essential for:  
        - Ensuring patient safety  
        - Reviewing edge cases  
        - Handling potential labeling inconsistencies within datasets  

    This maintains an appropriate standard of medical care while effectively leveraging automation.
    """
    )

    st.markdown("---")

    st.header("Citations")
    st.write(
        """
        [1] Wikipedia. Heart: https://en.wikipedia.org/wiki/Heart#/media/

        [2] Electrocardiogram Heartbeat Classification for Arrhythmias and Myocardial Infarction; Pham BT, Le PT, Tai TC, Hsu YC, Li YH, Wang JC (2023). Sensors (Basel). 2023 Mar 9;23(6):2993. doi: 10.3390/s23062993. PMID: 36991703; PMCID: PMC10051525.

        [3] ECG Heartbeat Classification: A Deep Transferable Representation; M. Kachuee, S. Fazeli, M. Sarrafzadeh (2018); CoRR; doi: 10.48550/arXiv.1805.00794

        [4] https://www.datasci.com/solutions/cardiovascular/ecg-research

        [5] Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); doi: 10.3389/fphys.2023.1246746

        [6] Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; F. Murat, O. Yildirim, M, Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); Computers in Biology and Medicine; doi:10.1016/j.compbiomed.2020.103726

        [7] ECG-based heartbeat classification for arrhythmia detection: A survey; E. J. da S. Luz, W. R. Schwartz, G. Câmara-Chávez, D. Menotti (2015); Computer Methods and Programs in Biomedicine; doi: 10.1016/j.cmpb.2015.12.008

        [8] Wikipedia. Heart: https://en.wikipedia.org/wiki/Heart#/media/

        [9] Electrocardiogram Heartbeat Classification for Arrhythmias and Myocardial Infarction; Pham BT, Le PT, Tai TC, Hsu YC, Li YH, Wang JC (2023). Sensors (Basel). 2023 Mar 9;23(6):2993. doi: 10.3390/s23062993. PMID: 36991703; PMCID: PMC10051525.

        GitHub repository: https://github.com/chrmei/heartbeat_classification
        """
    )
