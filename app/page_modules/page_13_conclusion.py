"""
Page 13: Conclusion
Enhanced with better styling and clinical context
"""

import streamlit as st
from page_modules.styles import COLORS


def render():
    st.title("Conclusion")
    st.markdown("---")

    # ==========================================================================
    # PROJECT SUMMARY - HERO SECTION
    # ==========================================================================
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%); 
                    padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;
                    box-shadow: 0 4px 20px rgba(45, 106, 79, 0.3);">
            <h2 style="margin: 0 0 1rem 0; color: white;">✅ Mission Accomplished</h2>
            <p style="font-size: 1.1rem; opacity: 0.95; margin: 0;">
                We successfully automated both ECG classification tasks, with deep learning models 
                <strong>significantly outperforming</strong> the 2018 benchmark study.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key achievements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MIT-BIH Accuracy", "98.51%", "+5.11% vs benchmark")
    with col2:
        st.metric("PTB Accuracy", "98.42%", "+2.52% vs benchmark")
    with col3:
        st.metric("Interpretability", "SHAP Validated", "R-peaks confirmed")

    st.markdown(
        """
        **Key Results:**
        - **Strong Performance:** Our CNN8 model outperformed benchmark results on both datasets
        - **Interpretability:** SHAP analysis confirmed that the model focuses on physiologically 
          meaningful features (e.g., R-peaks) rather than noise
        - **Transfer Learning:** Pre-trained models effectively compensated for limited PTB dataset size
        
        Although overall accuracy is high, remaining misclassifications are clinically relevant and 
        highlight the need for improved data quality and expert review.
        """
    )

    st.markdown("---")

    # ==========================================================================
    # CRITICISM AND OUTLOOK
    # ==========================================================================
    
    st.header("🔬 Criticism & Future Outlook")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['warning']}; margin-bottom: 1rem;">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">📊 Data Quality</h4>
                <ul style="margin-bottom: 0;">
                    <li>Class imbalance remains the key bottleneck</li>
                    <li>Dataset should ideally be <strong>reviewed by medical experts</strong></li>
                    <li>Future work: collect <strong>additional real clinical data</strong> for 
                        underrepresented classes (MIT Classes 1 & 3)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['heart_red']};">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">⚠️ False Negatives</h4>
                <ul style="margin-bottom: 0;">
                    <li>Some true arrhythmias incorrectly predicted as "Normal"</li>
                    <li>In clinical contexts, <strong>reducing false negatives is highest priority</strong></li>
                    <li>Missed abnormalities can delay critical interventions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['success']}; margin-bottom: 1rem;">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">🔄 Transfer Learning</h4>
                <ul style="margin-bottom: 0;">
                    <li>Strong PTB performance suggests <strong>pre-trained models</strong> help 
                        with limited data</li>
                    <li>Opens potential for improving generalization when clinical data is scarce</li>
                    <li>Can be extended to other cardiac conditions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['clinical_blue_light']};">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">🚀 Future Extensions</h4>
                <ul style="margin-bottom: 0;">
                    <li>Enable clinicians to upload raw ECGs directly</li>
                    <li>Integrate DL models into ECG devices for real-time analysis</li>
                    <li>Multi-lead ECG support for broader applicability</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ==========================================================================
    # CLINICAL WORKFLOW INTEGRATION
    # ==========================================================================
    
    st.header("🏥 Clinical Workflow Integration")
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;">
            <h3 style="margin-top: 0; color: white;">Potential Integration Pathway</h3>
            <p style="font-size: 1.05rem; opacity: 0.95;">
                Although this is a research prototype, the approach could support an 
                <strong>ECG triage workflow</strong>:
            </p>
            <div style="display: flex; justify-content: space-around; margin-top: 1.5rem; flex-wrap: wrap;">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">1️⃣</div>
                    <strong>Automated Screening</strong><br>
                    <span style="font-size: 0.9rem; opacity: 0.85;">Model analyzes incoming ECGs</span>
                </div>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">2️⃣</div>
                    <strong>Risk Flagging</strong><br>
                    <span style="font-size: 0.9rem; opacity: 0.85;">Abnormal patterns highlighted</span>
                </div>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">3️⃣</div>
                    <strong>Clinical Review</strong><br>
                    <span style="font-size: 0.9rem; opacity: 0.85;">Experts verify flagged cases</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['clinical_blue_light']};">
                <span style="font-size: 2rem;">⚡</span>
                <h4 style="margin: 0.5rem 0;">Reduced Workload</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Automates routine manual screening
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['heart_red']};">
                <span style="font-size: 2rem;">🔍</span>
                <h4 style="margin: 0.5rem 0;">Early Detection</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Catches subtle abnormalities
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['success']};">
                <span style="font-size: 2rem;">⏱️</span>
                <h4 style="margin: 0.5rem 0;">Faster Decisions</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Accelerates time-critical care
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ==========================================================================
    # CRITICAL CONSIDERATIONS
    # ==========================================================================
    
    st.header("⚕️ Critical Considerations")
    
    st.warning(
        """
        **Decision Support, Not Diagnosis**
        
        The model should act as a **"second pair of eyes"** and a **decision-support tool**, 
        not a replacement for qualified medical experts.
        """
    )
    
    st.info(
        """
        **Human-in-the-Loop Requirement**
        
        Expert validation remains essential for:
        - Ensuring patient safety
        - Reviewing edge cases  
        - Handling potential labeling inconsistencies within datasets
        
        This maintains an appropriate standard of medical care while effectively leveraging automation.
        """
    )

    st.markdown("---")

    # ==========================================================================
    # CITATIONS
    # ==========================================================================
    
    with st.expander("📚 Citations", expanded=False):
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']};">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[1]</strong> Wikipedia. <em>Heart anatomy</em>. 
                    <a href="https://en.wikipedia.org/wiki/Heart" style="color: {COLORS['clinical_blue_light']};">https://en.wikipedia.org/wiki/Heart</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[2]</strong> Pham BT, Le PT, Tai TC, et al. (2023). <em>Electrocardiogram Heartbeat Classification 
                    for Arrhythmias and Myocardial Infarction</em>. Sensors, 23(6), 2993. 
                    <a href="https://doi.org/10.3390/s23062993" style="color: {COLORS['clinical_blue_light']};">DOI: 10.3390/s23062993</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[3]</strong> Kachuee M, Fazeli S, Sarrafzadeh M. (2018). <em>ECG Heartbeat Classification: 
                    A Deep Transferable Representation</em>. arXiv:1805.00794. 
                    <a href="https://arxiv.org/abs/1805.00794" style="color: {COLORS['clinical_blue_light']};">https://arxiv.org/abs/1805.00794</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[4]</strong> PhysioNet. <em>MIT-BIH Arrhythmia Database & PTB Diagnostic ECG Database</em>. 
                    <a href="https://physionet.org/" style="color: {COLORS['clinical_blue_light']};">https://physionet.org/</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[5]</strong> Ansari Y, Mourad O, Qaraqe K, Serpedin E. (2023). <em>Deep learning for ECG 
                    Arrhythmia detection and classification: an overview of progress for period 2017–2023</em>. 
                    Frontiers in Physiology, 14. 
                    <a href="https://doi.org/10.3389/fphys.2023.1246746" style="color: {COLORS['clinical_blue_light']};">DOI: 10.3389/fphys.2023.1246746</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[6]</strong> Murat F, Yildirim O, Talo M, et al. (2020). <em>Application of deep learning 
                    techniques for heartbeats detection using ECG signals-analysis and review</em>. 
                    Computers in Biology and Medicine. 
                    <a href="https://doi.org/10.1016/j.compbiomed.2020.103726" style="color: {COLORS['clinical_blue_light']};">DOI: 10.1016/j.compbiomed.2020.103726</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[7]</strong> Luz EJS, Schwartz WR, Cámara-Chávez G, Menotti D. (2015). <em>ECG-based heartbeat 
                    classification for arrhythmia detection: A survey</em>. Computer Methods and Programs 
                    in Biomedicine. 
                    <a href="https://doi.org/10.1016/j.cmpb.2015.12.008" style="color: {COLORS['clinical_blue_light']};">DOI: 10.1016/j.cmpb.2015.12.008</a>
                </p>
                <hr style="border: none; border-top: 1px solid {COLORS['text_secondary']}; opacity: 0.3; margin: 1rem 0;">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>📦 Project Repository:</strong> 
                    <a href="https://github.com/chrmei/heartbeat_classification" style="color: {COLORS['clinical_blue_light']};">github.com/chrmei/heartbeat_classification</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
