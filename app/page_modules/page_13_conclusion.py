"""
Page 13: Conclusion
Enhanced with better styling and clinical context
"""

import streamlit as st
import base64
from pathlib import Path
from page_modules.styles import COLORS

# Base path for images
IMAGES_DIR = Path(__file__).parent.parent / "images" / "page_13"


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
    ext = image_path.suffix.lower()
    mime_types = {
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    mime = mime_types.get(ext, "image/png")
    b64 = get_image_base64(image_path)

    caption_html = (
        f'<p style="text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{caption}</p>'
        if caption
        else ""
    )

    return f"""
        <img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">
        {caption_html}
    """


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🎯 Conclusion</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ==========================================================================
    # PROJECT SUMMARY - MISSION ACCOMPLISHED SECTION
    # ==========================================================================

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title" style="font-size: 1.8rem;">🎯 Mission Accomplished</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div class="hero-metric">
                    <div class="hero-metric-value">98.51%</div>
                    <div class="hero-metric-label">MIT-BIH Accuracy</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.25rem;">+5.11% vs benchmark</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">98.42%</div>
                    <div class="hero-metric-label">PTB Accuracy</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.25rem;">+2.52% vs benchmark</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">SHAP</div>
                    <div class="hero-metric-label">Interpretability</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.25rem;">R-peaks confirmed</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid {COLORS['clinical_blue']}; margin-bottom: 1.5rem;">
            <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">📊 Key Results</h4>
            <ul style="margin-bottom: 0.75rem; padding-left: 1.25rem;">
                <li><strong>Strong Performance:</strong> Our CNN8 model outperformed benchmark results on both datasets</li>
                <li><strong>Interpretability:</strong> SHAP analysis confirmed that the model focuses on physiologically 
                  meaningful features (e.g., R-peaks) rather than noise</li>
                <li><strong>Transfer Learning:</strong> Pre-trained models effectively compensated for limited PTB dataset size</li>
            </ul>
            <p style="margin-bottom: 0; color: {COLORS['text_secondary']};">
                Although overall accuracy is high, remaining misclassifications are clinically relevant and 
                highlight the need for improved data quality and expert review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ==========================================================================
    # KEY FINDINGS
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔬 Key Findings</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View Key Findings Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['success']}; height: 100%;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">✅ Strengths</h4>
                    <ul style="margin-bottom: 0; color: {COLORS['text_primary']};">
                        <li><strong style="color: {COLORS['text_primary']};">Benchmark exceeded</strong> on both datasets</li>
                        <li><strong style="color: {COLORS['text_primary']};">Transfer learning</strong> effective for PTB with limited data</li>
                        <li><strong style="color: {COLORS['text_primary']};">High recall</strong> for critical classes (Class 0, Class 4)</li>
                        <li><strong style="color: {COLORS['text_primary']};">Interpretable</strong> — SHAP confirms focus on R-peaks</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['warning']}; height: 100%;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">⚠️ Limitations</h4>
                    <ul style="margin-bottom: 0; color: {COLORS['text_primary']};">
                        <li><strong style="color: {COLORS['text_primary']};">Class 1 & 3</strong> have lower F1 due to imbalance</li>
                        <li><strong style="color: {COLORS['text_primary']};">False negatives</strong> present for abnormal classes</li>
                        <li><strong style="color: {COLORS['text_primary']};">Data quality</strong> concerns in extreme RR-distance samples</li>
                        <li><strong style="color: {COLORS['text_primary']};">Single-lead ECG</strong> limits generalization</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ==========================================================================
    # CRITICISM AND OUTLOOK
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔬 Criticism & Future Outlook</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View Criticism & Future Outlook Details", expanded=False):
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['warning']}; display: flex; flex-direction: column;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">📊 Data Quality</h4>
                    <ul style="margin-bottom: 0; flex-grow: 1;">
                        <li>Class imbalance remains the key bottleneck</li>
                        <li>Dataset should ideally be <strong>reviewed by medical experts</strong></li>
                        <li>Future work: collect <strong>additional real clinical data</strong> for 
                            underrepresented classes (MIT Classes 1 & 3)</li>
                    </ul>
                </div>
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['success']}; display: flex; flex-direction: column;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">🔄 Transfer Learning</h4>
                    <ul style="margin-bottom: 0; flex-grow: 1;">
                        <li>Strong PTB performance suggests <strong>pre-trained models</strong> help 
                            with limited data</li>
                        <li>Opens potential for improving generalization when clinical data is scarce</li>
                        <li>Can be extended to other cardiac conditions</li>
                    </ul>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['heart_red']}; display: flex; flex-direction: column;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">⚠️ False Negatives</h4>
                    <ul style="margin-bottom: 0; flex-grow: 1;">
                        <li>Some true arrhythmias incorrectly predicted as "Normal"</li>
                        <li>In clinical contexts, <strong>reducing false negatives is highest priority</strong></li>
                        <li>Missed abnormalities can delay critical interventions</li>
                    </ul>
                </div>
                <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                            border-left: 4px solid {COLORS['clinical_blue_light']}; display: flex; flex-direction: column;">
                    <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">🚀 Future Extensions</h4>
                    <ul style="margin-bottom: 0; flex-grow: 1;">
                        <li>Enable clinicians to upload raw ECGs directly</li>
                        <li>Integrate DL models into ECG devices for real-time analysis</li>
                        <li>Multi-lead ECG support for broader applicability</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ==========================================================================
    # CLINICAL WORKFLOW INTEGRATION
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🏥 Clinical Workflow Integration</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View Clinical Workflow Integration Details", expanded=False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                        padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;">
                <h3 style="margin-top: 0; color: white;">Potential Clinical Workflow Integration</h3>
                <p style="font-size: 1.05rem; opacity: 0.95;">
                    While this is a research prototype, the approach could support an 
                    <strong>ECG triage workflow</strong>:
                </p>
                <ol style="font-size: 1rem;">
                    <li><strong>Automated Screening</strong> — Model performs initial screening of incoming ECG signals</li>
                    <li><strong>Flagging System</strong> — "Abnormal" or high-risk patterns are flagged for review</li>
                    <li><strong>Clinical Review</strong> — Cardiologists confirm or dismiss the model's suggestions</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                            text-align: center; border-top: 3px solid {COLORS['clinical_blue_light']};">
                    <span style="font-size: 2rem;">⚡</span>
                    <h4 style="margin: 0.5rem 0;">Reduced Workload</h4>
                    <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                        Automates routine screening, letting clinicians focus on complex cases
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                            text-align: center; border-top: 3px solid {COLORS['heart_red']};">
                    <span style="font-size: 2rem;">🔍</span>
                    <h4 style="margin: 0.5rem 0;">Early Detection</h4>
                    <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                        Identifies subtle abnormalities that might be missed in manual review
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                            text-align: center; border-top: 3px solid {COLORS['success']};">
                    <span style="font-size: 2rem;">⏱️</span>
                    <h4 style="margin: 0.5rem 0;">Faster Decisions</h4>
                    <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                        Accelerates clinical decision-making in time-critical situations
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ==========================================================================
    # CRITICAL CONSIDERATIONS
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">⚕️ Critical Considerations</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View Critical Considerations Details", expanded=False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['warning']} 0%, #9B2226 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white;
                        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin-top: 0;">⚠️ Decision Support, Not Diagnosis</h4>
                <p style="opacity: 0.95; margin-bottom: 0;">
                    The model should act as a <strong>"second pair of eyes"</strong> and a <strong>decision-support tool</strong>, 
                    not a replacement for qualified medical experts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3);">
                <h4 style="color: white; margin-top: 0;">👥 Human-in-the-Loop Requirement</h4>
                <p style="opacity: 0.95; margin-bottom: 0.5rem;">
                    Expert validation remains essential for:
                </p>
                <ul style="opacity: 0.95; margin-bottom: 0; padding-left: 1.25rem;">
                    <li>Ensuring patient safety</li>
                    <li>Reviewing edge cases</li>
                    <li>Handling potential labeling inconsistencies within datasets</li>
                </ul>
                <p style="opacity: 0.95; margin-top: 0.75rem; margin-bottom: 0;">
                    This maintains an appropriate standard of medical care while effectively leveraging automation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
        )
