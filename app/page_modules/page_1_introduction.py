"""
Page 1: Presentation of the subject, the problem and the issues
Enhanced with hero section and key metrics
"""

import streamlit as st
import base64
from pathlib import Path
from page_modules.styles import render_hero_section, COLORS

# Base path for images
IMAGES_DIR = Path(__file__).parent.parent / "images" / "page_1"


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
    ext = image_path.suffix.lower()
    mime_types = {".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_types.get(ext, "image/png")
    b64 = get_image_base64(image_path)
    
    caption_html = f'<p style="text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{caption}</p>' if caption else ''
    
    return f'''
        <img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">
        {caption_html}
    '''


def render():
    # Hero Section with Key Results
    render_hero_section(
        title="Heartbeat Classification",
        subtitle="Automated ECG Analysis using Deep Learning for Arrhythmia Detection & Myocardial Infarction Diagnosis",
        metrics=[
            {"value": "98.51%", "label": "MIT-BIH Accuracy"},
            {"value": "98.42%", "label": "PTB Accuracy"},
            {"value": "0.9236", "label": "F1-Score (MIT)"},
            {"value": "CNN + Transfer", "label": "Best Approach"},
        ]
    )
    
    st.markdown("---")

    # Tabbed Content
    tab1, tab2, tab3 = st.tabs([
        "📋 Context & Motivation",
        "🎯 Project Goals",
        "⚡ Why Automation?"
    ])

    # --- Tab 1: Context, Problem & Motivation ---
    with tab1:
        # Get base64 encoded images
        heart_img = get_image_html(IMAGES_DIR / "human_heart.svg", "Heart anatomy", "Heart anatomy [1]")
        ecg_img = get_image_html(IMAGES_DIR / "ECG_wave.jpg", "ECG waveform", "ECG waveform [2]")
        
        # Row 1: Global Health Crisis with Heart Image - Single container
        st.markdown(f"""<div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); padding: 1.5rem; border-radius: 12px; color: white; box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1rem;">
<div style="display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap;">
<div style="flex: 1.5; min-width: 300px;">
<h4 style="color: white; margin-top: 0; font-size: 1.3rem;">🌍 Global Health Crisis</h4>
<p style="opacity: 0.95; margin-bottom: 0.75rem;"><strong>Cardiovascular diseases are the leading cause of death worldwide.</strong></p>
<p style="opacity: 0.9; margin-bottom: 0.5rem;">The heart maintains blood circulation throughout the body, supplying organs with oxygen and nutrients. Detecting abnormalities early is essential to prevent severe outcomes:</p>
<ul style="opacity: 0.9; margin-bottom: 0; padding-left: 1.25rem;">
<li><strong>Arrhythmias</strong> — Irregular heartbeat patterns</li>
<li><strong>Myocardial Infarction (MI)</strong> — Heart attack</li>
</ul>
</div>
<div style="flex: 1; min-width: 200px; max-width: 300px; text-align: center;">{heart_img}</div>
</div>
</div>""", unsafe_allow_html=True)
        
        # Row 2: ECGs with ECG Waveform Image - Single container
        st.markdown(f"""<div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); padding: 1.5rem; border-radius: 12px; color: white; box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
<div style="display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap;">
<div style="flex: 1.5; min-width: 300px;">
<h4 style="color: white; margin-top: 0; font-size: 1.3rem;">📈 Electrocardiograms (ECGs)</h4>
<p style="opacity: 0.95; margin-bottom: 0.75rem;">ECGs are the <strong>gold standard</strong> for non-invasive heart monitoring:</p>
<ul style="opacity: 0.9; margin-bottom: 0; padding-left: 1.25rem;">
<li>Record the electrical activity of the heart to identify irregularities</li>
<li>Capture characteristic components for each heartbeat:
<ul style="margin-top: 0.25rem;">
<li><strong>P wave</strong> — Atrial contraction</li>
<li><strong>QRS complex</strong> — Ventricular contraction</li>
<li><strong>T wave</strong> — Ventricular relaxation</li>
</ul>
</li>
</ul>
</div>
<div style="flex: 1; min-width: 200px; max-width: 300px; text-align: center;">{ecg_img}</div>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #9B2226 0%, #6D1A1D 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(155, 34, 38, 0.3);">
                    <h4 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.2rem;">⚠️ The Problem</h4>
                    <p style="margin: 0; font-size: 1rem; line-height: 1.5; opacity: 0.95;">
                        Manual ECG interpretation is <strong>time-consuming</strong> and prone to 
                        <strong>human error</strong>, especially when dealing with large volumes of patient data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(45, 106, 79, 0.3);">
                    <h4 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.2rem;">💡 Our Solution</h4>
                    <p style="margin: 0; font-size: 1rem; line-height: 1.5; opacity: 0.95;">
                        Develop an <strong>automated Deep Learning classification system</strong> as a 
                        decision-support tool for faster, more consistent screening.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Tab 2: Project Goals ---
    with tab2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_lighter']} 0%, #A8DADC 100%); 
                        padding: 1rem 1.5rem; border-radius: 12px; color: {COLORS['clinical_blue']};
                        box-shadow: 0 4px 15px rgba(168, 218, 220, 0.4); margin-bottom: 1rem; text-align: center;">
                <p style="margin: 0; font-size: 1.1rem; font-weight: 500;">
                    This project focuses on automating <strong>two clinically relevant ECG classification tasks</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="display: flex; gap: 1rem; align-items: stretch;">
                <div style="flex: 1; background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3);">
                    <h4 style="color: white; margin-top: 0; font-size: 1.2rem;">🫀 Task 1: Arrhythmia Classification</h4>
                    <p style="opacity: 0.95; margin-bottom: 0.5rem;"><strong>Dataset:</strong> MIT-BIH (5 Classes)</p>
                    <p style="opacity: 0.95; margin-bottom: 0.25rem;">Categorizing heartbeats into 5 distinct types:</p>
                    <ul style="opacity: 0.9; margin-bottom: 0.5rem; padding-left: 1.25rem;">
                        <li>Normal (N)</li>
                        <li>Supraventricular (S)</li>
                        <li>Ventricular (V)</li>
                        <li>Fusion (F)</li>
                        <li>Unknown (Q)</li>
                    </ul>
                    <p style="opacity: 0.95; margin-bottom: 0;"><strong>Target:</strong> Beat benchmark of 93.4% accuracy</p>
                </div>
                <div style="flex: 1; background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3);">
                    <h4 style="color: white; margin-top: 0; font-size: 1.2rem;">❤️‍🩹 Task 2: MI Detection</h4>
                    <p style="opacity: 0.95; margin-bottom: 0.5rem;"><strong>Dataset:</strong> PTB (Binary Classification)</p>
                    <p style="opacity: 0.95; margin-bottom: 0.25rem;">Distinguishing between:</p>
                    <ul style="opacity: 0.9; margin-bottom: 0.5rem; padding-left: 1.25rem;">
                        <li>Healthy heartbeats</li>
                        <li>Myocardial Infarction (MI)</li>
                    </ul>
                    <p style="opacity: 0.95; margin-bottom: 0.5rem;">Using a <strong>transfer learning</strong> approach from the MIT-BIH trained model.</p>
                    <p style="opacity: 0.95; margin-bottom: 0;"><strong>Target:</strong> Beat benchmark of 95.9% accuracy</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['heart_red']} 0%, #9B2226 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white;
                        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3);">
                <h4 style="color: white; margin-top: 0; font-size: 1.2rem;">🏆 Primary Objective</h4>
                <p style="opacity: 0.95; margin-bottom: 1rem;">
                    <strong>Goal:</strong> Build a model that outperforms the 2018 benchmark study by Kachuee et al. [3]
                </p>
                <table style="width: 100%; border-collapse: collapse; color: white;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(255,255,255,0.3);">
                            <th style="text-align: left; padding: 0.5rem;">Dataset</th>
                            <th style="text-align: center; padding: 0.5rem;">Benchmark</th>
                            <th style="text-align: center; padding: 0.5rem;">Our Target</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                            <td style="padding: 0.5rem;">MIT-BIH</td>
                            <td style="text-align: center; padding: 0.5rem;">93.4%</td>
                            <td style="text-align: center; padding: 0.5rem; font-weight: bold;">>93.4%</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.5rem;">PTB</td>
                            <td style="text-align: center; padding: 0.5rem;">95.9%</td>
                            <td style="text-align: center; padding: 0.5rem; font-weight: bold;">>95.9%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Tab 3: Why Automate this? ---
    with tab3:
        st.markdown(
            '<div class="hero-container" style="padding: 1.5rem;">'
            '<div class="hero-title" style="font-size: 1.8rem;">🔑 Key Reasons for Automation</div>'
            '</div>',
            unsafe_allow_html=True
        )
        
        reasons = [
            {
                "icon": "💔",
                "title": "Leading Cause of Death",
                "desc": "Cardiovascular diseases are the #1 global cause of mortality."
            },
            {
                "icon": "⏰",
                "title": "Early Detection is Critical",
                "desc": "Early detection of arrhythmias and MI prevents severe outcomes."
            },
            {
                "icon": "📊",
                "title": "Scalability Challenge",
                "desc": "ECGs are widely available but challenging to analyze efficiently at scale."
            },
            {
                "icon": "✨",
                "title": "Better Outcomes",
                "desc": "Automated systems standardize interpretation and reduce diagnostic delays."
            },
        ]
        
        # First row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white; height: 120px;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">{reasons[0]['icon']} {reasons[0]['title']}</h4>
                    <p style="margin: 0; opacity: 0.9;">{reasons[0]['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white; height: 120px;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">{reasons[1]['icon']} {reasons[1]['title']}</h4>
                    <p style="margin: 0; opacity: 0.9;">{reasons[1]['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("")
        
        # Second row
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white; height: 120px;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">{reasons[2]['icon']} {reasons[2]['title']}</h4>
                    <p style="margin: 0; opacity: 0.9;">{reasons[2]['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white; height: 120px;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">{reasons[3]['icon']} {reasons[3]['title']}</h4>
                    <p style="margin: 0; opacity: 0.9;">{reasons[3]['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Citations
    with st.expander("📚 References", expanded=False):
        st.markdown(
            """
            **[1]** Wikipedia. Heart anatomy. 
            [https://en.wikipedia.org/wiki/Heart](https://en.wikipedia.org/wiki/Heart)

            **[2]** Pham BT, Le PT, Tai TC, et al. (2023). *Electrocardiogram Heartbeat Classification 
            for Arrhythmias and Myocardial Infarction*. Sensors. 
            [DOI: 10.3390/s23062993](https://doi.org/10.3390/s23062993)

            **[3]** Kachuee M, Fazeli S, Sarrafzadeh M. (2018). *ECG Heartbeat Classification: 
            A Deep Transferable Representation*. 
            [arXiv:1805.00794](https://arxiv.org/abs/1805.00794)
            """
        )


if __name__ == "__main__":
    render()
