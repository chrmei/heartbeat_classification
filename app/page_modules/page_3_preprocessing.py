"""
Page 3: Pre-Processing discussion around extreme RR-Distances
First MIT, then PTB
Enhanced with better styling
"""

import base64
import streamlit as st
from pathlib import Path
from page_modules.styles import COLORS

# Base path for images
IMAGES_DIR = Path(__file__).parent.parent / "images" / "page_3"


# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================


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
        '<div class="hero-title" style="justify-content: center;">🔬 Pre-Processing: RR-Distance Analysis</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ==========================================================================
    # RR-DISTANCE OVERVIEW SECTION
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📏 RR-Distance Overview</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View RR-Distance Details", expanded=False):
        st.markdown(
            """
            To better understand signal quality and dataset consistency before training, 
            we performed an analysis of the **RR-distance**.
            """
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
                **Definition:**
                - The RR-distance represents the time interval between two consecutive R-peaks 
                  (i.e., the duration of one full heartbeat cycle)
                
                **Extraction:**
                - Each row in the dataset corresponds to approximately 1.2 heartbeats
                - Because signal lengths vary, the unused tail of each row was padded with zeros
                - The index of the first zero-padding point estimates the RR-distance for each sample
                
                **Relevance:**
                - Extremely short or long RR-intervals may indicate:
                    - Physiological abnormalities (certain arrhythmias or conduction issues)
                    - Potential annotation inconsistencies or mislabeled samples
                """
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3);">
                    <h4 style="color: white; margin-top: 0;">🔬 Why This Matters</h4>
                    <p style="font-size: 0.9rem; margin: 0; opacity: 0.95;">
                        Identifying extreme RR values helps us understand potential data quality issues 
                        and mislabeled samples that could affect model training.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ==========================================================================
    # MIT-BIH ANALYSIS
    # ==========================================================================

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title" style="font-size: 1.8rem;">🫀 MIT-BIH: Extreme RR-Distance Analysis</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div class="hero-metric">
                    <div class="hero-metric-value">5</div>
                    <div class="hero-metric-label">Heartbeat Classes</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">109k</div>
                    <div class="hero-metric-label">Total Samples</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">IQR</div>
                    <div class="hero-metric-label">Outlier Method</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View MIT-BIH Analysis Details", expanded=False):
        st.markdown(
            """
            The MIT-BIH dataset contains five heartbeat categories. After computing RR-distances 
            for all classes, we analyzed samples with unusually short or long R-R intervals.
            """
        )

        # RR-distance distribution image
        mit_rr_img = get_image_html(
            IMAGES_DIR / "MIT_r.png",
            "MIT RR-distances distribution per class",
            "MIT RR-distances distribution per class with extreme values highlighted",
        )
        st.markdown(
            f'<div style="min-width: 200px; max-width: 800px; margin: 1rem auto; text-align: center;">{mit_rr_img}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Key Observations")

        with st.expander("**Class 0 (Normal)** — Potential Mislabeling Detected", expanded=True):
            st.markdown(
                """
                - **Lower Extremes:** Samples with very short durations often displayed distorted 
                  waveforms that **did not look normal** → suggests potential misclassification
                - **Upper Extremes:** Longer durations generally appeared morphologically normal
                """
            )

            tab1, tab2 = st.tabs(["Lower Extremes", "Upper Extremes"])

            with tab1:
                low_extreme_img = get_image_html(
                    IMAGES_DIR / "mit_c0_low-extreme.png",
                    "MIT Class 0 heartbeats with extremely short RR-distances",
                    "MIT Class 0 heartbeats with extremely short RR-distances",
                )
                st.markdown(
                    f'<div style="width: 100%; text-align: center;">{low_extreme_img}</div>',
                    unsafe_allow_html=True,
                )

            with tab2:
                up_extreme_img = get_image_html(
                    IMAGES_DIR / "mit_c0_up-extreme.png",
                    "MIT Class 0 heartbeats with extremely long RR-distances",
                    "MIT Class 0 heartbeats with extremely long RR-distances",
                )
                st.markdown(
                    f'<div style="width: 100%; text-align: center;">{up_extreme_img}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("**Classes 1-4** — Expected Variability", expanded=False):
            st.markdown(
                """
                - **Classes 3 and 4:** Show a **high number of extreme RR values**, which is expected 
                  due to the pathological nature of these arrhythmias
                - **Classes 1 and 2:** Extreme examples are difficult to interpret visually because 
                  arrhythmias naturally distort timing
                """
            )

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #9B2226 0%, #6D1A1D 100%); 
                        padding: 1rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(155, 34, 38, 0.3); margin: 1rem 0;">
                <strong>⚠️ Interpretation:</strong> The extreme low-RR samples in Class 0 suggest that 
                the "normal" category may contain mislabeled beats. These anomalies can negatively 
                affect model training.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ==========================================================================
    # PTB ANALYSIS
    # ==========================================================================

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title" style="font-size: 1.8rem;">❤️‍🩹 PTB: Extreme RR-Distance Analysis</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div class="hero-metric">
                    <div class="hero-metric-value">2</div>
                    <div class="hero-metric-label">Classes (Binary)</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">14.5k</div>
                    <div class="hero-metric-label">Total Samples</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">Asymmetric</div>
                    <div class="hero-metric-label">Extreme Pattern</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View PTB Analysis Details", expanded=False):
        st.markdown(
            """
            The PTB dataset (normal vs. MI) shows an asymmetric extreme-value pattern in its 
            R-R distance distribution.
            """
        )

        # RR-distance distribution image
        ptb_rr_img = get_image_html(
            IMAGES_DIR / "PTB_r.png",
            "PTB RR-distances distribution per class",
            "PTB RR-distances distribution per class with extreme values highlighted",
        )
        st.markdown(
            f'<div style="min-width: 200px; max-width: 800px; margin: 1rem auto; text-align: center;">{ptb_rr_img}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Key Observations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; height: 200px;
                            box-shadow: 0 4px 15px rgba(45, 106, 79, 0.3);">
                    <h4 style="color: white; margin-top: 0;">Class 0 (Normal)</h4>
                    <ul style="margin-bottom: 0; opacity: 0.95;">
                        <li>Displays <strong>only lower extreme values</strong> (around 50-60)</li>
                        <li>Several low-RR samples show abnormal ECG morphology</li>
                        <li>Raises possibility of <strong>annotation inconsistencies</strong></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['heart_red']} 0%, #9B2226 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; height: 200px;
                            box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3);">
                    <h4 style="color: white; margin-top: 0;">Class 1 (Abnormal/MI)</h4>
                    <ul style="margin-bottom: 0; opacity: 0.95;">
                        <li>Shows <strong>only upper extreme values</strong> (140-150 ms)</li>
                        <li>High-RR values difficult to interpret visually</li>
                        <li>MI pathology can naturally broaden the waveform</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        with st.expander("PTB Class 0 — Lower Extreme Examples", expanded=False):
            ptb_low_img = get_image_html(
                IMAGES_DIR / "ptb_c0_low-extreme.png",
                "PTB Class 0 heartbeats with extremely short RR-distances",
                "PTB Class 0 heartbeats with extremely short RR-distances",
            )
            st.markdown(
                f'<div style="width: 100%; text-align: center;">{ptb_low_img}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ==========================================================================
    # CONCLUSIONS
    # ==========================================================================

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📝 Important Findings</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.success(
        """
        **Modeling Decision:** Removing the suspicious Class-0 samples with extremely low 
        RR-distances **did not improve performance metrics**. To avoid reducing dataset size, 
        we **kept all samples** in the modeling pipeline.
        """
    )

    st.info(
        """
        **Key Takeaway:** While data quality concerns exist (particularly in the Normal class), 
        the models appear robust enough to handle these edge cases. Future work should consider 
        expert review of flagged samples.
        """
    )
