"""
Page 12: SHAP Analysis on PTB
Same as Page 11 but on PTB with 2 classes
Todo by Julia
"""

import os
import base64
from pathlib import Path

import streamlit as st
from page_modules.styles import COLORS

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_12"

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
        '<div class="hero-title" style="justify-content: center;">🔍 SHAP Analysis - PTB</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Goal description in styled container
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 1.25rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1.5rem;">
            <p style="margin: 0; opacity: 0.95;">
                <strong>Goal:</strong> Identification of important ECG signal parts for decision making of the model
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero header for SHAP Analysis Results
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📊 SHAP Analysis Results</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander(
        "Results from feature importance analysis and overlay of ECG signal and SHAP values (PTB)",
        expanded=False,
    ):
        # Create tabs for each class
        tab0, tab1 = st.tabs(
            [
                "Class 0",
                "Class 1",
            ]
        )

        # Class 0
        with tab0:
            st.markdown(
                """
            - 6 out of 20 most important features before timestep 20
            - 12 out of 20 most important features distributed between 20 to 40
            - **Very early and early timesteps are important**
            - Both R peaks important
            - P and T wave are also important features for decision towards class 0
            """
            )

        # Class 1
        with tab1:
            st.markdown(
                """
            - Region of second R peak and beginning of the signal have strong influence
            - P wave before R peak is important
            - pattern of high SHAP values more complex and spread as for class 0
            """
            )

    with st.expander("Misclassification Explanation (PTB)", expanded=False):
        st.markdown(
            """
        **Largest Misclassification: True Class 1 → Predicted Class 0**
        - Very early timesteps  and R peaks are important for prediction for both classes
        """
        )

    # Hero header for SHAP Analysis Plots
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📈 SHAP Analysis Plots</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Top 20 Most Important Features per Class (PTB)", expanded=False):
        st.markdown(
            """
        The visualization helps understand which timesteps in the ECG signal contribute most to the model's predictions.
        """
        )

        # Create tabs for each class
        tab0, tab1 = st.tabs(
            [
                "Class 0",
                "Class 1",
            ]
        )

        # Class 0
        with tab0:
            image_path = IMAGES_DIR / "shap_ptb_class0.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 0", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 0 not found")

        # Class 1
        with tab1:
            image_path = IMAGES_DIR / "shap_ptb_class1.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 1", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 1 not found")

    with st.expander("ECG Signal with SHAP Values Overlay (PTB)", expanded=False):
        st.markdown(
            """
        Explore individual ECG signals overlaid with their corresponding SHAP values.
        Select an example to see how SHAP values highlight important signal regions.
        """
        )

        # Create tabs for each class
        tab0, tab1 = st.tabs(
            [
                "Class 0",
                "Class 1",
            ]
        )

        # Class 0
        with tab0:
            example_num = st.selectbox("Select example:", [1, 2], key="ptb_class0_example")
            image_path = IMAGES_DIR / f"shap_ecg_ptb_class_0_example_{example_num}.png"
            if image_path.exists():
                shap_img = get_image_html(
                    image_path, f"SHAP ECG overlay Class 0 Example {example_num}", ""
                )
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 0, Example {example_num} not found")

        # Class 1
        with tab1:
            example_num = st.selectbox("Select example:", [1, 2], key="ptb_class1_example")
            image_path = IMAGES_DIR / f"shap_ecg_ptb_class_1_example_{example_num}.png"
            if image_path.exists():
                shap_img = get_image_html(
                    image_path, f"SHAP ECG overlay Class 1 Example {example_num}", ""
                )
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 1, Example {example_num} not found")

    with st.expander("SHAP Summary Plots (PTB)", expanded=False):
        st.markdown(
            """
        Summary plots showing the distribution of SHAP values for each feature across all samples.
        These plots help understand the overall impact and spread of feature importance.
        """
        )

        # Create tabs for each class
        tab0, tab1 = st.tabs(
            [
                "Class 0",
                "Class 1",
            ]
        )

        # Class 0
        with tab0:
            image_path = IMAGES_DIR / "shap_summary_class_0.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP summary Class 0", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 600px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP summary plot for Class 0 not found")

        # Class 1
        with tab1:
            image_path = IMAGES_DIR / "shap_summary_class_1.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP summary Class 1", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 600px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP summary plot for Class 1 not found")
