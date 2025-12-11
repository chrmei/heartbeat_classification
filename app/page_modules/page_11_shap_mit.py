"""
Page 11: SHAP Analysis on MIT - best models
Loading of SHAP graphics (pre-created)
Todo by Julia
"""

import os
import base64
from pathlib import Path

import streamlit as st
from page_modules.styles import COLORS

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_11"

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
        '<div class="hero-title" style="justify-content: center;">🔍 SHAP Analysis - MIT</div>'
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
                ⚠️ Analysis with SHAP was only done for the DL models.
            </p>
            <p style="margin: 0; opacity: 0.95;">
                🎯 <strong>Goal:</strong> Identification of important ECG signal parts for decision making of the model
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
        "Results from feature importance analysis and overlay of ECG signal and SHAP values (MIT)",
        expanded=False,
    ):
        # Create tabs for each class
        tab0, tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4",
            ]
        )

        # Class 0
        with tab0:
            st.markdown(
                """
            - 11 out of 20 most important features before timestep 20
            - 6 out of 20 most important features between timestep 90 and 110
            - R peak around feature 70-90 is of very high importance
            - R peak at the beginning of the signal is important
            - **Very early timesteps and timesteps around middle of signal are important**
            """
            )

        # Class 1
        with tab1:
            st.markdown(
                """
            - 15 out of 20 most important features before timestep 20
            - 3 out of 20 most important features between timestep 90 and 110
            - **Very early timesteps are important**
            - identified important signal parts:
                * feature 0-10 (R peak)
                * feature 25-50 (T wave, also after second R peak)
                * feature 75-110 (R peak)
            - More detailed understanding of the influence of the signal parts on the model prediction by analysing overlay plots
            """
            )

        # Class 2
        with tab2:
            st.markdown(
                """
            - 12 out of 20 most important features before timestep 20
            - 7 out of 20 most important features between timestep 20 and 30
            - Overlay plots: no R peak present at the beginnig of the signal -> very important and explains good classification results for class 2 (unique ECG characteristic)
            - **Very early and early timesteps are important**
            """
            )

        # Class 3
        with tab3:
            st.markdown(
                """
            - 8 out of 20 most important features before timestep 20
            - 3 out of 20 most important features between timestep 60 and 70
            - 5 out of 20 most important features between timestep 90 and 110
            - **Very early timesteps and timesteps around the middle of the signal are important**
            - R peak important, but distribution of high SHAP values is spread and complex
            - Zero-padding seems to be important for prediction: RR interval might contribute to identification
            """
            )

        # Class 4
        with tab4:
            st.markdown(
                """
            - 7 out of 20 most important features before timestep 20
            - 10 out of 20 most important features between timestep 90 and 110
            - High importance: second R peak (around feature 100) -> different location than class 0, important for class separation
            - **Very early timesteps and timesteps around middle of signal are important**
            """
            )

    with st.expander("Misclassification Explanation (MIT)", expanded=False):
        st.markdown(
            """
        **Largest Misclassification: True Class 1 → Predicted Class 0**
        - Very early timesteps important for prediction for both classes

        **True Class 3 -> Predicted Class 0**
        - Very early timesteps and timesteps around middle of signal are important for prediction for both classes

        **True Class 3 -> Predicted Class 2 and vice versa**
        - Very early timesteps important for prediction for both classes

        **True Class 3 -> Predicted Class 1**
        - Very early timesteps important for prediction for both classes

        Class 1 and class 3 are the most underrepresented classes in the dataset which likely contributes to the model's difficulty in predicting class 1 and 3.
        """
        )

    # Hero header for SHAP Analysis Plots
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📈 SHAP Analysis Plots</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Top 20 Most Important Features per Class (MIT)", expanded=False):
        # Create tabs for each class
        tab0, tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4",
            ]
        )

        # Class 0
        with tab0:
            image_path = IMAGES_DIR / "shap_mit_class0.png"
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
            image_path = IMAGES_DIR / "shap_mit_class1.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 1", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 1 not found")

        # Class 2
        with tab2:
            image_path = IMAGES_DIR / "shap_mit_class2.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 2", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 2 not found")

        # Class 3
        with tab3:
            image_path = IMAGES_DIR / "shap_mit_class3.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 3", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 3 not found")

        # Class 4
        with tab4:
            image_path = IMAGES_DIR / "shap_mit_class4.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP analysis Class 4", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP image for Class 4 not found")

    with st.expander("ECG Signal with SHAP Values Overlay (MIT)", expanded=False):
        st.markdown(
            """
        Explore individual ECG signals overlaid with their corresponding SHAP values.
        Select an example to see how SHAP values highlight important signal regions.
        """
        )

        # Create tabs for each class
        tab0, tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4",
            ]
        )

        # Class 0
        with tab0:
            example_num = st.selectbox("Select example:", [1, 2, 3], key="mit_class0_example")
            image_path = IMAGES_DIR / f"shap_ecg_mit_class_0_example_{example_num}.png"
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
            example_num = st.selectbox("Select example:", [1, 2, 3], key="mit_class1_example")
            image_path = IMAGES_DIR / f"shap_ecg_mit_class_1_example_{example_num}.png"
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

        # Class 2
        with tab2:
            example_num = st.selectbox("Select example:", [1, 2, 3], key="mit_class2_example")
            image_path = IMAGES_DIR / f"shap_ecg_mit_class_2_example_{example_num}.png"
            if image_path.exists():
                shap_img = get_image_html(
                    image_path, f"SHAP ECG overlay Class 2 Example {example_num}", ""
                )
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 2, Example {example_num} not found")

        # Class 3
        with tab3:
            example_num = st.selectbox("Select example:", [1, 2, 3], key="mit_class3_example")
            image_path = IMAGES_DIR / f"shap_ecg_mit_class_3_example_{example_num}.png"
            if image_path.exists():
                shap_img = get_image_html(
                    image_path, f"SHAP ECG overlay Class 3 Example {example_num}", ""
                )
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 3, Example {example_num} not found")

        # Class 4
        with tab4:
            example_num = st.selectbox("Select example:", [1, 2, 3], key="mit_class4_example")
            image_path = IMAGES_DIR / f"shap_ecg_mit_class_4_example_{example_num}.png"
            if image_path.exists():
                shap_img = get_image_html(
                    image_path, f"SHAP ECG overlay Class 4 Example {example_num}", ""
                )
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 800px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 4, Example {example_num} not found")

    with st.expander("SHAP Summary Plots per Class (MIT)", expanded=False):
        st.markdown(
            """
        Summary plots showing the distribution of SHAP values for each feature across all samples.
        These plots help understand the overall impact and spread of feature importance.
        """
        )

        # Create tabs for each class
        tab0, tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Class 0",
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4",
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

        # Class 2
        with tab2:
            image_path = IMAGES_DIR / "shap_summary_class_2.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP summary Class 2", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 600px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP summary plot for Class 2 not found")

        # Class 3
        with tab3:
            image_path = IMAGES_DIR / "shap_summary_class_3.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP summary Class 3", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 600px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP summary plot for Class 3 not found")

        # Class 4
        with tab4:
            image_path = IMAGES_DIR / "shap_summary_class_4.png"
            if image_path.exists():
                shap_img = get_image_html(image_path, "SHAP summary Class 4", "")
                st.markdown(
                    f'<div style="min-width: 200px; max-width: 600px; margin: 0 auto; text-align: center;">{shap_img}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ SHAP summary plot for Class 4 not found")
