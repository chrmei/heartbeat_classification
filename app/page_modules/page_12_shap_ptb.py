"""
Page 12: SHAP Analysis on PTB
Same as Page 11 but on PTB with 2 classes
"""

from pathlib import Path

import streamlit as st

from page_modules.styles import COLORS, render_page_hero, render_sub_hero, render_info_box
from page_modules.utils import get_image_html

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_12"


def render():
    # Hero-style header
    render_page_hero("SHAP Analysis - PTB", icon="🔍")

    st.markdown("---")

    # Goal description in styled container
    render_info_box(
        """
        <p style="margin: 0; opacity: 0.95;">
            <strong>Goal:</strong> Identification of important ECG signal parts for decision making of the model
        </p>
        """,
        variant="info",
    )

    # Hero header for SHAP Analysis Results
    render_sub_hero("SHAP Analysis Results", icon="📊")

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
    render_sub_hero("SHAP Analysis Plots", icon="📈")

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
