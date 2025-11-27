"""
Page 12: SHAP Analysis on PTB
Same as Page 11 but on PTB with 2 classes
Todo by Julia
"""

import os

import streamlit as st


def render():
    st.title("12: SHAP Analysis - PTB")
    st.markdown("---")

    st.write(
        """
    **Goal: Identification of important ECG signal parts for decision making of the model**
    """
    )

    st.header("SHAP Analysis Results")

    with st.expander(
        "Results from feature importance analysis and overlay of ECG signal and SHAP values",
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
            - P and T wave are also important features for decision towards class 0            """
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

    with st.expander("Misclassification Explanation", expanded=False):
        st.markdown(
            """
        **Largest Misclassification: True Class 1 → Predicted Class 0**
        - Very early timesteps  and R peaks are important for prediction for both classes
        """
        )

    st.header("SHAP Analysis Plots")

    with st.expander("Top 20 Most Important Features per Class", expanded=False):
        st.markdown(
            """
        The visualization helps understand which timesteps in the ECG signal contribute most to the model's predictions.
        """
        )

        (tab0,) = st.tabs(["Class 0"])

        with tab0:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_12",
                "shap_ptb_class0.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP combined image not found")

    with st.expander("ECG Signal with SHAP Values Overlay", expanded=False):
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
            example_num = st.selectbox("Select example:", [1, 2], key="class0_example")
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_12",
                f"shap_ecg_ptb_class_0_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 0, Example {example_num} not found")

        # Class 1
        with tab1:
            example_num = st.selectbox("Select example:", [1, 2], key="class1_example")
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_12",
                f"shap_ecg_ptb_class_1_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(f"⚠️ SHAP ECG overlay image for Class 1, Example {example_num} not found")

    with st.expander("SHAP Summary Plots", expanded=False):
        st.markdown(
            """
        Summary plot showing the distribution of SHAP values for each feature across all samples.
        This plot helps understand the overall impact and spread of feature importance.
        """
        )

        (tab0,) = st.tabs(["Class 0"])

        with tab0:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "images",
                "page_12",
                "shap_summary_class_0.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=600)
            else:
                st.error("⚠️ SHAP summary image for Class 0 not found")
