"""
Page 11: SHAP Analysis on MIT - best models
Loading of SHAP graphics (pre-created)
Todo by Julia
"""

import os

import streamlit as st


def render():
    st.title("SHAP Analysis - MIT")
    st.markdown("---")

    st.write("""
    **Goal: Identification of important ECG signal parts for decision making of the model**
    """)

    st.header("SHAP Analysis Results")

    with st.expander(
        "Results from feature importance analysis and overlay of ECG signal and SHAP values",
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
            st.markdown("""
            - 11 out of 20 most important features before timestep 20
            - 6 out of 20 most important features between timestep 90 and 110
            - R peak around feature 70-90 is of very high importance
            - R peak at the beginning of the signal is important
            - **Very early timesteps and timesteps around middle of signal are important**
            """)

        # Class 1
        with tab1:
            st.markdown("""
            - 15 out of 20 most important features before timestep 20
            - 3 out of 20 most important features between timestep 90 and 110
            - **Very early timesteps are important**
            - identified important signal parts:
                * feature 0-10 (R peak)
                * feature 25-50 (T wave, also after second R peak)
                * feature 75-110 (R peak)
            - More detailed understanding of the influence of the signal parts on the model prediction by analysing overlay plots
            """)

        # Class 2
        with tab2:
            st.markdown("""
            - 12 out of 20 most important features before timestep 20
            - 7 out of 20 most important features between timestep 20 and 30
            - Overlay plots: no R peak present at the beginnig of the signal -> very important and explains good classification results for class 2 (unique ECG characteristic)
            - **Very early and early timesteps are important**
            """)

        # Class 3
        with tab3:
            st.markdown("""
            - 8 out of 20 most important features before timestep 20
            - 3 out of 20 most important features between timestep 60 and 70
            - 5 out of 20 most important features between timestep 90 and 110
            - **Very early timesteps and timesteps around the middle of the signal are important**
            - R peak important, but distribution of high SHAP values is spread and complex
            - Zero-padding seems to be important for prediction: RR interval might contribute to identification
            """)

        # Class 4
        with tab4:
            st.markdown("""
            - 7 out of 20 most important features before timestep 20
            - 10 out of 20 most important features between timestep 90 and 110
            - High importance: second R peak (around feature 100) -> different location than class 0, important for class separation
            - **Very early timesteps and timesteps around middle of signal are important**
            """)

    with st.expander("Misclassification Explanation", expanded=False):
        st.markdown("""
        **Largest Misclassification: True Class 1 → Predicted Class 0**
        - Very early timesteps important for prediction for both classes

        **True Class 3 -> Predicted Class 0**
        - Very early timesteps and timesteps around middle of signal are important for prediction for both classes

        **True Class 3 -> Predicted Class 2 and vice versa**
        - Very early timesteps important for prediction for both classes

        **True Class 3 -> Predicted Class 1**
        - Very early timesteps important for prediction for both classes

        Class 1 and class 3 are the most underrepresented classes in the dataset which likely contributes to the model's difficulty in predicting class 1 and 3.
        """)

    st.header("SHAP Analysis Plots")

    with st.expander("Top 20 Most Important Features per Class", expanded=False):
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
            image_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_11", "shap_mit_class0.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP image for Class 0 not found")

        # Class 1
        with tab1:
            image_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_11", "shap_mit_class1.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP image for Class 1 not found")

        # Class 2
        with tab2:
            image_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_11", "shap_mit_class2.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP image for Class 2 not found")

        # Class 3
        with tab3:
            image_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_11", "shap_mit_class3.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP image for Class 3 not found")

        # Class 4
        with tab4:
            image_path = os.path.join(
                os.path.dirname(__file__), "..", "images", "page_11", "shap_mit_class4.png"
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error("⚠️ SHAP image for Class 4 not found")

    with st.expander("ECG Signal with SHAP Values Overlay", expanded=False):
        st.markdown("""
        Explore individual ECG signals overlaid with their corresponding SHAP values.
        Select an example to see how SHAP values highlight important signal regions.
        """)

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
            example_num = st.selectbox(
                "Select example:", [1, 2, 3], key="class0_example"
            )
            image_path = os.path.join(
                os.path.dirname(__file__),
                "images",
                "page_11",
                f"shap_ecg_mit_class_0_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(
                    f"⚠️ SHAP ECG overlay image for Class 0, Example {example_num} not found"
                )

        # Class 1
        with tab1:
            example_num = st.selectbox(
                "Select example:", [1, 2, 3], key="class1_example"
            )
            image_path = os.path.join(
                os.path.dirname(__file__),
                "images",
                "page_11",
                f"shap_ecg_mit_class_1_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(
                    f"⚠️ SHAP ECG overlay image for Class 1, Example {example_num} not found"
                )

        # Class 2
        with tab2:
            example_num = st.selectbox(
                "Select example:", [1, 2, 3], key="class2_example"
            )
            image_path = os.path.join(
                os.path.dirname(__file__),
                "images",
                "page_11",
                f"shap_ecg_mit_class_2_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(
                    f"⚠️ SHAP ECG overlay image for Class 2, Example {example_num} not found"
                )

        # Class 3
        with tab3:
            example_num = st.selectbox(
                "Select example:", [1, 2, 3], key="class3_example"
            )
            image_path = os.path.join(
                os.path.dirname(__file__),
                "images",
                "page_11",
                f"shap_ecg_mit_class_3_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(
                    f"⚠️ SHAP ECG overlay image for Class 3, Example {example_num} not found"
                )

        # Class 4
        with tab4:
            example_num = st.selectbox(
                "Select example:", [1, 2, 3], key="class4_example"
            )
            image_path = os.path.join(
                os.path.dirname(__file__),
                "images",
                "page_11",
                f"shap_ecg_mit_class_4_example_{example_num}.png",
            )
            if os.path.exists(image_path):
                st.image(image_path, width=800)
            else:
                st.error(
                    f"⚠️ SHAP ECG overlay image for Class 4, Example {example_num} not found"
                )
