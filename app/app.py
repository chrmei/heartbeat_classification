"""
Heartbeat Classification - Streamlit Presentation App
Main entry point for the Streamlit application
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heartbeat Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

# Define pages with their module names
pages = {
    "1. Introduction": "page_1_introduction",
    "2. Data Overview": "page_2_data_overview",
    "3. Pre-Processing (RR-Distances)": "page_3_preprocessing",
    "4. Modeling Overview": "page_4_general_modeling_overview",
    "5. Baseline Results - MIT": "page_5_baseline_mit",
    "6. Baseline Results - PTB": "page_6_baseline_ptb",
    "7. Deep Learning Models": "page_7_dl_models",
    "8. Deep Learning - MIT": "page_8_dl_mit",
    "9. Deep Learning - PTB (Transfer)": "page_9_dl_ptb",
    "10. Result Summary": "page_10_summary",
    "11. SHAP Analysis - MIT": "page_11_shap_mit",
    "12. SHAP Analysis - PTB": "page_12_shap_ptb",
    "13. Conclusion": "page_13_conclusion",
}

# Page selection
selected_page = st.sidebar.radio("Select a page:", list(pages.keys()))

# Route to selected page
page_module_name = pages[selected_page]

# Import and render the selected page
if page_module_name == "page_1_introduction":
    from page_modules import page_1_introduction

    page_1_introduction.render()

elif page_module_name == "page_2_data_overview":
    from page_modules import page_2_data_overview

    page_2_data_overview.render()

elif page_module_name == "page_3_preprocessing":
    from page_modules import page_3_preprocessing

    page_3_preprocessing.render()

elif page_module_name == "page_4_general_modeling_overview":
    from page_modules import page_4_general_modeling_overview

    page_4_general_modeling_overview.render()

elif page_module_name == "page_5_baseline_mit":
    from page_modules import page_5_baseline_mit

    page_5_baseline_mit.render()

elif page_module_name == "page_6_baseline_ptb":
    from page_modules import page_6_baseline_ptb

    page_6_baseline_ptb.render()

elif page_module_name == "page_7_dl_models":
    from page_modules import page_7_dl_models

    page_7_dl_models.render()

elif page_module_name == "page_8_dl_mit":
    from page_modules import page_8_dl_mit

    page_8_dl_mit.render()

elif page_module_name == "page_9_dl_ptb":
    from page_modules import page_9_dl_ptb

    page_9_dl_ptb.render()

elif page_module_name == "page_10_summary":
    from page_modules import page_10_summary

    page_10_summary.render()


elif page_module_name == "page_11_shap_mit":
    from page_modules import page_11_shap_mit

    page_11_shap_mit.render()

elif page_module_name == "page_12_shap_ptb":
    from page_modules import page_12_shap_ptb

    page_12_shap_ptb.render()

elif page_module_name == "page_13_conclusion":
    from page_modules import page_13_conclusion

    page_13_conclusion.render()
