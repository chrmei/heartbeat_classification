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
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

# Define pages with their module names
pages = {
    "1. Introduction": "page_1_introduction",
    "2. Data Overview": "page_2_data_overview",
    "3. Pre-Processing (RR-Distances)": "page_3_preprocessing",
    "4. Baseline Models": "page_4_baseline_models",
    "5. Baseline Results - MIT": "page_5_baseline_mit",
    "6. Baseline Results - PTB": "page_6_baseline_ptb",
    "7. Deep Learning - MIT": "page_7_dl_mit",
    "8. Deep Learning - PTB (Transfer)": "page_8_dl_ptb",
    "9. Result Summary": "page_9_summary",
    "10. SHAP Analysis - MIT": "page_10_shap_mit",
    "11. SHAP Analysis - PTB": "page_11_shap_ptb",
    "12. Conclusion": "page_12_conclusion"
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
elif page_module_name == "page_4_baseline_models":
    from page_modules import page_4_baseline_models
    page_4_baseline_models.render()
elif page_module_name == "page_5_baseline_mit":
    from page_modules import page_5_baseline_mit
    page_5_baseline_mit.render()
elif page_module_name == "page_6_baseline_ptb":
    from page_modules import page_6_baseline_ptb
    page_6_baseline_ptb.render()
elif page_module_name == "page_7_dl_mit":
    from page_modules import page_7_dl_mit
    page_7_dl_mit.render()
elif page_module_name == "page_8_dl_ptb":
    from page_modules import page_8_dl_ptb
    page_8_dl_ptb.render()
elif page_module_name == "page_9_summary":
    from page_modules import page_9_summary
    page_9_summary.render()
elif page_module_name == "page_10_shap_mit":
    from page_modules import page_10_shap_mit
    page_10_shap_mit.render()
elif page_module_name == "page_11_shap_ptb":
    from page_modules import page_11_shap_ptb
    page_11_shap_ptb.render()
elif page_module_name == "page_12_conclusion":
    from page_modules import page_12_conclusion
    page_12_conclusion.render()

