"""
Heartbeat Classification - Streamlit Presentation App
Main entry point for the Streamlit application
"""

import streamlit as st
from page_modules.styles import inject_custom_css, render_nav_progress, render_section_header

# Page configuration
st.set_page_config(
    page_title="Heartbeat Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()

# =============================================================================
# NAVIGATION STRUCTURE - Grouped by section
# =============================================================================

# Define navigation sections and their pages
NAV_SECTIONS = {
    "Introduction": {
        "Introduction": "page_1_introduction",
    },
    "Data Analysis": {
        "Data Overview": "page_2_data_overview",
        "Preprocessing": "page_3_preprocessing",
    },
    "Baseline Models": {
        "Modeling Overview": "page_4_general_modeling_overview",
        "MIT-BIH Results": "page_5_baseline_mit",
        "PTB Results": "page_6_baseline_ptb",
    },
    "Deep Learning": {
        "DL Architecture": "page_7_dl_models",
        "DL MIT-BIH Results": "page_8_dl_mit",
        "DL PTB Transfer": "page_9_dl_ptb",
    },
    "Interpretability": {
        "SHAP - MIT": "page_11_shap_mit",
        "SHAP - PTB": "page_12_shap_ptb",
    },
    "Conclusion": {
        "Summary": "page_10_summary",
        "Conclusion": "page_13_conclusion",
    },
}

# Flatten pages for indexing
ALL_PAGES = {}
PAGE_ORDER = []
for section, pages in NAV_SECTIONS.items():
    for page_name, module_name in pages.items():
        full_name = f"{page_name}"
        ALL_PAGES[full_name] = {"module": module_name, "section": section}
        PAGE_ORDER.append(full_name)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

# Sidebar header
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">❤️</span>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">ECG Classification</h2>
        <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">Deep Learning for Healthcare</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Build navigation with sections
selected_page = None

for section_name, section_pages in NAV_SECTIONS.items():
    page_names = list(section_pages.keys())
    
    # Create radio for this section's pages
    for page_name in page_names:
        full_name = page_name
        if st.sidebar.button(
            f"{'📍 ' if selected_page == full_name else ''}{page_name}",
            key=f"nav_{section_name}_{page_name}",
            width='stretch',
        ):
            st.session_state["current_page"] = full_name

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = PAGE_ORDER[0]

selected_page = st.session_state["current_page"]

# Progress indicator
st.sidebar.markdown("---")
current_idx = PAGE_ORDER.index(selected_page) + 1
render_nav_progress(current_idx, len(PAGE_ORDER))

# Navigation buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if current_idx > 1:
        if st.button("← Previous", width='stretch'):
            st.session_state["current_page"] = PAGE_ORDER[current_idx - 2]
            st.rerun()

with col2:
    if current_idx < len(PAGE_ORDER):
        if st.button("Next →", width='stretch'):
            st.session_state["current_page"] = PAGE_ORDER[current_idx]
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.75rem; opacity: 0.7;">
        <p>DataScientest Project</p>
        <a href="https://github.com/chrmei/heartbeat_classification" target="_blank" style="color: white;">
            📦 GitHub Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# PAGE ROUTING
# =============================================================================

page_info = ALL_PAGES[selected_page]
page_module_name = page_info["module"]

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
