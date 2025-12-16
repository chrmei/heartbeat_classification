"""
Heartbeat Classification - Streamlit Presentation App
Main entry point for the Streamlit application
"""

import importlib

import streamlit as st
from page_modules.styles import inject_custom_css, render_nav_progress

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
# NAVIGATION STRUCTURE - Single source of truth
# =============================================================================

# All pages defined in order with their properties
# nav_type: "section" (grouped in expander), "standalone" (direct button), "footer" (after separator)
PAGES = [
    # Introduction section
    {
        "name": "Introduction",
        "module": "page_1_introduction",
        "section": "Introduction",
        "nav_type": "section",
    },
    # Data Analysis section
    {
        "name": "Data Overview",
        "module": "page_2_data_overview",
        "section": "Data Analysis",
        "nav_type": "section",
    },
    {
        "name": "Preprocessing",
        "module": "page_3_preprocessing",
        "section": "Data Analysis",
        "nav_type": "section",
    },
    # Modeling Overview section
    {
        "name": "Modeling Overview",
        "module": "page_4_general_modeling_overview",
        "section": "Modeling Overview",
        "nav_type": "section",
    },
    # Baseline Models section
    {
        "name": "MIT-BIH Results",
        "module": "page_5_baseline_mit",
        "section": "Baseline Models",
        "nav_type": "section",
    },
    {
        "name": "PTB Results",
        "module": "page_6_baseline_ptb",
        "section": "Baseline Models",
        "nav_type": "section",
    },
    # Deep Learning section
    {
        "name": "DL Architecture",
        "module": "page_7_dl_models",
        "section": "Deep Learning",
        "nav_type": "section",
    },
    {
        "name": "DL MIT-BIH Results",
        "module": "page_8_dl_mit",
        "section": "Deep Learning",
        "nav_type": "section",
    },
    {
        "name": "DL PTB Transfer",
        "module": "page_9_dl_ptb",
        "section": "Deep Learning",
        "nav_type": "section",
    },
    # Standalone pages
    {"name": "Summary", "module": "page_10_summary", "section": None, "nav_type": "standalone"},
    # Interpretability section
    {
        "name": "SHAP - MIT",
        "module": "page_11_shap_mit",
        "section": "Interpretability",
        "nav_type": "section",
    },
    {
        "name": "SHAP - PTB",
        "module": "page_12_shap_ptb",
        "section": "Interpretability",
        "nav_type": "section",
    },
    # More standalone
    {
        "name": "Conclusion",
        "module": "page_13_conclusion",
        "section": None,
        "nav_type": "standalone",
    },
    # Footer pages
    {"name": "Authors", "module": "page_14_authors", "section": None, "nav_type": "footer"},
]


def build_derived_structures():
    """Build derived navigation structures from the single PAGES list."""
    all_pages = {}
    page_order = []
    nav_sections = {}
    standalone_pages = {}
    footer_pages = {}

    for page in PAGES:
        name = page["name"]
        page_order.append(name)
        all_pages[name] = {"module": page["module"], "section": page["section"]}

        if page["nav_type"] == "section":
            section = page["section"]
            if section not in nav_sections:
                nav_sections[section] = {}
            nav_sections[section][name] = page["module"]
        elif page["nav_type"] == "standalone":
            standalone_pages[name] = page["module"]
        elif page["nav_type"] == "footer":
            footer_pages[name] = page["module"]

    return all_pages, page_order, nav_sections, standalone_pages, footer_pages


# Derive all structures from the single source
ALL_PAGES, PAGE_ORDER, NAV_SECTIONS, STANDALONE_PAGES, FOOTER_PAGES = build_derived_structures()

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
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# =============================================================================
# PAGE STATE MANAGEMENT - Using URL query parameters for persistence
# =============================================================================


def get_current_page():
    """Get current page from URL query params, session state, or default."""
    # First, check URL query parameters (persists across reloads)
    page_from_url = st.query_params.get("page", None)
    if page_from_url and page_from_url in PAGE_ORDER:
        # Sync to session state for consistency
        st.session_state["current_page"] = page_from_url
        return page_from_url

    # Second, check session state
    if "current_page" in st.session_state:
        page_from_state = st.session_state["current_page"]
        if page_from_state in PAGE_ORDER:
            # Sync to URL query params
            st.query_params["page"] = page_from_state
            return page_from_state

    # Default to first page
    default_page = PAGE_ORDER[0]
    st.session_state["current_page"] = default_page
    st.query_params["page"] = default_page
    return default_page


def set_current_page(page_name: str):
    """Update current page in both session state and URL query params."""
    if page_name in PAGE_ORDER:
        st.session_state["current_page"] = page_name
        st.query_params["page"] = page_name


# Get current page (from URL, session state, or default)
selected_page = get_current_page()

# Inject JavaScript to make expander headers navigate to first page when clicked
st.sidebar.markdown(
    """
<script>
function setupExpanderNavigation() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;
    
    const expanders = sidebar.querySelectorAll('[data-testid="stExpander"]');
    expanders.forEach((expander) => {
        const summary = expander.querySelector('details > summary');
        const details = expander.querySelector('details');
        if (summary && details && !summary.hasAttribute('data-nav-setup')) {
            summary.setAttribute('data-nav-setup', 'true');
            
            // Get section name from summary text (remove arrow/icon if present)
            const sectionText = summary.textContent.trim().replace(/[▶▼]/g, '').trim();
            const sectionToFirstPage = {
                'Data Analysis': 'Data Overview',
                'Modeling Overview': 'Modeling Overview',
                'Baseline Models': 'MIT-BIH Results',
                'Deep Learning': 'DL Architecture',
                'Interpretability': 'SHAP - MIT'
            };
            
            const firstPageName = sectionToFirstPage[sectionText];
            if (firstPageName) {
                let clickTimeout;
                summary.addEventListener('click', function(e) {
                    // Check if expander is currently open
                    const isOpen = details.hasAttribute('open');
                    
                    // If clicking to expand (was closed), navigate to first page
                    if (!isOpen) {
                        // Small delay to allow expand animation, then navigate
                        clearTimeout(clickTimeout);
                        clickTimeout = setTimeout(() => {
                            const url = new URL(window.location);
                            url.searchParams.set('page', firstPageName);
                            window.location.href = url.toString();
                        }, 100);
                    }
                    // If clicking to collapse (was open), just collapse (default behavior)
                });
            }
        }
    });
}

// Run on load and after Streamlit renders
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupExpanderNavigation);
} else {
    setupExpanderNavigation();
}
setTimeout(setupExpanderNavigation, 100);
setTimeout(setupExpanderNavigation, 500);

// Watch for Streamlit reruns
const observer = new MutationObserver(setupExpanderNavigation);
observer.observe(document.body, { childList: true, subtree: true });
</script>
""",
    unsafe_allow_html=True,
)

# Build navigation with sections as expandable dropdowns
for section_name, section_pages in NAV_SECTIONS.items():
    page_names = list(section_pages.keys())
    first_page_name = page_names[0]
    first_page_full_name = first_page_name

    # If section has only 1 page, show it directly without expander
    if len(page_names) == 1:
        page_name = page_names[0]
        full_name = page_name
        if st.sidebar.button(
            f"{'📍 ' if selected_page == full_name else ''}{page_name}",
            key=f"nav_{section_name}_{page_name}",
            use_container_width=True,
        ):
            set_current_page(full_name)
            st.rerun()
    else:
        # Check if current page is in this section to auto-expand
        current_section = ALL_PAGES[selected_page]["section"]
        is_expanded = section_name == current_section

        # Create expander for this section (collapsed by default, expanded only if current page is in section)
        # The header will navigate to first page via JavaScript when clicked
        with st.sidebar.expander(section_name, expanded=is_expanded):
            # Create buttons for pages in this section
            for page_name in page_names:
                full_name = page_name
                if st.button(
                    f"{'📍 ' if selected_page == full_name else ''}{page_name}",
                    key=f"nav_{section_name}_{page_name}",
                    use_container_width=True,
                ):
                    set_current_page(full_name)
                    st.rerun()

# Add standalone pages (Summary and Conclusion) directly in nav bar
st.sidebar.markdown("---")
for page_name, module_name in STANDALONE_PAGES.items():
    if st.sidebar.button(
        f"{'📍 ' if selected_page == page_name else ''}{page_name}",
        key=f"nav_standalone_{page_name}",
        use_container_width=True,
    ):
        set_current_page(page_name)
        st.rerun()

# Add footer pages (Authors) after a separator
st.sidebar.markdown("---")
for page_name, module_name in FOOTER_PAGES.items():
    if st.sidebar.button(
        f"{'📍 ' if selected_page == page_name else ''}{page_name}",
        key=f"nav_footer_{page_name}",
        use_container_width=True,
    ):
        set_current_page(page_name)
        st.rerun()

# Progress indicator and navigation buttons - exclude Authors page
# Create list of pages for progress (exclude footer pages like Authors)
PROGRESS_PAGES = [p for p in PAGE_ORDER if p not in FOOTER_PAGES]

if selected_page in PROGRESS_PAGES:
    # Only show progress bar for main content pages
    st.sidebar.markdown("---")
    current_idx = PROGRESS_PAGES.index(selected_page) + 1
    render_nav_progress(current_idx, len(PROGRESS_PAGES))

    # Navigation buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if current_idx > 1:
            if st.button("← Previous", width="stretch"):
                set_current_page(PROGRESS_PAGES[current_idx - 2])
                st.rerun()

    with col2:
        if current_idx < len(PROGRESS_PAGES):
            if st.button("Next →", width="stretch"):
                set_current_page(PROGRESS_PAGES[current_idx])
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
    unsafe_allow_html=True,
)

# =============================================================================
# PAGE ROUTING - Dynamic import based on module name
# =============================================================================

page_info = ALL_PAGES[selected_page]
page_module_name = page_info["module"]

# Dynamically import and render the selected page module
module = importlib.import_module(f"page_modules.{page_module_name}")
module.render()
