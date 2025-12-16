"""
Pages module for Streamlit app

Contains page modules and shared utilities for the Heartbeat Classification app.
"""

# Re-export commonly used utilities for convenience
from page_modules.utils import (
    get_image_base64,
    get_image_html,
    load_mitbih_test_data,
    load_mitbih_train_data,
    load_ptb_data,
    load_keras_model,
    load_xgboost_model,
)

from page_modules.styles import (
    COLORS,
    inject_custom_css,
    render_hero_section,
    render_card,
    render_metric_row,
    render_citations,
    render_nav_progress,
    render_section_header,
    render_page_hero,
    render_sub_hero,
    render_step_header,
    render_info_box,
    render_info_card,
    render_metric_grid,
    render_citations_expander,
    render_flex_cards,
    apply_matplotlib_style,
    get_matplotlib_style,
    get_ecg_colors,
)

from page_modules.state_utils import (
    PageState,
    get_page_state,
    update_page_state,
    reset_page_state,
    BaselineModelState,
    DeepLearningState,
    SHAPState,
    init_baseline_state,
    init_dl_state,
    init_shap_state,
)

__all__ = [
    # Utils
    "get_image_base64",
    "get_image_html",
    "load_mitbih_test_data",
    "load_mitbih_train_data",
    "load_ptb_data",
    "load_keras_model",
    "load_xgboost_model",
    # Styles
    "COLORS",
    "inject_custom_css",
    "render_hero_section",
    "render_card",
    "render_metric_row",
    "render_citations",
    "render_nav_progress",
    "render_section_header",
    "render_page_hero",
    "render_sub_hero",
    "render_step_header",
    "render_info_box",
    "render_info_card",
    "render_metric_grid",
    "render_citations_expander",
    "render_flex_cards",
    "apply_matplotlib_style",
    "get_matplotlib_style",
    "get_ecg_colors",
    # State utils
    "PageState",
    "get_page_state",
    "update_page_state",
    "reset_page_state",
    "BaselineModelState",
    "DeepLearningState",
    "SHAPState",
    "init_baseline_state",
    "init_dl_state",
    "init_shap_state",
]
