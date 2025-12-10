"""
Session State Management Utilities for Streamlit Pages.
Provides simplified state management using dataclasses.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import streamlit as st


@dataclass
class PageState:
    """Base class for page-specific state."""
    pass


def get_page_state(page_key: str, state_class: type, **defaults) -> Any:
    """
    Get or initialize page state from session state.
    
    Args:
        page_key: Unique key for this page's state
        state_class: Dataclass type for the state
        **defaults: Default values for state initialization
    
    Returns:
        Instance of state_class from session state
    """
    if page_key not in st.session_state:
        st.session_state[page_key] = state_class(**defaults)
    return st.session_state[page_key]


def update_page_state(page_key: str, **updates) -> None:
    """
    Update specific fields in page state.
    
    Args:
        page_key: Unique key for this page's state
        **updates: Field-value pairs to update
    """
    if page_key in st.session_state:
        state = st.session_state[page_key]
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)


def reset_page_state(page_key: str, state_class: type, **defaults) -> Any:
    """
    Reset page state to defaults.
    
    Args:
        page_key: Unique key for this page's state
        state_class: Dataclass type for the state
        **defaults: Default values for state initialization
    
    Returns:
        New instance of state_class
    """
    st.session_state[page_key] = state_class(**defaults)
    return st.session_state[page_key]


# =============================================================================
# PAGE-SPECIFIC STATE CLASSES
# =============================================================================

@dataclass
class BaselineModelState:
    """State for baseline model pages (page 5 and 6)."""
    # UI state
    show_report: bool = False
    show_logloss: bool = False
    show_confusion: bool = False
    model_loaded: bool = False
    
    # Model and data
    model: Any = None
    X_test: Any = None
    y_test: Any = None
    results: Any = None
    
    # Sample selection
    selected_sample: Any = None
    selected_sample_idx: Any = None
    selected_sample_label: Any = None
    
    # Normal/abnormal comparison
    normal_sample: Any = None
    normal_sample_idx: Any = None
    abnormal_sample: Any = None
    abnormal_sample_idx: Any = None
    abnormal_sample_label: Any = None


@dataclass 
class DeepLearningState:
    """State for deep learning model pages (page 8 and 9)."""
    model_loaded: bool = False
    model: Any = None
    X_test: Any = None
    y_test: Any = None
    predictions: Any = None
    show_predictions: bool = False


@dataclass
class SHAPState:
    """State for SHAP analysis pages (page 11 and 12)."""
    selected_class: int = 0
    selected_example: int = 1
    show_summary: bool = False
    show_waterfall: bool = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def init_baseline_state(page_prefix: str) -> BaselineModelState:
    """Initialize baseline model state for a page."""
    return get_page_state(
        f"{page_prefix}_state",
        BaselineModelState
    )


def init_dl_state(page_prefix: str) -> DeepLearningState:
    """Initialize deep learning state for a page."""
    return get_page_state(
        f"{page_prefix}_state", 
        DeepLearningState
    )


def init_shap_state(page_prefix: str) -> SHAPState:
    """Initialize SHAP analysis state for a page."""
    return get_page_state(
        f"{page_prefix}_state",
        SHAPState
    )

