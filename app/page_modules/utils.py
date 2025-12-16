"""
Shared utility functions for Streamlit page modules.
Centralizes common functionality to eliminate code duplication.
"""

import base64
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image.

    Args:
        image_path: Path to the image file
        alt: Alt text for the image
        caption: Optional caption to display below the image

    Returns:
        HTML string with embedded base64 image
    """
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


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

# Base paths - computed relative to this file
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR.parent / "data" / "original"
MODELS_DIR = APP_DIR.parent / "models"


@st.cache_data
def load_mitbih_test_data() -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load MIT-BIH test data with caching.

    Returns:
        Tuple of (X_test, y_test) or (None, None) on error
    """
    try:
        df_mitbih_test = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)
        X_test = df_mitbih_test.drop(187, axis=1)
        y_test = df_mitbih_test[187]
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading MIT-BIH test data: {e}")
        return None, None


@st.cache_data
def load_mitbih_train_data() -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load MIT-BIH training data with caching.

    Returns:
        Tuple of (X_train, y_train) or (None, None) on error
    """
    try:
        df_mitbih_train = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
        X_train = df_mitbih_train.drop(187, axis=1)
        y_train = df_mitbih_train[187]
        return X_train, y_train
    except Exception as e:
        st.error(f"Error loading MIT-BIH training data: {e}")
        return None, None


@st.cache_data
def load_ptb_data() -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load PTB dataset (combined normal and abnormal) with caching.

    Returns:
        Tuple of (X, y) or (None, None) on error
    """
    try:
        df_normal = pd.read_csv(DATA_DIR / "ptbdb_normal.csv", header=None)
        df_abnormal = pd.read_csv(DATA_DIR / "ptbdb_abnormal.csv", header=None)

        # Combine datasets
        df_ptb = pd.concat([df_normal, df_abnormal], ignore_index=True)
        X = df_ptb.drop(187, axis=1)
        y = df_ptb[187]
        return X, y
    except Exception as e:
        st.error(f"Error loading PTB data: {e}")
        return None, None


@st.cache_resource
def load_keras_model(model_path: str):
    """Load Keras model with caching.

    Uses standalone Keras 3 API for compatibility with models saved
    using Keras 3.x (which uses keras.src.models.functional internally).

    Args:
        model_path: Path to the .keras model file

    Returns:
        Loaded Keras model or None on error
    """
    try:
        import keras

        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None


@st.cache_resource
def load_xgboost_model(model_path: str, params: dict = None):
    """Load XGBoost model with caching.

    Args:
        model_path: Path to the .json model file
        params: Optional model parameters for XGBClassifier

    Returns:
        Loaded XGBoost model or None on error
    """
    try:
        import xgboost as xgb

        default_params = {
            "n_estimators": 500,
            "max_depth": 9,
            "learning_rate": 0.2,
            "subsample": 0.7,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 0.05,
            "min_child_weight": 5,
            "gamma": 0.0,
        }

        if params:
            default_params.update(params)

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            **default_params,
        )
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None
