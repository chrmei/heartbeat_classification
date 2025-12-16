"""
Page 6: Results baseline models PTB
Same as Page 5 but on PTB
Todo by Christian
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import base64
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.visualization import plot_heartbeat
from page_modules.state_utils import init_baseline_state, BaselineModelState
from page_modules.styles import apply_matplotlib_style, COLORS

# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
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


# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_6"
TABLES_DIR = APP_DIR / "tables" / "page_6"
DATA_DIR = APP_DIR.parent / "data" / "processed" / "ptb"
MODELS_DIR = APP_DIR.parent / "models"

# Page-specific state key
PAGE_KEY = "page6"

# PTB label mappings (binary classification)
PTB_LABELS_MAP = {0: "Normal", 1: "Abnormal"}
PTB_LABELS_TO_DESC = {
    "Normal": "Normal heartbeat",
    "Abnormal": "Myocardial infarction",
}


def smart_format(x):
    """Format numbers intelligently for display."""
    if not isinstance(x, (int, float, np.integer, np.floating)):
        return x
    if isinstance(x, float) and x.is_integer():
        return f"{int(x)}"
    return f"{x:.4f}"


def get_metric_color_gradient(value: float) -> str:
    """
    Generate a CSS gradient based on metric value.
    0.0 = red, 0.5 = yellow, 1.0 = green
    Returns a CSS linear-gradient string.
    """
    value = max(0.0, min(1.0, value))  # Clamp between 0 and 1

    if value <= 0.5:
        # Interpolate between red and yellow
        ratio = value / 0.5
        r = 220
        g = int(53 + (180 - 53) * ratio)  # 53 -> 180
        b = int(69 + (0 - 69) * ratio)  # 69 -> 0
        r2 = 180
        g2 = int(30 + (140 - 30) * ratio)
        b2 = int(40 + (0 - 40) * ratio)
    else:
        # Interpolate between yellow and green
        ratio = (value - 0.5) / 0.5
        r = int(220 - (220 - 45) * ratio)  # 220 -> 45
        g = int(180 + (106 - 180) * ratio)  # 180 -> 106
        b = int(0 + (79 - 0) * ratio)  # 0 -> 79
        r2 = int(180 - (180 - 27) * ratio)
        g2 = int(140 + (67 - 140) * ratio)
        b2 = int(0 + (50 - 0) * ratio)

    return f"linear-gradient(135deg, rgb({r}, {g}, {b}) 0%, rgb({r2}, {g2}, {b2}) 100%)"


def get_state() -> BaselineModelState:
    """Get or initialize page state."""
    return init_baseline_state(PAGE_KEY)


def render_citations():
    """Render citations section with horizontal separator."""
    st.markdown("---")
    with st.expander("📚 Citations", expanded=False):
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']};">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[3]</strong> Kachuee M, Fazeli S, Sarrafzadeh M. (2018). <em>ECG Heartbeat Classification: 
                    A Deep Transferable Representation</em>. 
                    <a href="https://arxiv.org/abs/1805.00794" style="color: {COLORS['clinical_blue_light']};">arXiv:1805.00794</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render():
    # Apply consistent matplotlib styling
    apply_matplotlib_style()

    # Initialize state using dataclass pattern (new approach)
    state = get_state()

    # Use page-specific session state keys to avoid conflicts with other pages
    PREFIX = "page6_"

    # Sync dataclass state to session state for existing code
    st.session_state.setdefault(f"{PREFIX}show_report", state.show_report)
    st.session_state.setdefault(f"{PREFIX}show_logloss", state.show_logloss)
    st.session_state.setdefault(f"{PREFIX}show_confusion", state.show_confusion)
    st.session_state.setdefault(f"{PREFIX}model", state.model)
    st.session_state.setdefault(f"{PREFIX}X_test", state.X_test)
    st.session_state.setdefault(f"{PREFIX}y_test", state.y_test)
    st.session_state.setdefault(f"{PREFIX}results", state.results)
    st.session_state.setdefault(f"{PREFIX}selected_sample", state.selected_sample)
    st.session_state.setdefault(f"{PREFIX}selected_sample_idx", state.selected_sample_idx)
    st.session_state.setdefault(f"{PREFIX}selected_sample_label", state.selected_sample_label)
    st.session_state.setdefault(f"{PREFIX}model_loaded", state.model_loaded)
    st.session_state.setdefault(f"{PREFIX}normal_sample", state.normal_sample)
    st.session_state.setdefault(f"{PREFIX}normal_sample_idx", state.normal_sample_idx)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample", state.abnormal_sample)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample_idx", state.abnormal_sample_idx)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample_label", state.abnormal_sample_label)

    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🫀 Baseline Models Results - PTB Dataset</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Results Overview", "🔬 Model Evaluation", "🔮 Model Prediction"]
    )

    with tab1:
        _render_results_overview_tab()
        render_citations()

    with tab2:
        _render_model_evaluation_tab()
        render_citations()

    with tab3:
        _render_example_prediction_tab()
        render_citations()


def _render_results_overview_tab():
    """Render the Results Overview tab"""

    # Summary container with key methodology points
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1.5rem;">
            <h4 style="color: white; margin-top: 0; font-size: 1.2rem;">📋 Methodology Summary</h4>
            <ul style="opacity: 0.95; margin-bottom: 0; padding-left: 1.25rem;">
                <li><strong>Simpler, quicker approach:</strong> LazyClassifier instead of randomized search</li>
                <li>All models trained with <strong>SMOTE data</strong> for comparison</li>
                <li><strong>GridSearch</strong> for best model → Best Model: <strong>XGBoost</strong></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # LAZYCLASSIFIER RESULTS SECTION
    # -------------------------------------------------------------

    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📊 Results Overview - Baseline Models</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("LazyClassifier - SMOTE Data", expanded=False):

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🎲 LazyClassifier - SMOTE Data</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hard-coded file path
        RESULTS_PATH = str(TABLES_DIR / "ptb_lazy_classifier_results.csv")

        if not os.path.exists(RESULTS_PATH):
            st.error(f"File not found: {RESULTS_PATH}")
        else:
            try:
                df = pd.read_csv(RESULTS_PATH)
                st.write(
                    f"Loaded results from: `{RESULTS_PATH}`. Highlighted result (XGBoost) is selected for GridSearch"
                )

                # Sort by F1 Score (or Accuracy) descending
                if "Balanced Accuracy" in df.columns:
                    df.sort_values(by="Balanced Accuracy", ascending=False, inplace=True)
                elif "F1 Score" in df.columns:
                    df.sort_values(by="F1 Score", ascending=False, inplace=True)
                elif "Accuracy" in df.columns:
                    df.sort_values(by="Accuracy", ascending=False, inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Highlight XGBoost (should be first after sorting)
                HIGHLGHT_INDICES = {0} if "XGBClassifier" in df["Model"].iloc[0] else set()

                def highlight_specific(row):
                    if row.name in HIGHLGHT_INDICES:
                        return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
                    return [""] * len(row)

                styled_df = df.style.apply(highlight_specific, axis=1)

                st.dataframe(
                    styled_df,
                    width="stretch",
                    height=600,
                )

            except Exception as e:
                st.error(f"Failed to load CSV: {e}")


def _render_model_evaluation_tab():
    """Render the Model Evaluation tab"""
    PREFIX = "page6_"

    # Hero header for Model Evaluation
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔬 Model Evaluation – PTB XGBoost</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -------------------------------------------------------
    # Button 1 — Load Model + Data
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">📥 Step 1 – Load Test Data & Model</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state[f"{PREFIX}model_loaded"] and st.session_state[f"{PREFIX}model"] is not None:
        st.success("✅ Model and data are already loaded.")
        st.write(f"**Dataset size:** {st.session_state[f'{PREFIX}X_test'].shape}")
        st.write(
            f"**Model file size:** ~{round((os.path.getsize(MODELS_DIR / 'PTB_03_final_evaluation' / 'XGBoost_smote_outliers_False.json')/1024), 2)} KB"
        )

        # Class distribution pie chart
        class_counts = st.session_state[f"{PREFIX}y_test"].value_counts().sort_index()
        # Convert to int and filter to only valid PTB classes (0, 1)
        valid_classes = [int(i) for i in class_counts.index if int(i) in PTB_LABELS_MAP]
        labels = [PTB_LABELS_MAP[i] for i in valid_classes]
        colors = ["#66b3ff", "#ff9999"]  # Blue for Normal, Red for Abnormal
        # Filter class_counts to match valid_classes
        class_counts = class_counts.loc[[i for i in class_counts.index if int(i) in PTB_LABELS_MAP]]

        fig_pie, ax_pie = plt.subplots(figsize=(8, 4))  # Wide format
        ax_pie.pie(
            class_counts.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
        )
        ax_pie.set_title("Test Data Class Distribution", fontsize=12, pad=10)
        st.pyplot(fig_pie, width=400)

        if st.button("🔄 Reload Model & Data"):
            st.session_state[f"{PREFIX}model_loaded"] = False
            st.session_state[f"{PREFIX}model"] = None
            st.session_state[f"{PREFIX}X_test"] = None
            st.session_state[f"{PREFIX}y_test"] = None
            st.rerun()
    elif st.button("📥 Load Test Data & Model"):
        try:
            # Load PTB test data from processed directory
            X_test = pd.read_csv(DATA_DIR / "X_ptb_test.csv")
            y_test = pd.read_csv(DATA_DIR / "y_ptb_test.csv").iloc[:, 0].astype(int)

            # Create model
            RANDOM_STATE = 42
            PARAMS = {
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

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                **PARAMS,
            )

            # Load trained model
            model.load_model(
                str(MODELS_DIR / "PTB_03_final_evaluation" / "XGBoost_smote_outliers_False.json")
            )

            # Save to session state
            st.session_state[f"{PREFIX}X_test"] = X_test
            st.session_state[f"{PREFIX}y_test"] = y_test
            st.session_state[f"{PREFIX}model"] = model
            st.session_state[f"{PREFIX}results"] = None  # reset eval history
            st.session_state[f"{PREFIX}model_loaded"] = True

            # Stats
            st.success("Model & Data successfully loaded.")
            st.write(f"**Dataset size:** {X_test.shape}")
            st.write(
                f"**Model file size:** ~{round((os.path.getsize(MODELS_DIR / 'PTB_03_final_evaluation' / 'XGBoost_smote_outliers_False.json')/1024), 2)} KB"
            )

            # Class distribution pie chart
            class_counts = y_test.value_counts().sort_index()
            # Convert to int and filter to only valid PTB classes (0, 1)
            valid_classes = [int(i) for i in class_counts.index if int(i) in PTB_LABELS_MAP]
            labels = [PTB_LABELS_MAP[i] for i in valid_classes]
            colors = ["#66b3ff", "#ff9999"]  # Blue for Normal, Red for Abnormal
            # Filter class_counts to match valid_classes
            class_counts = class_counts.loc[
                [i for i in class_counts.index if int(i) in PTB_LABELS_MAP]
            ]

            fig_pie, ax_pie = plt.subplots(figsize=(8, 4))  # Wide format
            ax_pie.pie(
                class_counts.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
            )
            ax_pie.set_title("Test Data Class Distribution", fontsize=12, pad=10)
            st.pyplot(fig_pie, width=400)

            st.rerun()

        except Exception as e:
            st.error(f"Error loading model/data: {str(e)}")

    st.markdown("---")

    # -------------------------------------------------------
    # Button 2 — Classification Report
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">📊 Step 2 – Classification Report</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📊 Generate Classification Report"):
        st.session_state[f"{PREFIX}show_report"] = True
    if st.session_state[f"{PREFIX}show_report"]:
        model = st.session_state[f"{PREFIX}model"]
        X_test = st.session_state[f"{PREFIX}X_test"]
        y_test = st.session_state[f"{PREFIX}y_test"]

        y_pred = model.predict(X_test)

        # macro metrics
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        # Display metrics in a styled row with value-dependent colors
        f1_gradient = get_metric_color_gradient(f1_macro)
        prec_gradient = get_metric_color_gradient(prec_macro)
        rec_gradient = get_metric_color_gradient(rec_macro)

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: {f1_gradient}; 
                            padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                    <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{f1_macro:.4f}</div>
                    <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">F1-Macro</div>
                </div>
                <div style="background: {prec_gradient}; 
                            padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                    <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{prec_macro:.4f}</div>
                    <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Precision-Macro</div>
                </div>
                <div style="background: {rec_gradient}; 
                            padding: 1.25rem; border-radius: 12px; text-align: center; color: white;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                    <div style="font-size: 2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{rec_macro:.4f}</div>
                    <div style="font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Recall-Macro</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # --- accuracy-Zeile säubern ---
        if "accuracy" in report_df.index:
            # Use np.nan instead of empty strings to avoid Arrow serialization issues
            for col in ["precision", "recall"]:
                if col in report_df.columns:
                    report_df.at["accuracy", col] = np.nan  # Use NaN instead of empty string
            for col in ["support"]:
                if col in report_df.columns:
                    report_df.at["accuracy", col] = report_df.at[
                        "macro avg", col
                    ]  # support of macro avg

        # Format the dataframe, handling NaN values
        def format_with_nan(val):
            if pd.isna(val):
                return ""
            return smart_format(val)

        st.markdown("**📋 Classification Report**")
        st.dataframe(report_df.style.format(format_with_nan))

    st.markdown("---")

    # -------------------------------------------------------
    # Button 3 — Logloss Curve
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">📈 Step 3 – Log Loss Evaluation History</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📈 Show Log-Loss Plot"):
        st.session_state[f"{PREFIX}show_logloss"] = True

    if st.session_state[f"{PREFIX}show_logloss"]:
        image_path = str(IMAGES_DIR / "XGBoost_Loss_ON_PTB.png")
        try:
            st.image(image_path, caption="XGBoost Log-Loss Curve (precomputed)", width=600)
        except Exception:
            st.error("Image not found. Check the path: " + image_path)

    st.markdown("---")

    # -------------------------------------------------------
    # Button 4 — Confusion Matrix
    # -------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">🧩 Step 4 – Confusion Matrix</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🧩 Show Confusion Matrix"):
        st.session_state[f"{PREFIX}show_confusion"] = True

    if st.session_state[f"{PREFIX}show_confusion"]:
        if st.session_state[f"{PREFIX}model"] is None:
            st.error("Please load the model first.")
        else:
            model = st.session_state[f"{PREFIX}model"]
            X_test = st.session_state[f"{PREFIX}X_test"]
            y_test = st.session_state[f"{PREFIX}y_test"]

            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                cmap="Blues",
                fmt="d",
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"],
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            st.pyplot(fig, width=600)

        render_citations()


def _render_example_prediction_tab():
    """Render the Example Prediction tab"""
    PREFIX = "page6_"

    # Hero header for Model Prediction
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🔮 Model Prediction - XGBoost</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Description in styled container
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 1.25rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1.5rem;">
            <p style="margin: 0; opacity: 0.95;">
                Compare predictions for a <strong>normal heartbeat (Class 0)</strong> and an <strong>abnormal heartbeat (Class 1)</strong>.
                You can select samples randomly or by specific occurrence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model and data if not already loaded
    if st.session_state[f"{PREFIX}model"] is None or st.session_state[f"{PREFIX}X_test"] is None:
        st.info(
            "⚠️ Model and data need to be loaded first. Please go to 'Model Evaluation' tab and click 'Load Test Data & Model'."
        )
        if st.button("🔄 Load Model & Data Now"):
            _load_model_and_data()

    if st.session_state[f"{PREFIX}model"] is None or st.session_state[f"{PREFIX}X_test"] is None:
        return

    # Get class 0 indices for normal samples
    class_0_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == 0
    ].index.tolist()
    max_n_normal = len(class_0_indices)

    # Get class 1 indices for abnormal samples
    class_1_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == 1
    ].index.tolist()
    max_n_abnormal = len(class_1_indices)

    # Initialize normal sample (class 0) - fixed to first if not set
    if st.session_state[f"{PREFIX}normal_sample"] is None and max_n_normal > 0:
        normal_idx = class_0_indices[0]
        st.session_state[f"{PREFIX}normal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[
            normal_idx
        ]
        st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

    # Selection method section header
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; color: white;
                    box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">🎯 Select Sample Method</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selection_method = st.radio(
        "Selection method for abnormal class:",
        ["Random sample", "Nth occurrence"],
        key="selection_method",
    )

    if selection_method == "Random sample":
        if st.button("🔮 Predict!"):
            # Randomize both normal and abnormal samples
            if max_n_normal > 0:
                random_pos_normal = random.randint(0, max_n_normal - 1)
                normal_idx = class_0_indices[random_pos_normal]
                st.session_state[f"{PREFIX}normal_sample"] = st.session_state[
                    f"{PREFIX}X_test"
                ].loc[normal_idx]
                st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

            if max_n_abnormal > 0:
                random_pos_abnormal = random.randint(0, max_n_abnormal - 1)
                abnormal_idx = class_1_indices[random_pos_abnormal]
                abnormal_sample = st.session_state[f"{PREFIX}X_test"].loc[abnormal_idx]
                abnormal_label = st.session_state[f"{PREFIX}y_test"].loc[abnormal_idx]

                st.session_state[f"{PREFIX}abnormal_sample"] = abnormal_sample
                st.session_state[f"{PREFIX}abnormal_sample_idx"] = abnormal_idx
                st.session_state[f"{PREFIX}abnormal_sample_label"] = abnormal_label
    else:  # Nth occurrence
        # Normal class (Class 0) nth occurrence
        if max_n_normal > 0:
            n_occurrence_normal = st.number_input(
                f"Normal class (Class 0) - Nth occurrence (1 to {max_n_normal}):",
                min_value=1,
                max_value=max_n_normal,
                value=1,
                key="n_occurrence_normal",
            )

        # Abnormal class (Class 1) nth occurrence
        if max_n_abnormal > 0:
            n_occurrence_abnormal = st.number_input(
                f"Abnormal class (Class 1) - Nth occurrence (1 to {max_n_abnormal}):",
                min_value=1,
                max_value=max_n_abnormal,
                value=1,
                key="n_occurrence_abnormal",
            )

        if st.button("🔮 Predict!"):
            # Set normal sample
            if max_n_normal > 0 and n_occurrence_normal <= max_n_normal:
                normal_idx = class_0_indices[n_occurrence_normal - 1]
                st.session_state[f"{PREFIX}normal_sample"] = st.session_state[
                    f"{PREFIX}X_test"
                ].loc[normal_idx]
                st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

            # Set abnormal sample
            if max_n_abnormal > 0 and n_occurrence_abnormal <= max_n_abnormal:
                abnormal_idx = class_1_indices[n_occurrence_abnormal - 1]
                abnormal_sample = st.session_state[f"{PREFIX}X_test"].loc[abnormal_idx]
                abnormal_label = st.session_state[f"{PREFIX}y_test"].loc[abnormal_idx]

                st.session_state[f"{PREFIX}abnormal_sample"] = abnormal_sample
                st.session_state[f"{PREFIX}abnormal_sample_idx"] = abnormal_idx
                st.session_state[f"{PREFIX}abnormal_sample_label"] = abnormal_label

    # Use session state if available
    normal_sample = st.session_state[f"{PREFIX}normal_sample"]
    normal_idx = st.session_state[f"{PREFIX}normal_sample_idx"]

    if st.session_state[f"{PREFIX}abnormal_sample"] is not None:
        abnormal_sample = st.session_state[f"{PREFIX}abnormal_sample"]
        abnormal_idx = st.session_state[f"{PREFIX}abnormal_sample_idx"]
        abnormal_label = st.session_state[f"{PREFIX}abnormal_sample_label"]
    else:
        abnormal_sample = None
        abnormal_idx = None
        abnormal_label = None

    # Display predictions if both samples are available
    if normal_sample is not None and abnormal_sample is not None:

        model = st.session_state[f"{PREFIX}model"]

        # Predict for normal sample (class 0)
        normal_array = normal_sample.values.reshape(1, -1)
        normal_prediction = model.predict(normal_array)[0]
        normal_probabilities = model.predict_proba(normal_array)[0]
        normal_true_label = 0  # Always class 0

        # Predict for abnormal sample
        abnormal_array = abnormal_sample.values.reshape(1, -1)
        abnormal_prediction = model.predict(abnormal_array)[0]
        abnormal_probabilities = model.predict_proba(abnormal_array)[0]

        # Side-by-side visualization
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">📈 ECG Signal Visualization</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_pred_str = PTB_LABELS_MAP[normal_prediction]
            fig1 = plot_heartbeat(
                normal_sample.values,
                title=f"Predicted: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]})",
                color="green" if normal_prediction == normal_true_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig1, width=600)

        with col2:
            st.write(f"**Abnormal Sample (Class {PTB_LABELS_MAP[abnormal_label]})**")
            abnormal_pred_str = PTB_LABELS_MAP[abnormal_prediction]
            fig2 = plot_heartbeat(
                abnormal_sample.values,
                title=f"Predicted: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]})",
                color="green" if abnormal_prediction == abnormal_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig2, width=600)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_true_str = PTB_LABELS_MAP[normal_true_label]
            normal_pred_str = PTB_LABELS_MAP[normal_prediction]

            st.markdown("**True Label:**")
            st.info(
                f"Class {normal_true_label}: {normal_true_str} ({PTB_LABELS_TO_DESC[normal_true_str]})"
            )

            st.markdown("**Predicted Label:**")
            if normal_prediction == normal_true_label:
                st.success(
                    f"Class {normal_prediction}: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]}) ✓"
                )
            else:
                st.error(
                    f"Class {normal_prediction}: {normal_pred_str} ({PTB_LABELS_TO_DESC[normal_pred_str]}) ✗"
                )

            st.write(f"**Sample Index:** {normal_idx}")

        with col2:
            st.write(f"**Abnormal Sample (Class {PTB_LABELS_MAP[abnormal_label]})**")
            abnormal_true_str = PTB_LABELS_MAP[abnormal_label]
            abnormal_pred_str = PTB_LABELS_MAP[abnormal_prediction]

            st.markdown("**True Label:**")
            st.info(
                f"Class {abnormal_label}: {abnormal_true_str} ({PTB_LABELS_TO_DESC[abnormal_true_str]})"
            )

            st.markdown("**Predicted Label:**")
            if abnormal_prediction == abnormal_label:
                st.success(
                    f"Class {abnormal_prediction}: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]}) ✓"
                )
            else:
                st.error(
                    f"Class {abnormal_prediction}: {abnormal_pred_str} ({PTB_LABELS_TO_DESC[abnormal_pred_str]}) ✗"
                )

            st.write(f"**Sample Index:** {abnormal_idx}")

        # Display probabilities tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Normal Sample Probabilities**")
            normal_prob_df = pd.DataFrame(
                {
                    "Class": [PTB_LABELS_MAP[i] for i in range(2)],
                    "Description": [PTB_LABELS_TO_DESC[PTB_LABELS_MAP[i]] for i in range(2)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in normal_probabilities],
                    "_sort": normal_probabilities,  # Numeric column for sorting
                }
            )
            normal_prob_df = normal_prob_df.sort_values("_sort", ascending=False)
            normal_prob_df = normal_prob_df.drop(
                columns=["_sort"]
            )  # Remove sorting column before display
            st.dataframe(normal_prob_df, width="stretch")

        with col2:
            st.markdown("**📊 Abnormal Sample Probabilities**")
            abnormal_prob_df = pd.DataFrame(
                {
                    "Class": [PTB_LABELS_MAP[i] for i in range(2)],
                    "Description": [PTB_LABELS_TO_DESC[PTB_LABELS_MAP[i]] for i in range(2)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in abnormal_probabilities],
                    "_sort": abnormal_probabilities,  # Numeric column for sorting
                }
            )
            abnormal_prob_df = abnormal_prob_df.sort_values("_sort", ascending=False)
            abnormal_prob_df = abnormal_prob_df.drop(
                columns=["_sort"]
            )  # Remove sorting column before display
            st.dataframe(abnormal_prob_df, width="stretch")

        render_citations()


def _load_model_and_data():
    """Helper function to load model and data"""
    PREFIX = "page6_"
    try:
        # Load PTB test data from processed directory
        X_test = pd.read_csv(DATA_DIR / "X_ptb_test.csv")
        y_test = pd.read_csv(DATA_DIR / "y_ptb_test.csv").iloc[:, 0].astype(int)

        # Create model
        RANDOM_STATE = 42
        PARAMS = {
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

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            **PARAMS,
        )

        # Load trained model
        model.load_model(
            str(MODELS_DIR / "PTB_03_final_evaluation" / "XGBoost_smote_outliers_False.json")
        )

        # Save to session state
        st.session_state[f"{PREFIX}X_test"] = X_test
        st.session_state[f"{PREFIX}y_test"] = y_test
        st.session_state[f"{PREFIX}model"] = model
        st.session_state[f"{PREFIX}results"] = None  # reset eval history
        st.session_state[f"{PREFIX}model_loaded"] = True

        st.success("Model & Data successfully loaded.")
        st.rerun()

    except Exception as e:
        st.error(f"Error loading model/data: {str(e)}")
