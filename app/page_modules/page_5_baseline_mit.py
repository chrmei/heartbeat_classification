"""
Page 5: Results baseline models MIT
Include "prediction" on live examples (user can choose randomly or a certain row and select a class)
Classification Report, Confusion Matrix (interactive)
Todo by Christian
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.visualization import plot_heartbeat
from src.utils.preprocessing import MITBIH_LABELS_MAP, MITBIH_LABELS_TO_DESC


def smart_format(x):
    # Nicht-numerische Werte einfach unverändert lassen
    if not isinstance(x, (int, float, np.integer, np.floating)):
        return x

    # Float → prüfen, ob Nachkommaanteil 0 ist
    if isinstance(x, float) and x.is_integer():
        return f"{int(x)}"

    # Normaler Float mit vier Nachkommastellen
    return f"{x:.4f}"


def render():
    # Use page-specific session state keys to avoid conflicts with other pages
    PREFIX = "page5_"

    st.session_state.setdefault(f"{PREFIX}show_report", False)
    st.session_state.setdefault(f"{PREFIX}show_logloss", False)
    st.session_state.setdefault(f"{PREFIX}show_confusion", False)
    st.session_state.setdefault(f"{PREFIX}model", None)
    st.session_state.setdefault(f"{PREFIX}X_test", None)
    st.session_state.setdefault(f"{PREFIX}y_test", None)
    st.session_state.setdefault(f"{PREFIX}results", None)
    st.session_state.setdefault(f"{PREFIX}selected_sample", None)
    st.session_state.setdefault(f"{PREFIX}selected_sample_idx", None)
    st.session_state.setdefault(f"{PREFIX}selected_sample_label", None)
    st.session_state.setdefault(f"{PREFIX}model_loaded", False)
    st.session_state.setdefault(f"{PREFIX}normal_sample", None)
    st.session_state.setdefault(f"{PREFIX}normal_sample_idx", None)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample", None)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample_idx", None)
    st.session_state.setdefault(f"{PREFIX}abnormal_sample_label", None)

    st.title("5: Baseline Models Results - MIT Dataset")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Results Overview", "Model Evaluation", "Model Prediction"])

    with tab1:
        _render_results_overview_tab()

    with tab2:
        _render_model_evaluation_tab()

    with tab3:
        _render_example_prediction_tab()


def _render_results_overview_tab():
    """Render the Results Overview tab"""

    st.write(
        """
        - Randomized Search: Best Models = XGBoost, ANN, SVM.
        - All modeels trained with 5-fold CV, scoring='f1_macro'.
        - SMOTE was chosen for oversampling underrepresented classes, generating synthetic samples without creating duplicates.
        - GridSearch for these 3 Models. Best Model: XGBoost
        """
    )

    # -------------------------------------------------------------
    # RANDOMIZED SEARCH RESULTS SECTION
    # -------------------------------------------------------------

    st.markdown("---")
    st.header("Results Overview - Baseline Models")

    with st.expander("Randomized Search - No Extreme Value Removal", expanded=False):

        st.subheader("Randomized Search - No  xtreme Value Removal")

        # Hard-coded file path (adjust to your real path)
        RESULTS_PATH = "app/tables/page_5/A_02_02_reduced.csv"

        if not os.path.exists(RESULTS_PATH):
            st.error(f"File not found: {RESULTS_PATH}")
        else:
            try:
                df = pd.read_csv(RESULTS_PATH)
                st.write(
                    f"Loaded results from: `{RESULTS_PATH}`. Highlighted results are selected for GridSearch"
                )
                # st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

                df.sort_values(by="F1 Test", ascending=False, inplace=True)
                df.reset_index(drop=True, inplace=True)

                HIGHLGHT_INDICES = {3, 5, 10}

                def highlight_specific(row):
                    if row.name in HIGHLGHT_INDICES:
                        return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
                    return [""] * len(row)

                styled_df = df.style.apply(highlight_specific, axis=1)

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600,
                )

            except Exception as e:
                st.error(f"Failed to load CSV: {e}")

    # -------------------------------------------------------------
    # GRID SEARCH RESULTS SECTION
    # -------------------------------------------------------------
    with st.expander("Grid Search - No Extreme Value  Removal", expanded=False):

        st.header("Results Overview - Baseline Models")
        st.subheader("Grid Search - No Extreme Value  Removal")
        df = pd.read_csv("app/tables/page_5/result_baseline_gridsearch.csv")
        st.write(
            "Loaded results from: app/tables/page_5/result_baseline_gridsearch.csv. Highlighted result is the best 'baseline' model selected for final evaluation."
        )

        HIGHLGHT_INDICES = {0}

        def highlight_specific(row):
            if row.name in HIGHLGHT_INDICES:
                return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
            return [""] * len(row)

        styled_df = df.style.apply(highlight_specific, axis=1)

        st.dataframe(styled_df, use_container_width=True)


def _render_model_evaluation_tab():
    """Render the Model Evaluation tab"""
    PREFIX = "page5_"

    st.header("Model Evaluation – MIT-BIH XGBoost")
    st.markdown("---")

    # -------------------------------------------------------
    # Button 1 — Load Model + Data
    # -------------------------------------------------------
    st.header("Step 1 – Load Test Data & Model")

    if st.session_state[f"{PREFIX}model_loaded"] and st.session_state[f"{PREFIX}model"] is not None:
        st.success("✅ Model and data are already loaded.")
        st.write(f"**Dataset size:** {st.session_state[f'{PREFIX}X_test'].shape}")
        st.write(
            f"**Model file size:** ~{round((os.path.getsize('models/MIT_04_final_evaluation/XGBoost_smote_outliers_False.json')/1024), 2)} KB"
        )

        # Class distribution pie chart
        class_counts = st.session_state[f"{PREFIX}y_test"].value_counts().sort_index()
        # Convert to int and filter to only valid MIT classes (0-4)
        valid_classes = [int(i) for i in class_counts.index if int(i) in MITBIH_LABELS_MAP]
        labels = [
            f"{MITBIH_LABELS_MAP[i]} ({MITBIH_LABELS_TO_DESC[MITBIH_LABELS_MAP[i]]})"
            for i in valid_classes
        ]
        colors = plt.cm.Set3(range(len(valid_classes)))
        # Filter class_counts to match valid_classes
        class_counts = class_counts.loc[
            [i for i in class_counts.index if int(i) in MITBIH_LABELS_MAP]
        ]

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
            # Load test data
            df_mitbih_test = pd.read_csv("data/original/mitbih_test.csv", header=None)
            X_test = df_mitbih_test.drop(187, axis=1)
            y_test = df_mitbih_test[187]

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
                objective="multi:softprob",
                num_class=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
                **PARAMS,
            )

            # Load trained model
            model.load_model("models/MIT_04_final_evaluation/XGBoost_smote_outliers_False.json")

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
                f"**Model file size:** ~{round((os.path.getsize('models/MIT_04_final_evaluation/XGBoost_smote_outliers_False.json')/1024), 2)} KB"
            )

            # Class distribution pie chart
            class_counts = y_test.value_counts().sort_index()
            # Convert to int and filter to only valid MIT classes (0-4)
            valid_classes = [int(i) for i in class_counts.index if int(i) in MITBIH_LABELS_MAP]
            labels = [
                f"{MITBIH_LABELS_MAP[i]} ({MITBIH_LABELS_TO_DESC[MITBIH_LABELS_MAP[i]]})"
                for i in valid_classes
            ]
            colors = plt.cm.Set3(range(len(valid_classes)))
            # Filter class_counts to match valid_classes
            class_counts = class_counts.loc[
                [i for i in class_counts.index if int(i) in MITBIH_LABELS_MAP]
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
    st.header("Step 2 – Classification Report")

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

        st.metric("F1-Macro", f"{f1_macro:.4f}")
        st.metric("Precision-Macro", f"{prec_macro:.4f}")
        st.metric("Recall-Macro", f"{rec_macro:.4f}")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # --- accuracy-Zeile säubern ---
        if "accuracy" in report_df.index:
            for col in ["precision", "recall"]:
                if col in report_df.columns:
                    report_df.at["accuracy", col] = ""  # leeren
            for col in ["support"]:
                if col in report_df.columns:
                    report_df.at["accuracy", col] = report_df.at[
                        "macro avg", col
                    ]  # support of macro avg
        st.subheader("Classification Report")
        st.dataframe(report_df.style.format(smart_format))

    st.markdown("---")

    # -------------------------------------------------------
    # Button 3 — Logloss Curve
    # -------------------------------------------------------
    st.header("Step 3 – Log Loss Evaluation History")

    if st.button("📈 Show Log-Loss Plot"):
        st.session_state[f"{PREFIX}show_logloss"] = True

    if st.session_state[f"{PREFIX}show_logloss"]:
        image_path = "app/images/page_5/MIT_MODEL/XGBoost_Loss_ON_MIT.png"
        try:
            st.image(image_path, caption="XGBoost Log-Loss Curve (precomputed)", width=600)
        except Exception:
            st.error("Image not found. Check the path: " + image_path)

    st.markdown("---")

    # -------------------------------------------------------
    # Button 4 — Confusion Matrix
    # -------------------------------------------------------
    st.header("Step 4 – Confusion Matrix")

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
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            st.pyplot(fig, width=600)


def _render_example_prediction_tab():
    """Render the Example Prediction tab"""
    PREFIX = "page5_"

    st.header("Model Prediction - XGBoost")

    st.write(
        """
        Compare predictions for a normal heartbeat (Class 0) and an abnormal heartbeat.
        The normal sample is fixed, while you can select which abnormal class to display.
        """
    )

    # Load model and data if not already loaded
    if st.session_state[f"{PREFIX}model"] is None or st.session_state[f"{PREFIX}X_test"] is None:
        st.info(
            "⚠️ Model and data need to be loaded first. Please go to 'Results Overview & Model Evaluation' tab and click 'Load Test Data & Model'."
        )
        if st.button("🔄 Load Model & Data Now"):
            _load_model_and_data()

    if st.session_state[f"{PREFIX}model"] is None or st.session_state[f"{PREFIX}X_test"] is None:
        return

    st.markdown("---")

    # Get class 0 indices for normal samples
    class_0_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == 0
    ].index.tolist()
    max_n_normal = len(class_0_indices)

    # Initialize normal sample (class 0) - fixed to first if not set
    if st.session_state[f"{PREFIX}normal_sample"] is None and max_n_normal > 0:
        normal_idx = class_0_indices[0]
        st.session_state[f"{PREFIX}normal_sample"] = st.session_state[f"{PREFIX}X_test"].loc[
            normal_idx
        ]
        st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

    # Selection for abnormal class
    st.subheader("Select Abnormal Class")

    # Only show abnormal classes (1, 2, 3, 4)
    abnormal_class_options = {
        f"{MITBIH_LABELS_MAP[i]} - {MITBIH_LABELS_TO_DESC[MITBIH_LABELS_MAP[i]]}": i
        for i in range(1, 5)  # Only classes 1-4
    }

    selected_abnormal_class_label = st.selectbox(
        "Select abnormal class:",
        options=list(abnormal_class_options.keys()),
        key="abnormal_class_selection",
    )
    selected_abnormal_class = abnormal_class_options[selected_abnormal_class_label]

    # Clear abnormal sample if it doesn't match the selected class
    if (
        st.session_state[f"{PREFIX}abnormal_sample_label"] is not None
        and st.session_state[f"{PREFIX}abnormal_sample_label"] != selected_abnormal_class
    ):
        st.session_state[f"{PREFIX}abnormal_sample"] = None
        st.session_state[f"{PREFIX}abnormal_sample_idx"] = None
        st.session_state[f"{PREFIX}abnormal_sample_label"] = None

    # Selection method for abnormal class
    selection_method = st.radio(
        "Selection method for abnormal class:",
        ["Random sample", "Nth occurrence"],
        key="selection_method",
    )

    # Get available indices for the selected abnormal class
    abnormal_class_indices = st.session_state[f"{PREFIX}y_test"][
        st.session_state[f"{PREFIX}y_test"] == selected_abnormal_class
    ].index.tolist()
    max_n = len(abnormal_class_indices)

    if max_n == 0:
        st.warning(f"No samples found for class {MITBIH_LABELS_MAP[selected_abnormal_class]}.")
        abnormal_sample = None
        abnormal_idx = None
        abnormal_label = None
    else:
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

                random_pos = random.randint(0, max_n - 1)
                abnormal_idx = abnormal_class_indices[random_pos]
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

            # Abnormal class nth occurrence
            n_occurrence = st.number_input(
                f"Abnormal class ({MITBIH_LABELS_MAP[selected_abnormal_class]}) - Nth occurrence (1 to {max_n}):",
                min_value=1,
                max_value=max_n,
                value=1,
                key="n_occurrence",
            )

            if st.button("🔍 Get Samples"):
                # Set normal sample
                if max_n_normal > 0 and n_occurrence_normal <= max_n_normal:
                    normal_idx = class_0_indices[n_occurrence_normal - 1]
                    st.session_state[f"{PREFIX}normal_sample"] = st.session_state[
                        f"{PREFIX}X_test"
                    ].loc[normal_idx]
                    st.session_state[f"{PREFIX}normal_sample_idx"] = normal_idx

                # Set abnormal sample
                if n_occurrence <= max_n:
                    abnormal_idx = abnormal_class_indices[n_occurrence - 1]
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
        st.markdown("---")
        st.subheader("ECG Signal Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_pred_str = MITBIH_LABELS_MAP[normal_prediction]
            fig1 = plot_heartbeat(
                normal_sample.values,
                title=f"Predicted: {normal_pred_str} ({MITBIH_LABELS_TO_DESC[normal_pred_str]})",
                color="green" if normal_prediction == normal_true_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig1, width=600)

        with col2:
            st.write(f"**Abnormal Sample (Class {MITBIH_LABELS_MAP[abnormal_label]})**")
            abnormal_pred_str = MITBIH_LABELS_MAP[abnormal_prediction]
            fig2 = plot_heartbeat(
                abnormal_sample.values,
                title=f"Predicted: {abnormal_pred_str} ({MITBIH_LABELS_TO_DESC[abnormal_pred_str]})",
                color="green" if abnormal_prediction == abnormal_label else "red",
                figsize=(8, 5),
            )
            st.pyplot(fig2, width=600)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Normal Sample (Class 0)**")
            normal_true_str = MITBIH_LABELS_MAP[normal_true_label]
            normal_pred_str = MITBIH_LABELS_MAP[normal_prediction]

            st.markdown("**True Label:**")
            st.info(
                f"Class {normal_true_label}: {normal_true_str} ({MITBIH_LABELS_TO_DESC[normal_true_str]})"
            )

            st.markdown("**Predicted Label:**")
            if normal_prediction == normal_true_label:
                st.success(
                    f"Class {normal_prediction}: {normal_pred_str} ({MITBIH_LABELS_TO_DESC[normal_pred_str]}) ✓"
                )
            else:
                st.error(
                    f"Class {normal_prediction}: {normal_pred_str} ({MITBIH_LABELS_TO_DESC[normal_pred_str]}) ✗"
                )

            st.write(f"**Sample Index:** {normal_idx}")

        with col2:
            st.write(f"**Abnormal Sample (Class {MITBIH_LABELS_MAP[abnormal_label]})**")
            abnormal_true_str = MITBIH_LABELS_MAP[abnormal_label]
            abnormal_pred_str = MITBIH_LABELS_MAP[abnormal_prediction]

            st.markdown("**True Label:**")
            st.info(
                f"Class {abnormal_label}: {abnormal_true_str} ({MITBIH_LABELS_TO_DESC[abnormal_true_str]})"
            )

            st.markdown("**Predicted Label:**")
            if abnormal_prediction == abnormal_label:
                st.success(
                    f"Class {abnormal_prediction}: {abnormal_pred_str} ({MITBIH_LABELS_TO_DESC[abnormal_pred_str]}) ✓"
                )
            else:
                st.error(
                    f"Class {abnormal_prediction}: {abnormal_pred_str} ({MITBIH_LABELS_TO_DESC[abnormal_pred_str]}) ✗"
                )

            st.write(f"**Sample Index:** {abnormal_idx}")

        # Display probabilities tables
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Normal Sample Probabilities")
            normal_prob_df = pd.DataFrame(
                {
                    "Class": [MITBIH_LABELS_MAP[i] for i in range(5)],
                    "Description": [MITBIH_LABELS_TO_DESC[MITBIH_LABELS_MAP[i]] for i in range(5)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in normal_probabilities],
                    "_sort": normal_probabilities,  # Numeric column for sorting
                }
            )
            normal_prob_df = normal_prob_df.sort_values("_sort", ascending=False)
            normal_prob_df = normal_prob_df.drop(
                columns=["_sort"]
            )  # Remove sorting column before display
            st.dataframe(normal_prob_df, use_container_width=True)

        with col2:
            st.subheader("Abnormal Sample Probabilities")
            abnormal_prob_df = pd.DataFrame(
                {
                    "Class": [MITBIH_LABELS_MAP[i] for i in range(5)],
                    "Description": [MITBIH_LABELS_TO_DESC[MITBIH_LABELS_MAP[i]] for i in range(5)],
                    "Probability": [f"{prob * 100:.2f}%" for prob in abnormal_probabilities],
                    "_sort": abnormal_probabilities,  # Numeric column for sorting
                }
            )
            abnormal_prob_df = abnormal_prob_df.sort_values("_sort", ascending=False)
            abnormal_prob_df = abnormal_prob_df.drop(
                columns=["_sort"]
            )  # Remove sorting column before display
            st.dataframe(abnormal_prob_df, use_container_width=True)


def _load_model_and_data():
    """Helper function to load model and data"""
    PREFIX = "page5_"
    try:
        # Load test data
        df_mitbih_test = pd.read_csv("data/original/mitbih_test.csv", header=None)
        X_test = df_mitbih_test.drop(187, axis=1)
        y_test = df_mitbih_test[187]

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
            objective="multi:softprob",
            num_class=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
            **PARAMS,
        )

        # Load trained model
        model.load_model("models/MIT_04_final_evaluation/XGBoost_smote_outliers_False.json")

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
