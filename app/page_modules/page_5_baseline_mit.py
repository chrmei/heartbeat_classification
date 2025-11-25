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
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns


def render():
    st.title("Baseline Models Results - MIT Dataset")

    st.write(
        """
        - Randomized Search: Best Models = XGBoost, ANN, SVM. 
        - SMOTE chosen as appropriate oversampling technique (generation of new synthetic samples for underrepresented classes).
        - GridSearch for these 3 Models. Best Model: XGBoost
        """
    )

    # -------------------------------------------------------------
    # RANDOMIZED SEARCH RESULTS SECTION (below all tabs)
    # -------------------------------------------------------------

    st.markdown("---")
    st.header("📊 Results Overview - Baseline Models")

    with st.expander("🔽 Randomized Search - No Outlier Removal", expanded=False):

        st.subheader("Randomized Search - No Outlier Removal")

        # Hard-coded file path (adjust to your real path)
        RESULTS_PATH = "app/tables/A_02_02_reduced.csv"

        if not os.path.exists(RESULTS_PATH):
            st.error(f"File not found: {RESULTS_PATH}")
        else:
            try:
                df = pd.read_csv(RESULTS_PATH)
                st.write(
                    f"Loaded results from: `{RESULTS_PATH}`. Highlighted results are selected for GridSearch"
                )
                # st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

                df_columns = [
                    "model",
                    "sampling_method",
                    "cv_mean_train_f1_macro",
                    "cv_mean_val_f1_macro",
                    "cv_mean_val_bal_acc",
                    "test_f1_macro",
                    "test_accuracy",
                    "test_f1_class_0.0",
                    "test_f1_class_1.0",
                    "test_f1_class_2.0",
                    "test_f1_class_3.0",
                    "test_f1_class_4.0",
                ]

                df = df[df_columns]

                df["diff_train_val"] = df["cv_mean_train_f1_macro"] - df["cv_mean_val_f1_macro"]

                df.rename(
                    {
                        "model": "Model",
                        "sampling_method": "Sampler",
                        "cv_mean_train_f1_macro": "Avg F1 Train",
                        "cv_mean_val_f1_macro": "Avg F1 Val",
                        "diff_train_val": "Diff F1 Train-Val",
                        "cv_mean_val_bal_acc": "Avg bal. accuracy Val",
                        "test_f1_macro": "F1 Test",
                        "test_accuracy": "Accuracy",
                        "test_f1_class_0.0": "1",
                        "test_f1_class_1.0": "2",
                        "test_f1_class_2.0": "3",
                        "test_f1_class_3.0": "4",
                        "test_f1_class_4.0": "5",
                    },
                    inplace=True,
                    axis=1,
                )

                df = df[
                    [
                        "Model",
                        "Sampler",
                        "F1 Test",
                        "Accuracy",
                        "Avg F1 Val",
                        "Avg F1 Train",
                        "Diff F1 Train-Val",
                        "Avg bal. accuracy Val",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                    ]
                ]

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
    # GRID SEARCH RESULTS SECTION (below all tabs)
    # -------------------------------------------------------------
    with st.expander("🔽 Grid Search - No Outlier Removal", expanded=False):

        st.header("📊 Results Overview - Baseline Models")
        st.subheader("Grid Search - No Outlier Removal")
        df = pd.read_csv("app/tables/result_baseline_gridsearch.csv")
        st.write(
            "Loaded results from: app/tables/result_baseline_gridsearch.csv. Highlighted result is the best 'baseline' model selected for final evaluation."
        )

        HIGHLGHT_INDICES = {0}

        def highlight_specific(row):
            if row.name in HIGHLGHT_INDICES:
                return ["background-color: rgba(255, 215, 0, 0.3)"] * len(row)
            return [""] * len(row)

        styled_df = df.style.apply(highlight_specific, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True
        )

    # -------------------------------------------------------
    # Button 1 — Load Model + Data
    # -------------------------------------------------------

    st.markdown("---")

    st.title("Model Evaluation – MIT-BIH XGBoost")
    st.markdown("---")

    st.session_state.setdefault("model", None)
    st.session_state.setdefault("X_test", None)
    st.session_state.setdefault("y_test", None)
    st.session_state.setdefault("results", None)

    # -------------------------------------------------------
    # Button 1 — Load Model + Data
    # -------------------------------------------------------
    st.header("Step 1 – Load Data & Model")

    if st.button("📥 Load Test Data & Model"):
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
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.session_state.results = None  # reset eval history

            # Stats
            st.success("Model & Data successfully loaded.")
            st.write(f"**Dataset size:** {X_test.shape}")
            st.write(
                f"**Model file size:** ~{round((os.path.getsize('models/MIT_04_final_evaluation/XGBoost_smote_outliers_False.json')/1024), 2)} KB"
            )

        except Exception as e:
            st.error(f"Error loading model/data: {str(e)}")

    st.markdown("---")

    # -------------------------------------------------------
    # Button 2 — Classification Report
    # -------------------------------------------------------
    st.header("Step 2 – Classification Report")

    if st.button("📊 Generate Classification Report"):
        if st.session_state.model is None:
            st.error("Please load the model first.")
        else:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

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

            st.subheader("Classification Report")
            st.dataframe(report_df.style.format("{:.4f}"))

    st.markdown("---")

    # -------------------------------------------------------
    # Button 3 — Logloss Curve
    # -------------------------------------------------------
    st.header("Step 3 – Log Loss Evaluation History")

    if st.button("📈 Show Log-Loss Plot"):
        if st.session_state.model is None:
            st.error("Please load the model first.")
        else:
            model = st.session_state.model

            try:
                results = model.evals_result()
                train_loss = results["validation_0"]["mlogloss"]
                val_loss = results["validation_1"]["mlogloss"]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(train_loss, label="Train")
                ax.plot(val_loss, label="Validation")
                ax.set_title("XGBoost Log-Loss Over Epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Log Loss")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.6)

                st.pyplot(fig)

            except Exception:
                st.info(
                    "No evaluation history found (model was probably loaded from file without eval tracking)."
                )

    st.markdown("---")

    # -------------------------------------------------------
    # Button 4 — Confusion Matrix
    # -------------------------------------------------------
    st.header("Step 4 – Confusion Matrix")

    if st.button("🧩 Show Confusion Matrix"):
        if st.session_state.model is None:
            st.error("Please load the model first.")
        else:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            st.pyplot(fig)

    st.header("Model Performance")

    # TODO by Christian: Add live prediction functionality
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Load test data
    - User can choose:
      * Random sample
      * Specific row number
      * Filter by class
    - Display ECG signal visualization
    - Show model prediction
    - Show prediction probabilities for all classes
    - Allow user to select different models to compare
    """
    )

    st.header("Best Model: XGBoost")

    # TODO by Christian: Add best model details
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Show best parameters identified using GridSearch
    - Display model performance metrics
    - Explain why XGBoost was selected as best model
    """
    )

    st.header("Model Loading")

    st.header("Oversampling Technique")

    # TODO by Christian: Add oversampling information
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Christian:**
    - Explain SMOTE as appropriate oversampling technique
    - Generation of new synthetic samples for underrepresented classes
    - No generation of duplicates
    """
    )
