import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)


# new version
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test, average="macro"):
    """Compute consistent metrics for train/test sets (incl. ROC-AUC)."""

    labels = np.unique(np.concatenate([y_train, y_test]))

    def _compute_metrics(X, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # --- Safe ROC–AUC computation ---
        auc = np.nan
        try:
            # Try probability estimates
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
            # Fallback to decision function (e.g. SVM)
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X)
            else:
                y_proba = None

            # Validate probabilities/scores
            if y_proba is not None and not np.allclose(y_proba, y_proba.astype(int)):
                if len(np.unique(y_true)) > 2:
                    auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
                else:
                    # Binary case
                    if y_proba.ndim == 2:
                        y_proba = y_proba[:, 1]
                    auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = np.nan

        return {
            "accuracy": acc,
            "precision_macro": p_m,
            "recall_macro": r_m,
            "f1_macro": f1_m,
            "precision_per_class": p_c,
            "recall_per_class": r_c,
            "f1_per_class": f1_c,
            "support_per_class": sup_c,
            "confusion_matrix": cm,
            "roc_auc": auc,
        }

    # Compute metrics separately for train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "labels": labels,
        "train": _compute_metrics(X_train, y_train, y_pred_train),
        "test": _compute_metrics(X_test, y_test, y_pred_test),
    }


def eval_model(model, X_tr, y_tr, X_te, y_te):
    """Evaluate model on training and test sets.
    
    Args:
        model: Trained model to evaluate
        X_tr: Training features
        y_tr: Training labels
        X_te: Test features  
        y_te: Test labels
        
    Returns:
        Dictionary with evaluation results for train and test sets
    """
    model.fit(X_tr, y_tr)
    yt = model.predict(X_te)

    # Choose a consistent label order (dynamic)
    labels = np.unique(np.concatenate([y_tr, y_te]))

    # Test
    acc_t = accuracy_score(y_te, yt)
    p_t_m, r_t_m, f1_t_m, _ = precision_recall_fscore_support(
        y_te, yt, average='macro', zero_division=0
    )
    p_t_c, r_t_c, f1_t_c, sup_t = precision_recall_fscore_support(
        y_te, yt, average=None, labels=labels, zero_division=0
    )
    cm_t = confusion_matrix(y_te, yt, labels=labels)

    return {
        'labels': labels,  # order for per-class arrays below
        'test': {
            'accuracy': acc_t,
            'precision_macro': p_t_m,
            'recall_macro': r_t_m,
            'f1_macro': f1_t_m,
            'precision_per_class': p_t_c,
            'recall_per_class': r_t_c,
            'f1_per_class': f1_t_c,
            'support_per_class': sup_t,
            'confusion_matrix': cm_t,
        },
    }
