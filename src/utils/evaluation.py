import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)


def eval_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    model.fit(X_tr, y_tr)
    yv = model.predict(X_va)
    yt = model.predict(X_te)

    # Choose a consistent label order (dynamic)
    labels = np.unique(np.concatenate([y_tr, y_va, y_te]))

    # Validation
    acc_v = accuracy_score(y_va, yv)
    p_v_m, r_v_m, f1_v_m, _ = precision_recall_fscore_support(
        y_va, yv, average='macro', zero_division=0
    )
    p_v_c, r_v_c, f1_v_c, sup_v = precision_recall_fscore_support(
        y_va, yv, average=None, labels=labels, zero_division=0
    )
    cm_v = confusion_matrix(y_va, yv, labels=labels)

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
        'val': {
            'accuracy': acc_v,
            'precision_macro': p_v_m,
            'recall_macro': r_v_m,
            'f1_macro': f1_v_m,
            'precision_per_class': p_v_c,
            'recall_per_class': r_v_c,
            'f1_per_class': f1_v_c,
            'support_per_class': sup_v,
            'confusion_matrix': cm_v,
        },
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
