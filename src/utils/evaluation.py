import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)


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
