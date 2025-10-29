from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def compute_metrics_for_classification(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Args:
        eval_pred: a tuple (logits, labels) of evaluation predictions

    Returns:
        Dict[str, float]: A dictionary containing:
            - "accuracy": overall accuracy.
            - "precision": weighted precision.
            - "recall": weighted recall.
            - "f1": weighted F1 score.
    """
    logits, labels = eval_pred

    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    
    logits = np.array(logits)
    labels = np.array(labels)

    # predicted class = argmax over logits
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
