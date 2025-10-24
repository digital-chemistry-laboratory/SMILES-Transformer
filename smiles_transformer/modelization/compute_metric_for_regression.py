from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics_for_regression(eval_pred):
    """
    Compute metrics for regression

    Args:
        eval_pred: evaluation predictions

    Returns:
        dict: A dictionary containing:
            - "mse": mean squared error.
            - "mae": mean absolute error.
            - "r2": r2 score.
    """
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }
