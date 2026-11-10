"""Metrics for nominal tasks and the four-level priority queue."""

import numpy as np


def confusion(gold, predicted, classes):
    # The extra column retains rejected predictions as errors in the denominator.
    columns = np.where(predicted < 0, classes, predicted)
    return np.bincount(
        gold * (classes + 1) + columns, minlength=classes * (classes + 1)
    ).reshape(classes, classes + 1)


def classification(gold, predicted, classes):
    matrix = confusion(gold, predicted, classes)
    support = matrix.sum(axis=1)
    predicted_support = matrix[:, :classes].sum(axis=0)
    correct = matrix[:, :classes].diagonal()
    denom = support + predicted_support
    f1 = np.divide(2 * correct, denom, out=np.zeros(classes), where=denom > 0)
    return {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1 @ support / len(gold)),
        "accuracy": float(correct.sum() / len(gold)),
        "per_class_f1": f1.tolist(),
        "per_class_precision": np.divide(
            correct,
            predicted_support,
            out=np.zeros(classes),
            where=predicted_support > 0,
        ).tolist(),
        "per_class_recall": np.divide(
            correct, support, out=np.zeros(classes), where=support > 0
        ).tolist(),
        "confusion": matrix.tolist(),
        "n": len(gold),
        "rejected": int(matrix[:, -1].sum()),
    }


def kappa(matrix, quadratic=False):
    n = matrix.sum()
    if quadratic:
        indices = np.arange(matrix.shape[0])
        weights = (indices[:, None] - indices[None, :]) ** 2
    else:
        weights = 1 - np.eye(matrix.shape[0])
    expected = (weights * np.outer(matrix.sum(1), matrix.sum(0))).sum() / n
    return float(1 - (weights * matrix).sum() / expected) if expected > 0 else None


def agreement(first, second, classes, ordinal=False):
    matrix = confusion(first, second, classes)[:, :classes]
    result = {
        "n": len(first),
        "exact_agreement": float(np.trace(matrix) / len(first)),
        "cohen_kappa": kappa(matrix),
        "confusion": matrix.tolist(),
    }
    if ordinal:
        result["quadratic_weighted_kappa"] = kappa(matrix, quadratic=True)
    return result


def calibration(gold, predicted, probabilities):
    confidence = probabilities.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    rows, ece = [], 0.0
    for index in range(10):
        keep = bins == index
        n = int(keep.sum())
        observed = float(np.mean(gold[keep] == predicted[keep])) if n else None
        expected = float(confidence[keep].mean()) if n else None
        contribution = n / len(gold) * abs(observed - expected) if n else 0.0
        ece += contribution
        rows.append(
            {
                "bin": index + 1,
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "n": n,
                "accuracy": observed,
                "mean_confidence": expected,
                "ece_contribution": contribution,
            }
        )
    return ece, rows


def ranked_metrics(gold, score, tie_keys):
    positives = int((gold == 3).sum())
    if positives == 0:
        return None, None
    ranked = np.lexsort((tie_keys, -score))
    hit = gold[ranked] == 3
    # AP advances only at distinct score thresholds.
    boundaries = np.flatnonzero(np.r_[score[ranked][1:] != score[ranked][:-1], True])
    cumulative = hit.cumsum()[boundaries]
    increments = np.diff(np.r_[0, cumulative])
    return (
        float((increments * cumulative / (boundaries + 1)).sum() / positives),
        float(hit[:50].sum() / positives),
    )


def priority(gold, predicted, probabilities, tie_keys):
    result = classification(gold, predicted, 4)
    matrix = np.asarray(result["confusion"])[:, :4]
    support = int((gold == 3).sum())
    complete = np.isfinite(probabilities).all()
    precision = float(matrix[3, 3] / matrix[:, 3].sum()) if matrix[:, 3].sum() else 0.0
    result.update(
        qwk=kappa(matrix, quadratic=True) if not result["rejected"] else None,
        mae=float(np.abs(gold - predicted).mean()) if not result["rejected"] else None,
        high_precision=precision,
        high_recall=float(matrix[3, 3] / support) if support else None,
        high_support=support,
        high_true_positive=int(matrix[3, 3]),
        average_precision=None,
        recall_at_50=None,
        ece=None,
        brier=None,
    )
    if complete:
        ap, recall50 = ranked_metrics(gold, probabilities[:, 3], tie_keys)
        result.update(
            average_precision=ap,
            recall_at_50=recall50,
            brier=float(np.mean((probabilities - np.eye(4)[gold]) ** 2)),
        )
        if not result["rejected"]:
            result["ece"], result["calibration_bins"] = calibration(
                gold, predicted, probabilities
            )
    return result


def summarize(rows, fields):
    result = {}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_runs": len(values),
        }
    return result


def macro_f1_from_cm(matrix):
    support = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    return np.divide(
        2 * diagonal,
        support,
        out=np.zeros_like(diagonal, dtype=float),
        where=support > 0,
    ).mean(axis=-1)
