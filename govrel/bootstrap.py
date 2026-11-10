"""Paired post/event bootstrap for ensemble Macro-F1 and event recall."""

import numpy as np

from .metrics import macro_f1_from_cm


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None


def holm(contrasts):
    previous = 0.0
    for index, row in enumerate(
        sorted(contrasts.values(), key=lambda row: row["raw_p"])
    ):
        previous = max(previous, min(1.0, row["raw_p"] * (len(contrasts) - index)))
        row["holm_p"] = previous


def bootstrap(
    gold, ensembles, b7_runs, clusters, exploratory=False, draws=10000, seed=20261103
):
    names, index = np.unique(clusters, return_inverse=True)
    count = len(names)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(count, np.full(count, 1 / count), size=draws)

    def cluster_confusion(predicted):
        counts = np.zeros((count, 16), dtype=np.int64)
        np.add.at(counts, (index, gold * 4 + predicted), 1)
        return (weights @ counts).reshape(-1, 4, 4)

    family_models = {"primary": ["B5", "B6"]}
    if exploratory:
        family_models["exploratory"] = ["B8", "B9", "B10"]
    eligible = [
        model
        for model in {"B7", *(m for group in family_models.values() for m in group)}
        if model in ensembles
    ]
    matrices = {model: cluster_confusion(ensembles[model]) for model in eligible}
    scores = {model: macro_f1_from_cm(matrix) for model, matrix in matrices.items()}
    observed = {
        model: float(
            macro_f1_from_cm(
                np.bincount(gold * 4 + ensembles[model], minlength=16).reshape(4, 4)
            )
        )
        for model in eligible
    }
    output = {"n_clusters": count, "replicates": draws, "seed": seed}
    for family, comparators in family_models.items():
        if "B7" not in scores or any(model not in scores for model in comparators):
            output[family] = None
            continue
        tests = {}
        for model in comparators:
            delta = scores["B7"] - scores[model]
            tail = min(int((delta <= 0).sum()), int((delta >= 0).sum()))
            tests[f"B7-{model}"] = {
                "delta": observed["B7"] - observed[model],
                "ci95": interval(delta),
                "raw_p": min(1.0, 2 * (tail + 1) / (draws + 1)),
            }
        holm(tests)
        output[family] = tests
    if exploratory and "B7" in matrices and all((pred >= 0).all() for pred in b7_runs):
        den = matrices["B7"][:, 3, :].sum(axis=1)
        keep = den > 0
        positive = gold == 3
        run_hits = np.zeros(count, dtype=float)
        for pred in b7_runs:
            np.add.at(run_hits, index, positive & (pred == 3))
        run_hits /= len(b7_runs)
        output["b7_high_priority_recall"] = {
            "ensemble_ci95": interval(matrices["B7"][keep, 3, 3] / den[keep]),
            "mean_run_ci95": interval((weights @ run_hits)[keep] / den[keep]),
            "replicates_without_r3": int((~keep).sum()),
        }
    return output
