"""Compute corpus counts, annotation agreement, and model results."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .bootstrap import bootstrap
from .inputs import (
    CLASS_CODES,
    PROBS,
    SPLITS,
    TASKS,
    annotations,
    label,
    read_csv,
    write_templates,
)
from .metrics import agreement, classification, priority, summarize

MODELS = tuple(f"B{i}" for i in range(1, 11))
PRIORITY_METRICS = (
    "macro_f1",
    "qwk",
    "mae",
    "high_precision",
    "high_recall",
    "average_precision",
    "ece",
    "brier",
    "recall_at_50",
    "accuracy",
)
STATUSES = {"OK", "INVALID", "MISSING"}


def run_groups(rows, id_field):
    groups = defaultdict(dict)
    seed_runs = {}
    for row in rows:
        model, run_id, seed = row["model_id"], row["run_id"], int(row["seed"])
        if (
            model not in MODELS
            or not run_id
            or seed < 0
            or row["output_status"] not in STATUSES
        ):
            raise ValueError("Invalid model, run identifier, seed or output status")
        key = (model, seed, run_id)
        if seed_runs.setdefault((model, seed), run_id) != run_id:
            raise ValueError(f"Multiple runs for {model} seed {seed}")
        if row[id_field] in groups[key]:
            raise ValueError(f"Repeated {id_field} in {key}")
        groups[key][row[id_field]] = row
    return groups


def checked_prediction(row, task):
    return label(task, row[f"{task}_pred"], missing=row["output_status"] != "OK")


def posterior(row, pred):
    values = [row[key] for key in PROBS]
    if not any(values):
        return [float("nan")] * 4
    p = np.asarray(values, dtype=float)
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or abs(p.sum() - 1) > 2e-6
    ):
        raise ValueError("Invalid priority probability vector")
    if pred >= 0 and pred != int(p.argmax()):
        raise ValueError("Priority label does not match posterior argmax")
    return p


def corpus_report(posts, comments):
    counts = {}
    for split in SPLITS:
        p = [row for row in posts.values() if row["split"] == split]
        c = [row for row in comments.values() if row["split"] == split]
        counts[split] = {
            "posts": len(p),
            "comments": len(c),
            "events": len({row["event_id"] for row in p}),
            "class_counts": {
                task: dict.fromkeys(CLASS_CODES[task], 0)
                | dict(Counter(row[f"{task}_gold"] for row in c))
                for task in TASKS[1:]
            },
        }
    reliability = {}
    for task in TASKS:
        records = posts.values() if task == "post_type" else comments.values()
        first, second = (
            np.asarray([label(task, row[f"{task}_{suffix}"]) for row in records])
            for suffix in ("A1", "A2")
        )
        reliability[task] = agreement(
            first, second, len(CLASS_CODES[task]), task == "priority"
        )
    return counts, reliability


def prediction_report(rows, test, posts):
    ids = sorted(test)
    id_set = set(ids)
    gold = {
        task: np.asarray([label(task, test[key][f"{task}_gold"]) for key in ids])
        for task in TASKS[1:]
    }
    tie_keys = np.asarray([test[key]["rank_tie_key"] for key in ids])
    test_posts = [posts[test[key]["post_id"]] for key in ids]
    groups = run_groups(rows, "comment_id")
    by_model = defaultdict(list)
    for (model, seed, run_id), batch in sorted(groups.items()):
        if batch.keys() != id_set:
            raise ValueError(
                f"{model}/{run_id}: prediction coverage does not match the locked test"
            )
        aligned = [batch[key] for key in ids]
        pred = np.asarray([checked_prediction(row, "priority") for row in aligned])
        probabilities = np.asarray(
            [posterior(row, value) for row, value in zip(aligned, pred)]
        )
        scored = {"run_id": run_id, "seed": seed}
        for task in ("sentiment", "demand"):
            task_pred = np.asarray([checked_prediction(row, task) for row in aligned])
            scored[task] = classification(gold[task], task_pred, len(CLASS_CODES[task]))
        scored["priority"] = priority(gold["priority"], pred, probabilities, tie_keys)
        by_model[model].append((scored, pred, probabilities))

    reports, ensembles = {}, {}
    for model, entries in by_model.items():
        runs, predictions, posteriors = zip(*entries)
        summary = {
            task: summarize([run[task] for run in runs], ("macro_f1",))
            for task in ("sentiment", "demand")
        }
        summary["priority"] = summarize(
            [run["priority"] for run in runs], PRIORITY_METRICS
        )
        ensemble = None
        # B1 can have a single run because it is deterministic.
        if (len(runs) == 5 or (model == "B1" and len(runs) == 1)) and np.isfinite(
            posteriors
        ).all():
            mean = np.mean(posteriors, axis=0)
            ensembles[model] = mean.argmax(axis=1)
            ensemble = priority(gold["priority"], ensembles[model], mean, tie_keys)
        reports[model] = {"runs": runs, "summary": summary, "ensemble": ensemble}

    bootstrap_reports = {}
    exclusions, subgroup = {}, {}
    if "B7" in ensembles:
        b7_runs = [entry[1] for entry in by_model["B7"]]
        for cluster in ("post_id", "event_id"):
            bootstrap_reports[cluster] = bootstrap(
                gold["priority"],
                ensembles,
                b7_runs,
                [post[cluster] for post in test_posts],
                exploratory=cluster == "event_id",
            )
        accounts = np.asarray([post["account_id"] for post in test_posts])
        for account in np.unique(accounts):
            keep = accounts != account
            if keep.any():
                scores = {
                    model: classification(gold["priority"][keep], pred[keep], 4)[
                        "macro_f1"
                    ]
                    for model, pred in ensembles.items()
                    if model in ("B5", "B6", "B7", "B10")
                }
                exclusions[account] = {
                    "n": int(keep.sum()),
                    "macro_f1": scores,
                    "deltas": {
                        m: scores["B7"] - score
                        for m, score in scores.items()
                        if m != "B7"
                    },
                }
    if "B7" in by_model:
        group_values = {
            "account": [p["account_id"] for p in test_posts],
            "post_type": [p["post_type_gold"] for p in test_posts],
            "month": [test[key]["comment_month"] for key in ids],
        }
        for dimension, values in group_values.items():
            values = np.asarray(values)
            subgroup[dimension] = {}
            for value in sorted(set(values) - {""}):
                keep = values == value
                group_gold = gold["priority"][keep]
                support = int((group_gold == 3).sum())
                scores = []
                for _, pred, _ in by_model["B7"]:
                    group_pred = pred[keep]
                    scores.append(
                        {
                            "macro_f1": classification(group_gold, group_pred, 4)[
                                "macro_f1"
                            ],
                            "high_recall": float(
                                (group_pred[group_gold == 3] == 3).mean()
                            )
                            if support
                            else None,
                        }
                    )
                subgroup[dimension][value] = {
                    "n": int(keep.sum()),
                    "r3_support": support,
                    "mean_run": summarize(scores, ("macro_f1", "high_recall")),
                }
    return {
        "models": reports,
        "models_not_supplied": sorted(set(MODELS) - set(reports)),
        "cluster_bootstrap": bootstrap_reports,
        "account_exclusions": exclusions,
        "b7_within_test_subgroups": subgroup,
    }


def post_prediction_report(rows, posts):
    test = sorted(key for key, row in posts.items() if row["split"] == "test")
    gold = np.asarray(
        [label("post_type", posts[key]["post_type_gold"]) for key in test]
    )
    results = defaultdict(list)
    for (model, seed, run_id), batch in sorted(run_groups(rows, "post_id").items()):
        if batch.keys() != set(test):
            raise ValueError(
                f"{model}/{run_id}: auxiliary prediction coverage does not match test posts"
            )
        pred = np.asarray([checked_prediction(batch[key], "post_type") for key in test])
        results[model].append(
            {
                "seed": seed,
                "run_id": run_id,
                **classification(gold, pred, len(CLASS_CODES["post_type"])),
            }
        )
    return {
        model: {"runs": runs, "summary": summarize(runs, ("macro_f1", "accuracy"))}
        for model, runs in results.items()
    }


def finite(value, lower, upper=float("inf")):
    number = float(value)
    if not np.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"Metric outside expected range: {value!r}")
    return number


def log_summary(rows, grouping, fields):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in grouping)
        identity = (*key, int(row["seed"]))
        if identity in seen or not row["run_id"]:
            raise ValueError(f"Duplicate or missing run key: {identity}")
        seen.add(identity)
        groups[key].append(
            {
                field: finite(row[field], -1 if field == "priority_qwk" else 0, 1)
                for field in fields
            }
        )
    return [
        {**dict(zip(grouping, key)), "summary": summarize(values, fields)}
        for key, values in sorted(groups.items())
    ]


def training_curves(rows):
    seen = set()
    results = []
    tasks = ("val_sentiment_f1", "val_demand_f1", "val_priority_f1")
    for row in rows:
        model, run_id, seed, epoch = (
            row["model_id"],
            row["run_id"],
            int(row["seed"]),
            int(row["epoch"]),
        )
        key = (model, run_id, seed, epoch)
        if (
            key in seen
            or model not in ("B6", "B7")
            or epoch < 1
            or row["selected"] not in ("YES", "NO")
        ):
            raise ValueError(f"Invalid or repeated curve point: {key}")
        seen.add(key)
        values = {task: finite(row[task], 0, 1) for task in tasks}
        results.append(
            {
                "model_id": model,
                "run_id": run_id,
                "seed": seed,
                "epoch": epoch,
                "train_loss": finite(row["train_loss"], -float("inf")),
                **values,
                "validation_selection_score": sum(values.values()) / 3,
                "selected": row["selected"] == "YES",
            }
        )
    return sorted(
        results,
        key=lambda row: (row["model_id"], row["run_id"], row["seed"], row["epoch"]),
    )


def reproduce(directory):
    posts, comments = annotations(directory)
    test = {key: row for key, row in comments.items() if row["split"] == "test"}
    counts, reliability = corpus_report(posts, comments)
    predictions = prediction_report(read_csv(directory, "predictions"), test, posts)
    ablations = log_summary(
        read_csv(directory, "ablations", required=False),
        ("variant_id",),
        ("priority_macro_f1", "priority_qwk", "priority_ece"),
    )
    full = next(
        (
            row["summary"]["priority_macro_f1"]["mean"]
            for row in ablations
            if row["variant_id"] == "A0"
        ),
        None,
    )
    for row in ablations:
        row["delta_macro_f1_vs_full"] = (
            row["summary"]["priority_macro_f1"]["mean"] - full
            if full is not None
            else None
        )
    masking = log_summary(
        read_csv(directory, "masking", required=False),
        ("fusion_method", "mask_pct"),
        ("priority_macro_f1", "priority_ece"),
    )
    for row in masking:
        row["mask_pct"] = int(row["mask_pct"])
        if row["fusion_method"] not in ("gated", "direct_concat") or row[
            "mask_pct"
        ] not in (0, 20, 40, 60, 100):
            raise ValueError("Unknown metadata masking condition")
    flow = [
        {**row, "count": int(row["count"])}
        for row in read_csv(directory, "corpus_flow", required=False)
    ]
    if any(row["count"] < 0 for row in flow):
        raise ValueError("Corpus flow counts must be nonnegative")
    return {
        "corpus": counts,
        "annotation_reliability": reliability,
        "corpus_flow": flow,
        **predictions,
        "auxiliary_post_task": post_prediction_report(
            read_csv(directory, "post_predictions", required=False), posts
        ),
        "ablations": ablations,
        "metadata_masking": masking,
        "training_curves": training_curves(
            read_csv(directory, "training_curves", required=False)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--data-dir", type=Path)
    modes.add_argument("--init-inputs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.init_inputs:
        write_templates(args.init_inputs)
        return
    if args.out is None:
        parser.error("--out is required with --data-dir")
    report = reproduce(args.data_dir)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
