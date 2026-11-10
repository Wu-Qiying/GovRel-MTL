"""Render JoVE R2 Figures 2–4 from a reproduction report."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(figure, folder, name):
    figure.tight_layout()
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def figure2(report, folder):
    groups = {
        (row["fusion_method"], row["mask_pct"]): row["summary"]
        for row in report["metadata_masking"]
    }
    levels = [0, 60, 100]
    if any(
        (method, pct) not in groups
        for method in ("gated", "direct_concat")
        for pct in levels
    ):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for axis, metric, sign, title in zip(
        axes,
        ("priority_macro_f1", "priority_ece"),
        (1, -1),
        ("A  Macro-F1 gain", "B  ECE reduction"),
    ):
        values = [
            sign
            * (
                groups["gated", level][metric]["mean"]
                - groups["direct_concat", level][metric]["mean"]
            )
            for level in levels
        ]
        axis.bar([str(level) for level in levels], values, color="#28798a", width=0.55)
        axis.axhline(0, color="#444444", linewidth=0.7)
        axis.set(
            xlabel="Optional metadata masked (%)", ylabel="Difference", title=title
        )
    save(fig, folder, "figure_2_metadata_masking")
    return True


def figure3(report, folder):
    models = ("B5", "B6", "B7")
    ensembles = [
        (report["models"].get(model) or {}).get("ensemble") for model in models
    ]
    if any(not entry or entry["ece"] is None for entry in ensembles):
        return False
    fig, axis = plt.subplots(figsize=(4.4, 3))
    axis.bar(
        models,
        [entry["ece"] for entry in ensembles],
        color=["#8e9aa6", "#658497", "#28798a"],
        width=0.55,
    )
    axis.set(xlabel="Model", ylabel="Top-label expected calibration error", ylim=(0, 1))
    save(fig, folder, "figure_3_ensemble_calibration")
    return True


def figure4(report, folder):
    runs = defaultdict(list)
    for row in report["training_curves"]:
        runs[row["model_id"], row["run_id"], row["seed"]].append(row)
    if not runs:
        return False
    if len(runs) != 2 or {key[0] for key in runs} != {"B6", "B7"}:
        raise ValueError(
            "Figure 4 requires one designated B6 run and one designated B7 run"
        )
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for (model, _, _), rows in sorted(runs.items()):
        rows.sort(key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in rows]
        color = "#28798a" if model == "B7" else "#737f8b"
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in rows],
            marker="o",
            color=color,
            label=model,
        )
        axes[1].plot(
            epochs,
            [row["val_priority_f1"] for row in rows],
            marker="o",
            color=color,
            label=model,
        )
        if model == "B7":
            selected = [row["epoch"] for row in rows if row["selected"]]
            if len(selected) != 1:
                raise ValueError("Figure 4 requires exactly one selected B7 epoch")
            for axis in axes:
                axis.axvline(
                    selected[0], color="#444444", linestyle="--", linewidth=0.8
                )
    for axis, title, ylabel in zip(
        axes, ("A  Training", "B  Validation"), ("Training loss", "Priority Macro-F1")
    ):
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.legend(frameon=False)
    save(fig, folder, "figure_4_training_validation")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    for number, plot in ((2, figure2), (3, figure3), (4, figure4)):
        if not plot(report, args.out):
            print(f"Figure {number} skipped: required inputs are unavailable.")


if __name__ == "__main__":
    main()
