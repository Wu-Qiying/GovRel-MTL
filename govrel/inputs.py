"""Read annotation and prediction CSVs and check record IDs."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

LABELS = json.loads(
    (Path(__file__).with_name("labels.json")).read_text(encoding="utf-8")
)
SPLITS = ("train", "validation", "test")
OPTIONAL_FIELDS = ("image", "video", "topic", "account_level", "account_function")
LOCAL_TIME = timezone(timedelta(hours=8))
TASKS = ("post_type", "sentiment", "demand", "priority")
PREFIXES = ("G", "S", "D", "R")
CLASS_CODES = {
    task: tuple(
        f"{prefix}{i}"
        for i in range(
            0 if task == "priority" else 1, len(LABELS[task]) + (task != "priority")
        )
    )
    for task, prefix in zip(TASKS, PREFIXES)
}
ANNOTATIONS = ("A1", "A2", "gold")
POSTS = (
    "post_id",
    "event_id",
    "account_id",
    "split",
    *(f"post_type_{s}" for s in ANNOTATIONS),
)
COMMENTS = (
    "comment_id",
    "post_id",
    "split",
    "rank_tie_key",
    "comment_month",
    *(f"{task}_{s}" for task in TASKS[1:] for s in ANNOTATIONS),
)
PROBS = tuple(f"prob_R{i}" for i in range(4))
PREDICTIONS = (
    "model_id",
    "run_id",
    "seed",
    "comment_id",
    "sentiment_pred",
    "demand_pred",
    "priority_pred",
    *PROBS,
    "output_status",
)
POST_PREDICTIONS = (
    "model_id",
    "run_id",
    "seed",
    "post_id",
    "post_type_pred",
    "output_status",
)
ABLATIONS = (
    "variant_id",
    "run_id",
    "seed",
    "priority_macro_f1",
    "priority_qwk",
    "priority_ece",
)
MASKING = (
    "fusion_method",
    "mask_pct",
    "run_id",
    "seed",
    "priority_macro_f1",
    "priority_ece",
)
CURVES = (
    "model_id",
    "run_id",
    "seed",
    "epoch",
    "train_loss",
    "val_sentiment_f1",
    "val_demand_f1",
    "val_priority_f1",
    "selected",
)
FLOW = ("stage", "split", "count")
SCHEMAS = dict(
    posts=POSTS,
    comments=COMMENTS,
    predictions=PREDICTIONS,
    post_predictions=POST_PREDICTIONS,
    ablations=ABLATIONS,
    masking=MASKING,
    training_curves=CURVES,
    corpus_flow=FLOW,
)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Expected an ISO 8601 timestamp with timezone")
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.utcoffset() is None:
        raise ValueError("Timestamp has no timezone")
    return value.astimezone(LOCAL_TIME)


def read_csv(directory, name, required=True):
    path = directory / f"{name}.csv"
    if not path.is_file() and not required:
        return []
    fields = set(SCHEMAS[name])
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if (
            reader.fieldnames is None
            or set(reader.fieldnames) != fields
            or len(reader.fieldnames) != len(fields)
        ):
            raise ValueError(f"{path.name}: expected fields {', '.join(SCHEMAS[name])}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{path.name}:{reader.line_num}: incorrect column count"
                )
            rows.append(row)
    if required and not rows:
        raise ValueError(f"{path.name}: no data rows")
    return rows


def read_rows(path, fields):
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != set(fields):
                raise ValueError(
                    f"{path.name}:{number}: expected fields {', '.join(fields)}"
                )
            yield row, f"{path.name}:{number}"


def check_id(value, prefix, location):
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value[len(prefix) :])
    ):
        raise ValueError(f"{location}: invalid ID with prefix {prefix}")


def label(task, code, missing=False):
    if missing and not code:
        return -1
    try:
        return CLASS_CODES[task].index(code)
    except ValueError:
        raise ValueError(f"Unknown {task} label: {code!r}") from None


def unique(rows, key):
    indexed = {}
    for row in rows:
        value = row[key]
        if not value or value in indexed:
            raise ValueError(f"Empty or repeated {key}: {value!r}")
        indexed[value] = row
    return indexed


def annotations(directory, splits=SPLITS):
    posts = unique(read_csv(directory, "posts"), "post_id")
    comments = unique(read_csv(directory, "comments"), "comment_id")
    events = {}
    seen_ties = set()
    for post in posts.values():
        if (
            not all(post[field] for field in ("account_id", "event_id"))
            or post["split"] not in splits
        ):
            raise ValueError(f"Invalid metadata or split for post {post['post_id']}")
        if events.setdefault(post["event_id"], post["split"]) != post["split"]:
            raise ValueError("Event appears in more than one split")
        for suffix in ANNOTATIONS:
            label("post_type", post[f"post_type_{suffix}"])
    for comment in comments.values():
        post = posts.get(comment["post_id"])
        if post is None or post["split"] != comment["split"]:
            raise ValueError(
                f"Invalid source post or split for comment {comment['comment_id']}"
            )
        for task in TASKS[1:]:
            for suffix in ANNOTATIONS:
                label(task, comment[f"{task}_{suffix}"])
        if comment["split"] == "test":
            key = comment["rank_tie_key"]
            if not key or key in seen_ties:
                raise ValueError("Test rank_tie_key must be nonempty and unique")
            seen_ties.add(key)
        month = comment["comment_month"]
        if month and not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month):
            raise ValueError(f"Invalid comment_month: {month!r}")
    if set(comment["split"] for comment in comments.values()) != set(splits):
        raise ValueError(f"Expected comments in splits: {', '.join(splits)}")
    if {row["post_id"] for row in comments.values()} != posts.keys():
        raise ValueError("Each post must have at least one included comment")
    return posts, comments


def check_annotated_data(directory, splits):
    """Adapt the annotation exports to the reference model's integer labels."""
    posts, comments = annotations(directory, splits)
    for post in posts.values():
        post["post_type"] = label("post_type", post["post_type_gold"])
    for comment in comments.values():
        comment["record_id"] = comment["comment_id"]
        for task in TASKS[1:]:
            comment[task] = label(task, comment[f"{task}_gold"])
    return posts, comments


def write_templates(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, fields in SCHEMAS.items():
        with (directory / f"{name}.csv").open(
            "x", encoding="utf-8", newline=""
        ) as destination:
            csv.writer(destination).writerow(fields)
