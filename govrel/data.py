"""Build model inputs from text, timestamps, and post metadata."""

from math import cos, log1p, sin, tau
import random

import numpy as np
import torch

from .inputs import OPTIONAL_FIELDS, check_annotated_data, read_rows, timestamp

POST_INPUT_FIELDS = ("post_id", "normalized_text", "published_at", *OPTIONAL_FIELDS)
COMMENT_INPUT_FIELDS = ("record_id", "normalized_text", "published_at")


def read_private(path, fields, expected_ids):
    rows = {}
    for row, location in read_rows(path, fields):
        key = row[fields[0]]
        if key not in expected_ids or key in rows:
            raise ValueError(f"{location}: unknown or repeated input ID")
        if (
            not isinstance(row["normalized_text"], str)
            or not row["normalized_text"].strip()
        ):
            raise ValueError(f"{location}: empty normalized text")
        rows[key] = row
    if rows.keys() != expected_ids:
        raise ValueError(f"{path.name}: inputs do not cover the annotated IDs")
    return rows


def features(post, comment):
    post_at = timestamp(post["published_at"])
    comment_at = timestamp(comment["published_at"])
    minutes = (comment_at - post_at).total_seconds() / 60
    if minutes < 0:
        raise ValueError("Comment timestamp precedes the source post")
    hour, weekday = tau * post_at.hour / 24, tau * post_at.weekday() / 7
    numeric = [sin(hour), cos(hour), sin(weekday), cos(weekday), log1p(minutes)]
    categories = []
    for field in OPTIONAL_FIELDS:
        value = post[field]
        expected_type = bool if field in ("image", "video", "topic") else str
        if value is not None and (type(value) is not expected_type or value == ""):
            raise ValueError(f"Invalid value for optional metadata field {field}")
        categories.append(str(value) if value is not None else None)
    return numeric, categories


def load_corpus(directory, splits, inputs_dir=None):
    posts, comments = check_annotated_data(directory, splits)
    inputs_dir = inputs_dir or directory
    post_inputs = read_private(
        inputs_dir / "post_inputs.jsonl", POST_INPUT_FIELDS, posts.keys()
    )
    comment_inputs = read_private(
        inputs_dir / "comment_inputs.jsonl", COMMENT_INPUT_FIELDS, comments.keys()
    )
    corpus = {split: {} for split in splits}
    for post_id, post in sorted(posts.items()):
        if post["split"] in corpus:
            corpus[post["split"]][post_id] = {
                **post,
                **post_inputs[post_id],
                "comments": [],
            }
    for record_id, comment in sorted(comments.items()):
        if comment["split"] in corpus:
            post = corpus[comment["split"]][comment["post_id"]]
            row = {**comment, **comment_inputs[record_id]}
            row["numeric"], row["categories"] = features(post, row)
            post["comments"].append(row)
    return {split: list(posts.values()) for split, posts in corpus.items()}


def fit_metadata(train_posts):
    rows = [row for post in train_posts for row in post["comments"]]
    values = np.asarray([row["numeric"] for row in rows])
    std = values.std(axis=0)
    vocabularies = []
    for index in range(len(OPTIONAL_FIELDS)):
        observed = {row["categories"][index] for row in rows}
        vocabularies.append(sorted(observed - {None}))
    return {
        "mean": values.mean(axis=0).tolist(),
        "scale": np.where(std > 0, std, 1).tolist(),
        "vocabularies": vocabularies,
    }


def grouped_batches(posts, group_size, seed=None):
    order = list(posts)
    if seed is not None:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), group_size):
        yield order[start : start + group_size]


class BatchEncoder:
    def __init__(self, tokenizer, metadata, device):
        self.tokenizer = tokenizer
        self.device = device
        self.mean = np.asarray(metadata["mean"])
        self.scale = np.asarray(metadata["scale"])
        self.categories = [
            {value: index for index, value in enumerate(vocabulary, 1)}
            for vocabulary in metadata["vocabularies"]
        ]
        if any(
            token is None
            for token in (
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )
        ):
            raise ValueError("Expected a BERT-compatible tokenizer with CLS/SEP/PAD")
        self.token_cache = {}

    def tensor(self, values, dtype=torch.long):
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def tokens(self, key, text, limit):
        if key not in self.token_cache:
            tokens = self.tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=limit
            )
            if not tokens:
                raise ValueError(f"No wordpieces for {key}")
            self.token_cache[key] = tokens
        return self.token_cache[key]

    def pack(self, rows):
        ids, segments = zip(*rows)
        width = max(map(len, ids))
        lengths = self.tensor([len(row) for row in ids])
        return {
            "input_ids": self.tensor(
                [
                    row + [self.tokenizer.pad_token_id] * (width - len(row))
                    for row in ids
                ]
            ),
            "token_type_ids": self.tensor(
                [row + [0] * (width - len(row)) for row in segments]
            ),
            "attention_mask": torch.arange(width, device=self.device)[None, :]
            < lengths[:, None],
        }

    def posts(self, posts):
        rows, sizes = [], []
        for post in posts:
            ids = self.tokens(post["post_id"], post["normalized_text"], 254)
            rows.append(
                (
                    [self.tokenizer.cls_token_id, *ids, self.tokenizer.sep_token_id],
                    [0] * (len(ids) + 2),
                )
            )
            sizes.append(len(ids))
        batch = self.pack(rows)
        position = torch.arange(batch["input_ids"].shape[1], device=self.device)[
            None, :
        ]
        return batch, (position > 0) & (position <= self.tensor(sizes)[:, None])

    def comments(self, posts, examples):
        rows, post_sizes, comment_sizes = [], [], []
        for index, comment in examples:
            post = posts[index]
            c = self.tokens(comment["record_id"], comment["normalized_text"], 96)
            p = self.tokens(post["post_id"], post["normalized_text"], 254)[
                : 253 - len(c)
            ]
            ids = [
                self.tokenizer.cls_token_id,
                *p,
                self.tokenizer.sep_token_id,
                *c,
                self.tokenizer.sep_token_id,
            ]
            rows.append((ids, [0] * (len(p) + 2) + [1] * (len(c) + 1)))
            post_sizes.append(len(p))
            comment_sizes.append(len(c))
        batch = self.pack(rows)
        position = torch.arange(batch["input_ids"].shape[1], device=self.device)[
            None, :
        ]
        p_end = self.tensor(post_sizes)[:, None] + 1
        c_end = p_end + self.tensor(comment_sizes)[:, None]
        numeric = np.asarray([row["numeric"] for _, row in examples])
        categorical = [
            [
                vocab.get(value, 0)
                for vocab, value in zip(self.categories, row["categories"])
            ]
            for _, row in examples
        ]
        return {
            "pair": batch,
            "post_mask": (position > 0) & (position < p_end),
            "comment_mask": (position > p_end) & (position <= c_end),
            "numeric": self.tensor((numeric - self.mean) / self.scale, torch.float32),
            "categories": self.tensor(categorical),
            "post_index": self.tensor([index for index, _ in examples]),
        }
