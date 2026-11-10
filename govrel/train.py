"""Train the GovRel-MTL B7 reference model on restricted annotated inputs."""

import argparse
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .data import BatchEncoder, fit_metadata, grouped_batches, load_corpus
from .model import (
    CLASS_COUNTS,
    TASKS,
    GovRelMTL,
    comment_loss_sums,
    ordinal_probabilities,
    training_weights,
)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def comment_batches(posts, size):
    rows = [
        (index, row) for index, post in enumerate(posts) for row in post["comments"]
    ]
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def update(model, batch_encoder, posts, weights, microbatch):
    total_comments = sum(len(post["comments"]) for post in posts)
    post_logits = model.encode_posts(*batch_encoder.posts(posts))
    post_probabilities = post_logits.detach().softmax(1)
    post_labels = batch_encoder.tensor([post["post_type"] for post in posts])
    post_loss = F.cross_entropy(
        post_logits, post_labels, weight=weights[0], reduction="sum"
    ) / len(posts)
    (torch.exp(-model.log_variance[0]) * post_loss).backward()
    for batch in comment_batches(posts, microbatch):
        logits = model.forward_comments(
            post_probabilities, **batch_encoder.comments(posts, batch)
        )
        targets = [
            batch_encoder.tensor([row[task] for _, row in batch]) for task in TASKS[1:]
        ]
        losses = comment_loss_sums(logits, targets, weights[1:]) / total_comments
        (torch.exp(-model.log_variance[1:]) * losses).sum().backward()
    model.log_variance.sum().backward()


@torch.inference_mode()
def predictions(
    model, batch_encoder, posts, group_size, microbatch, post_callback=None
):
    model.eval()
    for group in grouped_batches(posts, group_size):
        post_probabilities = model.encode_posts(*batch_encoder.posts(group)).softmax(1)
        if post_callback is not None:
            for post, prediction in zip(group, post_probabilities.argmax(1).tolist()):
                post_callback(post["post_id"], prediction)
        for batch in comment_batches(group, microbatch):
            sentiment, demand, priority = model.forward_comments(
                post_probabilities, **batch_encoder.comments(group, batch)
            )
            outputs = (
                sentiment.softmax(1),
                demand.softmax(1),
                ordinal_probabilities(priority),
            )
            for (_, row), scores in zip(
                batch, zip(*(output.cpu().tolist() for output in outputs))
            ):
                yield row, scores


def validation_scores(model, batch_encoder, posts, args):
    counts = [np.zeros((size, size), dtype=int) for size in CLASS_COUNTS[1:]]
    for row, scores in predictions(
        model, batch_encoder, posts, args.group_size, args.microbatch
    ):
        for task, probabilities, matrix in zip(TASKS[1:], scores, counts):
            matrix[row[task], int(np.argmax(probabilities))] += 1
    metrics = {}
    for task, matrix in zip(TASKS[1:], counts):
        support = matrix.sum(axis=0) + matrix.sum(axis=1)
        scores = np.divide(
            2 * matrix.diagonal(), support, out=np.zeros(len(matrix)), where=support > 0
        )
        metrics[task] = float(scores.mean())
    return metrics


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    local = Path(args.checkpoint).is_dir()
    if not local and (
        not args.revision or not re.fullmatch("[0-9a-f]{40}", args.revision)
    ):
        raise ValueError(
            "A remote checkpoint requires --revision with its full commit SHA"
        )
    checkpoint_args = {"local_files_only": local}
    if not local:
        checkpoint_args["revision"] = args.revision
    corpus = load_corpus(args.data_dir, ("train", "validation"), args.inputs_dir)
    train_posts = corpus["train"]
    metadata = fit_metadata(train_posts)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **checkpoint_args)
    encoder = AutoModel.from_pretrained(args.checkpoint, **checkpoint_args)
    if (
        encoder.config.model_type != "bert"
        or encoder.config.max_position_embeddings < 256
    ):
        raise ValueError(
            "Use a BERT-compatible encoder supporting at least 256 positions"
        )
    batch_encoder = BatchEncoder(tokenizer, metadata, args.device)
    model = GovRelMTL(
        encoder, [len(vocab) + 1 for vocab in metadata["vocabularies"]]
    ).to(args.device)
    weights = training_weights(train_posts, args.beta, args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=0.01,
    )
    steps = ceil(len(train_posts) / args.group_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(args.out / "tokenizer")
    encoder.config.save_pretrained(args.out / "encoder")
    best, stale = -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for group in grouped_batches(
            train_posts, args.group_size, seed=args.seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            update(model, batch_encoder, group, weights, args.microbatch)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
        metrics = validation_scores(model, batch_encoder, corpus["validation"], args)
        score = sum(metrics.values()) / len(metrics)
        history.append({"epoch": epoch, "macro_f1": metrics, "selection_score": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            torch.save(model.state_dict(), args.out / "model_state.pt")
        else:
            stale += 1
            if stale >= 2:
                break

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = {
        "model": "B7",
        "implementation": "manuscript-reference",
        "config": config,
        "environment": {
            package: version(package) for package in ("torch", "transformers", "numpy")
        },
        "best_epoch": best_epoch,
        "metadata": metadata,
        "validation_history": history,
    }
    (args.out / "run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="hfl/chinese-macbert-base")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, choices=(11, 22, 33, 44, 55), required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--epochs", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, choices=range(1, 17), default=16)
    parser.add_argument(
        "--encoder-lr", type=float, choices=(1e-5, 2e-5, 3e-5), default=2e-5
    )
    parser.add_argument("--head-lr", type=float, choices=(5e-5, 1e-4), default=1e-4)
    parser.add_argument(
        "--beta", type=float, choices=(0.99, 0.999, 0.9999), default=0.999
    )
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be positive")
    train(args)


if __name__ == "__main__":
    main()
