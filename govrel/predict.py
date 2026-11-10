"""Export locked-test classifications and priority probabilities for one B7 seed."""

import argparse
from contextlib import ExitStack
import csv
import json
from math import isfinite
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import BatchEncoder, load_corpus
from .inputs import CLASS_CODES, POST_PREDICTIONS, PREDICTIONS
from .model import GovRelMTL
from .train import default_device, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--post-out", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    config, metadata = run["config"], run["metadata"]
    corpus = load_corpus(args.data_dir, ("test",), args.inputs_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.run / "tokenizer", local_files_only=True
    )
    encoder = AutoModel.from_config(
        AutoConfig.from_pretrained(args.run / "encoder", local_files_only=True)
    )
    model = GovRelMTL(encoder, [len(vocab) + 1 for vocab in metadata["vocabularies"]])
    state = torch.load(
        args.run / "model_state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.to(args.device)
    batch_encoder = BatchEncoder(tokenizer, metadata, args.device)
    with ExitStack() as stack:
        output = csv.writer(
            stack.enter_context(args.out.open("x", encoding="utf-8", newline=""))
        )
        output.writerow(PREDICTIONS)
        prefix = ("B7", args.run_id, config["seed"])
        post_callback = None
        if args.post_out:
            post_output = csv.writer(
                stack.enter_context(
                    args.post_out.open("x", encoding="utf-8", newline="")
                )
            )
            post_output.writerow(POST_PREDICTIONS)

            def post_callback(post_id, prediction):
                post_output.writerow(
                    (*prefix, post_id, CLASS_CODES["post_type"][prediction], "OK")
                )

        for row, scores in predictions(
            model,
            batch_encoder,
            corpus["test"],
            config["group_size"],
            config["microbatch"],
            post_callback,
        ):
            if not all(
                isfinite(value) for task_scores in scores for value in task_scores
            ):
                raise ValueError(f"Nonfinite output for {row['record_id']}")
            labels = [
                CLASS_CODES[task][
                    max(range(len(probabilities)), key=probabilities.__getitem__)
                ]
                for task, probabilities in zip(
                    ("sentiment", "demand", "priority"), scores
                )
            ]
            output.writerow((*prefix, row["record_id"], *labels, *scores[-1], "OK"))


if __name__ == "__main__":
    main()
