# Running GovRel-MTL

Run the commands from the repository root with Python 3.10. Use a local directory such as `private/` for study files.

## Clean posts and comments

Supply UTF-8 JSONL, with one record per line. Use the same post and comment IDs throughout.

| File | Fields |
| --- | --- |
| Posts | `post_id`, `text`, `published_at`, `image`, `video`, `topic`, `account_level`, `account_function` |
| Comments | `record_id`, `post_id`, `text`, `published_at` |

Use ISO 8601 timestamps with a timezone offset. The media and topic fields accept Boolean values or null. The account fields accept strings or null.

```sh
python -m govrel.prepare \
  --posts private/posts.jsonl --comments private/comments.jsonl \
  --rules private/cleaning_rules.json --out private/cleaned
```

The script normalizes Unicode, strips links and mentions, keeps the first occurrence of each ID, and drops comments without usable text or a matching post. It keeps short comments with linguistic content and emojis attached to that text. The optional rules file takes two arrays of strings, `short_text_whitelist` and `advertisement_patterns`. The output includes cleaned JSONL and counts for each exclusion rule.

To prepare files for training, rerun with `--annotation-dir private/development` and `--out private/cleaned_dev`. For the locked test set, use `--annotation-dir private/locked_test --split test` and `--out private/cleaned_test`. These options keep the annotated records and check each comment's post ID. Use a new output directory for each run.

## Group posts by event

Use JSONL with `post_id`, `account_id`, `published_at`, `normalized_text`, and `canonical_url`. The canonical URL can be null. Use opaque post and account IDs prefixed with `P-` and `A-`, respectively.

```sh
python -m govrel.group_events \
  --posts private/posts_for_grouping.jsonl \
  --out private/event_mapping_draft.jsonl
```

The script groups posts that share a canonical URL. It also groups posts from different accounts when they are at most seven days apart and their character-trigram SimHashes differ by at most three bits. It orders event groups by the date of their first post and splits them 70/10/20. This implementation uses 64-bit BLAKE2b hashes and rounds split boundaries down. For an existing experiment, use its saved event mapping and sample IDs.

## Train B7 and export predictions

Keep train and validation annotations in one directory and locked-test annotations in another. Both need `posts.csv` and `comments.csv`. To create the CSV headers, run the template command in the next section.

```sh
python -m pip install -r requirements-model.txt
python -m govrel.train \
  --data-dir private/development --inputs-dir private/cleaned_dev \
  --checkpoint /path/to/verified_macbert_snapshot \
  --seed 11 --out runs/b7-seed11
python -m govrel.predict \
  --data-dir private/locked_test --inputs-dir private/cleaned_test \
  --run runs/b7-seed11 --run-id b7-seed11 \
  --out runs/b7-seed11-comments.csv --post-out runs/b7-seed11-posts.csv
```

For a remote checkpoint, pass the repository ID to `--checkpoint` and the full commit SHA to `--revision`. Choose learning rates and the class-weight beta on development data. The B7 reference uses post hour and weekday in UTC+8, elapsed minutes, eight-dimensional categorical embeddings, and token-only pooling. It encodes each post separately and caps the paired input at 256 wordpieces, with up to 96 for the comment. Use the corresponding run exports for baselines and ablations.

## Recompute results and figures

Combine the annotations for all splits in `private/evaluation`. Merge the run exports into one predictions file with a single header.

```sh
python -m pip install -r requirements.txt
python -m govrel.reproduce --init-inputs private/templates
mkdir -p results
python -m govrel.reproduce --data-dir private/evaluation --out results/report.json
python -m govrel.figures --report results/report.json --out results/figures
```

| CSV | Contents |
| --- | --- |
| `posts.csv` | One row per annotated post with post, event and account IDs, split, two annotator labels, and the adjudicated label. |
| `comments.csv` | One row per annotated comment with its post ID, split, and all three labels for each task. Keep the tie key used for the locked test queue. Month is optional. |
| `predictions.csv` | One row per model, run, and test comment, with predicted labels and four priority probabilities. Used for Tables 5–6 and Figure 3. |
| `post_predictions.csv` | Optional post-type predictions, one row per model, run, and test post. |
| `ablations.csv` | Optional per-run metrics for Table 7. Changes are measured against A0, the full model. |
| `masking.csv` | Optional per-run results for direct concatenation and gated fusion. Figure 2 uses the 0%, 60%, and 100% conditions. |
| `training_curves.csv` | Optional epoch logs for the B6 and B7 runs in Figure 4. Mark the selected B7 epoch with `YES`. |
| `corpus_flow.csv` | Optional counts from the collection and sampling logs. The script computes split and class counts from the annotation files. |

Use label codes `G1`–`G9`, `S1`–`S6`, `D1`–`D8`, and `R0`–`R3` as defined in `govrel/labels.json`. If the exports use other codes, map them by class before running. Set the prediction status to `OK`, `INVALID`, or `MISSING`. Leave rejected labels and unavailable probabilities blank.

The report gives run means and ensemble scores separately. It builds each ensemble from complete posteriors for five runs, or from one run for deterministic B1. It reports unavailable probability metrics as JSON null and applies Holm correction separately to the primary and exploratory bootstrap comparisons.
