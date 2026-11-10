"""Clean post and comment exports and count exclusions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from .inputs import OPTIONAL_FIELDS, read_rows, timestamp, unique

URL = re.compile(r"(?:https?://|www\.)[^\s，。！？、；：）】]+", re.IGNORECASE)
MENTION = re.compile(r"(?<!\w)@[\w.-]+", re.UNICODE)


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Expected a text string")
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    return " ".join(text.split())


def meaningful(text):
    return any(unicodedata.category(char).startswith("L") for char in text)


def clean(posts, comments, whitelist, advertisements):
    post_index = {}
    seen_posts = set()
    counts = Counter()
    for row in posts:
        counts["posts_read"] += 1
        post_id = row["post_id"]
        if not post_id:
            raise ValueError("Empty post_id")
        if post_id in seen_posts:
            counts["posts_duplicate_id"] += 1
            continue
        seen_posts.add(post_id)
        text = normalize(row["text"])
        if not meaningful(text) and text not in whitelist:
            counts["posts_no_linguistic_content"] += 1
            continue
        timestamp(row["published_at"])
        post_index[post_id] = {
            **{key: row[key] for key in ("post_id", "published_at", *OPTIONAL_FIELDS)},
            "normalized_text": text,
        }

    cleaned = []
    seen_comments = set()
    for row in comments:
        counts["comments_read"] += 1
        record_id, post_id = row["record_id"], row["post_id"]
        if not record_id:
            raise ValueError("Empty record_id")
        if record_id in seen_comments:
            counts["comments_duplicate_id"] += 1
            continue
        seen_comments.add(record_id)
        if post_id not in post_index:
            counts["comments_missing_post"] += 1
            continue
        text = normalize(row["text"])
        if not text or (not meaningful(text) and text not in whitelist):
            counts["comments_no_linguistic_content"] += 1
            continue
        if text not in whitelist and any(
            pattern.search(text) for pattern in advertisements
        ):
            counts["comments_advertisement"] += 1
            continue
        if timestamp(row["published_at"]) < timestamp(
            post_index[post_id]["published_at"]
        ):
            counts["comments_before_post"] += 1
            continue
        cleaned.append(
            {
                "record_id": record_id,
                "post_id": post_id,
                "published_at": row["published_at"],
                "normalized_text": text,
            }
        )
    counts["posts_retained"] = len(post_index)
    counts["comments_retained"] = len(cleaned)
    return post_index, cleaned, dict(counts)


def project(posts, comments, post_ids, comment_to_post):
    comments = unique(comments, "record_id")
    for key, post_id in comment_to_post.items():
        if key not in comments or comments[key]["post_id"] != post_id:
            raise ValueError(f"Selected comment has no matching cleaned record: {key}")
    return (
        [posts[key] for key in sorted(post_ids)],
        [
            {
                key: value
                for key, value in comments[record_id].items()
                if key != "post_id"
            }
            for record_id in sorted(comment_to_post)
        ],
    )


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    from .inputs import annotations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--comments", type=Path, required=True)
    parser.add_argument(
        "--rules",
        type=Path,
        help="JSON with short_text_whitelist and advertisement_patterns",
    )
    parser.add_argument(
        "--annotation-dir", type=Path, help="Project onto a frozen partition export"
    )
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text(encoding="utf-8")) if args.rules else {}
    whitelist = {normalize(text) for text in rules.get("short_text_whitelist", [])} - {
        ""
    }
    advertisements = [
        re.compile(pattern) for pattern in rules.get("advertisement_patterns", [])
    ]
    posts, comments, counts = clean(
        (
            row
            for row, _ in read_rows(
                args.posts, ("post_id", "text", "published_at", *OPTIONAL_FIELDS)
            )
        ),
        (
            row
            for row, _ in read_rows(
                args.comments, ("record_id", "post_id", "text", "published_at")
            )
        ),
        whitelist,
        advertisements,
    )
    if args.annotation_dir:
        splits = ("test",) if args.split == "test" else ("train", "validation")
        labeled_posts, labeled_comments = annotations(args.annotation_dir, splits)
        selected = {key: row["post_id"] for key, row in labeled_comments.items()}
        posts, comments = project(posts, comments, set(labeled_posts), selected)
    else:
        posts = list(posts.values())
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "post_inputs.jsonl", posts)
    write_jsonl(args.out / "comment_inputs.jsonl", comments)
    (args.out / "cleaning_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
