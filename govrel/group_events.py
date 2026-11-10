"""Draft a chronological event partition from restricted, normalized post records."""

import argparse
from collections import Counter, deque
from datetime import timedelta
from hashlib import blake2b
import json
from pathlib import Path

from .inputs import check_id, read_rows, timestamp


def simhash(text):
    text = "".join(text.split())
    if len(text) < 3:
        return None
    grams = Counter(text[i : i + 3] for i in range(len(text) - 2))
    votes = [0] * 64
    for gram, count in grams.items():
        fingerprint = int.from_bytes(
            blake2b(gram.encode(), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            votes[bit] += count if fingerprint & (1 << bit) else -count
    return sum(1 << bit for bit, vote in enumerate(votes) if vote > 0)


def partition(rows):
    posts = sorted(
        (timestamp(row["published_at"]), row["post_id"], row) for row in rows
    )
    parents = list(range(len(posts)))

    def root(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def connect(a, b):
        a, b = sorted((root(a), root(b)))
        parents[b] = a

    previous_url = {}
    recent = deque()
    for index, (published_at, _, row) in enumerate(posts):
        url = row["canonical_url"]
        if url:
            connect(index, previous_url.setdefault(url, index))
        fingerprint = simhash(row["normalized_text"])
        while recent and published_at - recent[0][0] > timedelta(days=7):
            recent.popleft()
        if fingerprint is not None:
            for _, other, account, other_hash in recent:
                if (
                    account != row["account_id"]
                    and bin(fingerprint ^ other_hash).count("1") <= 3
                ):
                    connect(index, other)
            recent.append((published_at, index, row["account_id"], fingerprint))

    roots = sorted({root(index) for index in range(len(posts))})
    if len(roots) < 10:
        raise ValueError("Too few event groups for the 70/10/20 chronological draft")
    cut_train, cut_validation = 7 * len(roots) // 10, 8 * len(roots) // 10
    order = {group: rank for rank, group in enumerate(roots)}
    for index, (_, post_id, _) in enumerate(posts):
        rank = order[root(index)]
        split = (
            "train"
            if rank < cut_train
            else "validation"
            if rank < cut_validation
            else "test"
        )
        yield {"post_id": post_id, "event_id": f"E-{rank + 1:06}", "split": split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    posts = []
    seen = set()
    fields = (
        "post_id",
        "account_id",
        "published_at",
        "normalized_text",
        "canonical_url",
    )
    for row, location in read_rows(args.posts, fields):
        check_id(row["post_id"], "P-", location)
        check_id(row["account_id"], "A-", location)
        if (
            row["post_id"] in seen
            or not isinstance(row["normalized_text"], str)
            or not row["normalized_text"].strip()
        ):
            raise ValueError(f"{location}: repeated post or missing normalized text")
        if row["canonical_url"] is not None and not isinstance(
            row["canonical_url"], str
        ):
            raise ValueError(f"{location}: canonical_url must be a string or null")
        seen.add(row["post_id"])
        posts.append(row)
    mapping = list(partition(posts))
    with args.out.open("x", encoding="utf-8") as output:
        for row in mapping:
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
