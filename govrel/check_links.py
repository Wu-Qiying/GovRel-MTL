"""Check the archived pilot index offline."""

from pathlib import Path
import re

SOURCE_IDS = ("5000609535", "5342220662", "2418724427", "2539961154", "1523766213", "2605594314")


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    lists = sorted(data.glob("*.txt"))
    if len(lists) != len(SOURCE_IDS):
        raise ValueError("Expected six source-link lists")
    seen = set()
    for index, (path, uid) in enumerate(zip(lists, SOURCE_IDS), 1):
        rows = path.read_text(encoding="utf-8-sig").splitlines()
        if not path.name.startswith(f"{index:02}_") or len(rows) != 60:
            raise ValueError(f"{path.name}: unexpected list number or count")
        for row in rows:
            if not re.fullmatch(rf"https://weibo\.com/{uid}/[A-Za-z0-9]+", row) or row in seen:
                raise ValueError(f"{path.name}: invalid or duplicate URL")
            seen.add(row)
    print(f"{len(seen)} unique URLs across {len(lists)} accounts (offline check).")


if __name__ == "__main__":
    main()
