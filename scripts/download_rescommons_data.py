from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rescommons_raw" / "full_ecom_chatbot.jsonl"
DATASET = "rescommons/Full-Ecom-Chatbot-Dataset"
API = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=0, help="0 downloads the full train+test set")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    split_offsets = existing_offsets()
    total = sum(split_offsets.values())
    mode = "a" if OUT.exists() else "w"
    with OUT.open(mode, encoding="utf-8") as f:
        for split in ("train", "test"):
            offset = split_offsets[split]
            while True:
                if args.max_rows and total >= args.max_rows:
                    break
                length = args.page_size
                if args.max_rows:
                    length = min(length, args.max_rows - total)
                batch = fetch_rows(split=split, offset=offset, length=length)
                rows = batch.get("rows", [])
                if not rows:
                    break
                for item in rows:
                    row = item["row"]
                    row["_split"] = split
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
                offset += len(rows)
                print(f"{split}: downloaded {offset} rows; total={total}")
                if len(rows) < length:
                    break
                time.sleep(args.sleep)
            if args.max_rows and total >= args.max_rows:
                break
    print(f"Wrote {OUT} ({total} rows)")


def fetch_rows(split: str, offset: int, length: int) -> dict:
    wait = 2.0
    for attempt in range(8):
        response = requests.get(
            API,
            params={
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=60,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        print(f"429 rate limited at {split}:{offset}; retry {attempt + 1}/8 after {wait:.1f}s")
        time.sleep(wait)
        wait = min(wait * 1.8, 30.0)
    response.raise_for_status()
    return response.json()


def existing_offsets() -> Counter[str]:
    offsets: Counter[str] = Counter({"train": 0, "test": 0})
    if not OUT.exists():
        return offsets
    with OUT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            offsets[row.get("_split", "train")] += 1
    if sum(offsets.values()):
        print(f"Resuming from offsets: train={offsets['train']}, test={offsets['test']}")
    return offsets


if __name__ == "__main__":
    main()
