from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rescommons_raw"
FILES = {
    "train": "https://huggingface.co/datasets/rescommons/Full-Ecom-Chatbot-Dataset/resolve/main/data/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/rescommons/Full-Ecom-Chatbot-Dataset/resolve/main/data/test-00000-of-00001.parquet",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, url in FILES.items():
        path = OUT / f"{split}.parquet"
        if path.exists():
            print(f"Skip existing {path}")
            continue
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
