from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v1rtucious_raw"
URL = "https://huggingface.co/datasets/V1rtucious/Ecom-Chatbot-Test-Set/resolve/main/data/test-00000-of-00001.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "test.parquet"
    if path.exists():
        print(f"Skip existing {path}")
        return
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
