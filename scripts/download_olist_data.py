"""Download public Olist CSV mirrors into data/olist_raw/.

Original dataset page:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The URLs below are GitHub raw mirrors used only to make this demo reproducible
without requiring a Kaggle login.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "olist_raw"
BASE_URL = "https://raw.githubusercontent.com/Athospd/work-at-olist-data/master/datasets"

FILES = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = OUT / filename
        if target.exists():
            print(f"skip {filename} ({target.stat().st_size} bytes)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"download {url}")
        urlretrieve(url, target)
        print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
