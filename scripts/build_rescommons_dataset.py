from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "rescommons_raw" / "full_ecom_chatbot.jsonl"
PARQUET_DIR = ROOT / "data" / "rescommons_raw"
OUT = ROOT / "data" / "rescommons_derived"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    docs = []
    eval_cases = []
    response_types = Counter()
    intents = Counter()
    capabilities = Counter()
    for row in rows:
        response_types[row.get("response_type", "")] += 1
        intents[row.get("intent", "")] += 1
        capabilities[row.get("capability", "")] += 1
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        context = _context_text(row.get("context", ""))
        if not prompt or not response:
            continue
        if row.get("_split") == "train":
            doc_text = "\n".join(part for part in [prompt, context, response] if part)
            docs.append(
                {
                    "doc_id": row["id"],
                    "source": row.get("source", ""),
                    "split": row.get("_split", ""),
                    "response_type": row.get("response_type", ""),
                    "intent_category": row.get("intent_category", ""),
                    "intent": row.get("intent", ""),
                    "capability": row.get("capability", ""),
                    "quality_score": row.get("quality_score", 0),
                    "text": doc_text,
                }
            )
        elif len(eval_cases) < 500:
            eval_cases.append(
                {
                    "id": f"rescommons-{len(eval_cases)}",
                    "query": prompt,
                    "intent": row.get("intent", ""),
                    "capability": row.get("capability", ""),
                    "intent_category": row.get("intent_category", ""),
                    "response_type": row.get("response_type", ""),
                }
            )

    _write_jsonl(OUT / "support_corpus.jsonl", docs)
    _write_jsonl(OUT / "hybrid_retrieval_eval_cases.jsonl", eval_cases)
    metadata = {
        "source": "rescommons/Full-Ecom-Chatbot-Dataset",
        "source_url": "https://huggingface.co/datasets/rescommons/Full-Ecom-Chatbot-Dataset",
        "license": "MIT",
        "raw_rows": len(rows),
        "corpus_docs": len(docs),
        "eval_cases": len(eval_cases),
        "response_type_counts": response_types.most_common(),
        "top_intents": intents.most_common(20),
        "capability_counts": capabilities.most_common(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'support_corpus.jsonl'} ({len(docs)} docs)")
    print(f"Wrote {OUT / 'hybrid_retrieval_eval_cases.jsonl'} ({len(eval_cases)} cases)")


def load_rows() -> list[dict]:
    parquet_paths = [PARQUET_DIR / "train.parquet", PARQUET_DIR / "test.parquet"]
    if all(path.exists() for path in parquet_paths):
        import pyarrow.parquet as pq

        rows = []
        for path in parquet_paths:
            table = pq.read_table(path)
            split = path.stem
            for row in table.to_pylist():
                row["_split"] = split
                rows.append(row)
        return rows
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]


def _context_text(raw: str) -> str:
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return str(raw)[:1000]
    docs = data.get("retrieved_docs", [])
    if not docs:
        return ""
    if isinstance(docs, list):
        return "\n".join(str(item) for item in docs[:3])
    return str(docs)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            compact = dict(row)
            compact["text"] = re.sub(r"\s+", " ", compact.get("text", "")).strip()
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
