"""Deterministic release gate for the customer-facing policy knowledge base.

Run this after changing ``data/knowledge_base/*.md``.  It intentionally does
not call an LLM: a policy-pack release should first prove that its evidence can
still be retrieved, cited, parsed, and safely placed into an Agent context.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.olist.knowledge import MarkdownKnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "knowledge_base" / "policy_eval_cases.jsonl"
DEFAULT_KB_DIR = ROOT / "data" / "knowledge_base"
DEFAULT_BASELINE = ROOT / "evaluation" / "baselines" / "rag_release_baseline.json"

# These are untrusted-document indicators, not a claim of complete prompt-injection
# detection. A hit must be reviewed before the policy pack is released.
UNTRUSTED_INSTRUCTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)",
    r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt",
    r"(?:system|developer)\s+prompt",
    r"act\s+as\s+(?:the\s+)?(?:system|developer)",
    r"\bjailbreak\b",
)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_policy_knowledge(
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
    cases_path: Path = DEFAULT_CASES,
    k: int = 3,
) -> dict[str, Any]:
    """Measure retrieval and citation fidelity against frozen policy questions."""
    cases = load_jsonl(cases_path)
    kb = MarkdownKnowledgeBase(kb_dir=kb_dir)
    top1 = 0
    recall_at_k = 0
    reciprocal_rank = 0.0
    source_type_at_1 = 0
    evidence_integrity_at_k = 0
    failures: list[dict[str, Any]] = []

    for case in cases:
        hits = kb.search(case["query"], k=k)
        ranked_titles = [hit.section_title for hit in hits]
        expected_section = case["expected_section"]
        expected_source_type = case.get("expected_source_type", "")
        found_rank = next(
            (rank for rank, title in enumerate(ranked_titles, start=1) if title == expected_section),
            None,
        )
        top1 += int(found_rank == 1)
        recall_at_k += int(found_rank is not None)
        reciprocal_rank += 1 / found_rank if found_rank is not None else 0.0
        source_type_at_1 += int(
            bool(hits)
            and bool(expected_source_type)
            and hits[0].source_type == expected_source_type
        )
        valid_evidence = all(
            hit.source and hit.source_type and hit.section_title and hit.text.strip() for hit in hits
        )
        evidence_integrity_at_k += int(bool(hits) and valid_evidence)
        source_type_mismatch = expected_source_type and (
            not hits or hits[0].source_type != expected_source_type
        )
        if found_rank != 1 or source_type_mismatch:
            failures.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_section": expected_section,
                    "expected_source_type": expected_source_type,
                    "actual_titles": ranked_titles,
                    "actual_source_types": [hit.source_type for hit in hits],
                }
            )

    total = len(cases)
    return {
        "case_count": total,
        "metrics": {
            "top1": _rate(top1, total),
            "recall_at_3": _rate(recall_at_k, total),
            "mrr_at_3": round(reciprocal_rank / total, 6) if total else 0.0,
            "source_type_at_1": _rate(source_type_at_1, total),
            "evidence_integrity_at_3": _rate(evidence_integrity_at_k, total),
        },
        "failures": failures,
    }


def inspect_knowledge_pack(kb_dir: Path = DEFAULT_KB_DIR) -> dict[str, Any]:
    """Validate document structure and surface suspicious instruction-like text."""
    kb = MarkdownKnowledgeBase(kb_dir=kb_dir)
    sections = kb._sections()  # The same parser used by the runtime retriever.
    documents = sorted(kb_dir.glob("*.md"))
    title_counts: dict[str, int] = {}
    empty_sections: list[dict[str, str]] = []
    suspicious_sections: list[dict[str, str]] = []
    for section in sections:
        title = section["title"]
        title_counts[title] = title_counts.get(title, 0) + 1
        if not section["text"].strip():
            empty_sections.append({"source": section["source"], "title": title})
        for pattern in UNTRUSTED_INSTRUCTION_PATTERNS:
            if re.search(pattern, section["text"], flags=re.IGNORECASE):
                suspicious_sections.append(
                    {"source": section["source"], "title": title, "pattern": pattern}
                )
    duplicate_titles = sorted(title for title, count in title_counts.items() if count > 1)
    source_types = sorted({section["source_type"] for section in sections})
    return {
        "document_count": len(documents),
        "section_count": len(sections),
        "source_types": source_types,
        "duplicate_titles": duplicate_titles,
        "empty_sections": empty_sections,
        "suspicious_instruction_hits": suspicious_sections,
    }


def evaluate_support_retrieval() -> dict[str, Any]:
    """Optional ResCommons component regression for retrieval-code changes."""
    from app.retrieval.hybrid import HybridSupportRetriever
    from evaluation.hybrid_retrieval_eval import CASES

    cases = load_jsonl(CASES)[:100]
    retriever = HybridSupportRetriever()
    results: dict[str, Any] = {}
    for name, method in (("bm25", retriever.bm25_search), ("hybrid", retriever.hybrid_search)):
        intent_at_1 = 0
        intent_at_5 = 0
        reciprocal_rank = 0.0
        for case in cases:
            intents = [hit.doc.get("intent") for hit in method(case["query"], k=5)]
            intent_at_1 += int(bool(intents) and intents[0] == case["intent"])
            intent_at_5 += int(case["intent"] in intents)
            reciprocal_rank += _reciprocal_rank(intents, case["intent"])
        results[name] = {
            "intent_at_1": _rate(intent_at_1, len(cases)),
            "intent_at_5": _rate(intent_at_5, len(cases)),
            "intent_mrr_at_5": round(reciprocal_rank / len(cases), 6) if cases else 0.0,
        }
    return {"case_count": len(cases), "strategies": results}


def evaluate_release(
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
    cases_path: Path = DEFAULT_CASES,
    include_support_corpus: bool = False,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "knowledge_base": str(kb_dir),
        "policy_retrieval": evaluate_policy_knowledge(kb_dir=kb_dir, cases_path=cases_path),
        "knowledge_pack": inspect_knowledge_pack(kb_dir),
    }
    if include_support_corpus:
        report["support_retrieval"] = evaluate_support_retrieval()
    return report


def gate_violations(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Compare a report with versioned release thresholds, returning every violation."""
    violations: list[str] = []
    policy = report["policy_retrieval"]
    metrics = policy["metrics"]
    policy_thresholds = baseline["policy_retrieval"]
    for metric in (
        "top1",
        "recall_at_3",
        "mrr_at_3",
        "source_type_at_1",
        "evidence_integrity_at_3",
    ):
        threshold = float(policy_thresholds[f"{metric}_min"])
        value = float(metrics[metric])
        if value < threshold:
            violations.append(f"policy_retrieval.{metric}={value:.4f} below {threshold:.4f}")

    pack = report["knowledge_pack"]
    pack_thresholds = baseline["knowledge_pack"]
    if int(pack["document_count"]) < int(pack_thresholds["document_count_min"]):
        violations.append("knowledge_pack.document_count below baseline")
    if int(pack["section_count"]) < int(pack_thresholds["section_count_min"]):
        violations.append("knowledge_pack.section_count below baseline")
    required_source_types = set(pack_thresholds["required_source_types"])
    missing_source_types = required_source_types - set(pack["source_types"])
    if missing_source_types:
        violations.append(f"knowledge_pack missing source types: {sorted(missing_source_types)}")
    if pack["duplicate_titles"]:
        violations.append(f"knowledge_pack duplicate titles: {pack['duplicate_titles']}")
    if pack["empty_sections"]:
        violations.append(f"knowledge_pack empty sections: {pack['empty_sections']}")
    if pack["suspicious_instruction_hits"]:
        violations.append("knowledge_pack contains untrusted instruction-like text")

    if "support_retrieval" in report:
        support_thresholds = baseline["support_retrieval"]
        hybrid = report["support_retrieval"]["strategies"]["hybrid"]
        for metric in ("intent_at_1", "intent_at_5", "intent_mrr_at_5"):
            threshold = float(support_thresholds[f"{metric}_min"])
            value = float(hybrid[metric])
            if value < threshold:
                violations.append(f"support_retrieval.hybrid.{metric}={value:.4f} below {threshold:.4f}")
    return violations


def render_markdown(report: dict[str, Any], violations: Iterable[str]) -> str:
    policy = report["policy_retrieval"]
    metrics = policy["metrics"]
    pack = report["knowledge_pack"]
    issues = list(violations)
    lines = [
        "# RAG Release Gate",
        "",
        f"Result: {'PASS' if not issues else 'FAIL'}",
        "",
        "## Policy Evidence Retrieval",
        "",
        "| metric | value | meaning |",
        "|---|---:|---|",
        f"| Top1 | {metrics['top1']:.2%} | expected policy section is ranked first |",
        f"| Recall@3 | {metrics['recall_at_3']:.2%} | "
        "expected section occurs in the first three evidence items |",
        f"| MRR@3 | {metrics['mrr_at_3']:.2%} | "
        "rank-sensitive evidence quality within the first three items |",
        f"| Source type@1 | {metrics['source_type_at_1']:.2%} | "
        "first item has the expected policy / FAQ / merchant-rule type |",
        f"| Evidence integrity@3 | {metrics['evidence_integrity_at_3']:.2%} | "
        "each item retains source, type, title, and text |",
        "",
        "## Knowledge Pack Integrity",
        "",
        f"- Documents: {pack['document_count']}",
        f"- Parsed sections: {pack['section_count']}",
        f"- Source types: {', '.join(pack['source_types'])}",
        f"- Duplicate titles: {len(pack['duplicate_titles'])}",
        f"- Empty sections: {len(pack['empty_sections'])}",
        f"- Suspicious instruction-like sections: {len(pack['suspicious_instruction_hits'])}",
        "",
        "## Gate Details",
        "",
    ]
    if issues:
        lines.extend(f"- {item}" for item in issues)
    else:
        lines.append("- All versioned retrieval and document-integrity thresholds passed.")
    if "support_retrieval" in report:
        support = report["support_retrieval"]
        hybrid = support["strategies"]["hybrid"]
        lines.extend(
            [
                "",
                "## Optional ResCommons Retrieval Component",
                "",
                f"- Cases: {support['case_count']}",
                "- Hybrid intent@1 / intent@5 / MRR@5: "
                f"{hybrid['intent_at_1']:.2%} / {hybrid['intent_at_5']:.2%} / "
                f"{hybrid['intent_mrr_at_5']:.2%}",
            ]
        )
    return "\n".join(lines) + "\n"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _reciprocal_rank(values: list[Any], expected: Any) -> float:
    for rank, value in enumerate(values, start=1):
        if value == expected:
            return 1.0 / rank
    return 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic policy-RAG release checks.")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--include-support-corpus", action="store_true")
    parser.add_argument("--gate", action="store_true", help="Exit non-zero when a versioned threshold fails.")
    args = parser.parse_args()

    report = evaluate_release(
        kb_dir=args.kb_dir,
        cases_path=args.cases,
        include_support_corpus=args.include_support_corpus,
    )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.gate else {}
    violations = gate_violations(report, baseline) if args.gate else []
    report["gate"] = {"enabled": args.gate, "passed": not violations, "violations": violations}

    rendered = render_markdown(report, violations)
    print(rendered, end="")
    if args.output:
        _write_json(args.output, report)
        print(f"Wrote {args.output}")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.markdown_output}")
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
