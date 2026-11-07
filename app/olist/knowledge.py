from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@dataclass(frozen=True)
class KnowledgeHit:
    source: str
    source_type: str
    section_title: str
    text: str
    score: float


class MarkdownKnowledgeBase:
    """Small markdown KB adapter for policy/FAQ-style business documents."""

    def __init__(self, kb_dir: Path = _DEFAULT_KB_DIR) -> None:
        self._kb_dir = kb_dir

    def search(self, query: str, k: int = 3) -> list[KnowledgeHit]:
        sections = self._sections()
        expanded_query = _expand_query(query)
        query_tokens = set(_tokens(expanded_query))
        scored = []
        for section in sections:
            title_tokens = set(_tokens(section["title"]))
            text_tokens = set(_tokens(section["text"]))
            title_overlap = query_tokens & title_tokens
            text_overlap = query_tokens & text_tokens
            if not title_overlap and not text_overlap:
                continue
            score = len(title_overlap) * 3.0 + len(text_overlap) / max(len(query_tokens), 1)
            scored.append((score, section))
        candidates = sorted(scored, key=lambda item: item[0], reverse=True)[: max(k * 4, 8)]
        ranked = sorted(
            candidates,
            key=lambda item: _rerank_score(item[0], item[1], expanded_query),
            reverse=True,
        )[:k]
        return [
            KnowledgeHit(
                source=section["source"],
                source_type=section["source_type"],
                section_title=section["title"],
                text=section["text"],
                score=_rerank_score(score, section, expanded_query),
            )
            for score, section in ranked
        ]

    def _sections(self) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        for path in sorted(self._kb_dir.glob("*.md")):
            current_title = path.stem
            current_lines: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("## "):
                    if current_lines:
                        sections.append(_section(path.name, current_title, current_lines))
                    current_title = line.removeprefix("## ").strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            if current_lines:
                sections.append(_section(path.name, current_title, current_lines))
        return sections


def _section(source: str, title: str, lines: list[str]) -> dict[str, str]:
    text = "\n".join(line for line in lines if line.strip()).strip()
    return {
        "source": source,
        "source_type": _source_type(source),
        "title": title,
        "text": f"{title}\n{text}",
    }


def _source_type(source: str) -> str:
    stem = Path(source).stem.lower()
    if "faq" in stem:
        return "faq"
    if "merchant" in stem or "rule" in stem:
        return "merchant_rule"
    if "policy" in stem:
        return "policy"
    return "knowledge"


def _rerank_score(base_score: float, section: dict[str, str], expanded_query: str) -> float:
    query = expanded_query.lower()
    source_type = section["source_type"]
    title = section["title"].lower()
    boost = 0.0
    if source_type == "faq" and any(token in query for token in ("faq", "进度", "标签", "没收到", "投诉")):
        boost += 1.2
    if source_type == "merchant_rule" and any(
        token in query for token in ("商家", "类目", "美妆", "数码", "高金额", "merchant", "seller")
    ):
        boost += 1.2
    if source_type == "policy" and any(
        token in query for token in ("政策", "退款", "取消", "发票", "补偿", "policy", "refund", "invoice")
    ):
        boost += 0.8
    if "complaint" in query and "complaint" in title:
        boost += 1.0
    if "return-label" in query and "label" in title:
        boost += 1.0
    return base_score + boost


def _tokens(text: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", text.lower()) if part]


def _expand_query(query: str) -> str:
    expansions = {
        "订单": "order",
        "状态": "status",
        "当前状态": "order status",
        "查什么": "query tool facts",
        "延迟": "delivery delay delayed",
        "晚于": "delivery delay delayed",
        "送达": "delivery delivered",
        "退款": "refund compensation",
        "退货": "return return-label damaged wrong-item missing-item",
        "优惠券": "coupon compensation",
        "补偿": "compensation coupon refund",
        "完成": "completed confirmed status",
        "进度": "status pending review",
        "物流": "logistics delivery package",
        "没收到": "not arrived delivery complaint pending logistics",
        "破损": "damaged package return label inspection",
        "错发": "wrong item return label inspection",
        "少发": "missing item return label inspection",
        "低分": "low review recovery",
        "1星": "low review",
        "2星": "low review",
        "取消": "cancellation canceled",
        "发票": "invoice request billing tax profile",
        "开票": "invoice request billing tax profile",
        "地址": "address change shipping recipient street postal",
        "收货地址": "address change shipping recipient street postal",
        "修改": "change request",
        "人工": "human approval confirmation",
        "投诉": (
            "complaint escalation repeated failed contacts legal threats safety concerns conflicting facts"
        ),
        "复核": "review reviewer human escalation",
        "确认": "approval confirmation",
        "创建": "create case side effect",
        "case": "case",
        "话术": "customer message style",
        "技术词": "internal language tool rag checkpoint workflow",
        "事实": "facts tool boundary",
        "哪里来": "tool boundary",
        "类目": "category operations",
        "运营风险": "category operations risk",
        "替代": "specific order exact lookup",
        "轨迹": "audit trace",
        "记录": "record trace",
        "审计": "audit trace",
        "商家": "merchant rule seller",
        "美妆": "health beauty hygiene damaged packaging safety",
        "数码": "electronics computer accessories serial warranty",
        "高金额": "high value payment human review",
    }
    extra = [value for key, value in expansions.items() if key in query]
    return f"{query} {' '.join(extra)}"
