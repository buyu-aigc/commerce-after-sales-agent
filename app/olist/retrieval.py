from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    category: str
    score: float
    reason: str


CATEGORY_ALIAS_MAP: dict[str, set[str]] = {
    "health_beauty": {"beauty care", "cosmetics", "美妆个护", "健康美容"},
    "bed_bath_table": {"home bedding", "bath and bedding", "床品家居", "床上用品"},
    "furniture_decor": {"home decor", "furniture decoration", "家具装饰", "软装"},
    "computers_accessories": {"computer accessories", "pc accessories", "电脑配件", "外设"},
    "sports_leisure": {"fitness gear", "sports outdoor", "运动户外", "健身用品"},
    "watches_gifts": {"watch gift", "watches and presents", "手表礼品", "礼品手表"},
    "toys": {"kids toys", "children toy", "儿童玩具", "玩具"},
    "pet_shop": {"pet supplies", "pet store", "宠物用品", "猫狗用品"},
    "auto": {"car accessories", "automotive parts", "汽车用品", "车品"},
    "office_furniture": {"office chairs", "desk furniture", "办公家具", "工位桌椅"},
}


def exact_underscore_retrieval(query: str, categories: list[str], k: int = 4) -> list[RetrievalHit]:
    """Baseline: only matches the canonical category string used in the dataset."""
    normalized_query = query.lower()
    hits = [
        RetrievalHit(category=category, score=1.0, reason="canonical_substring")
        for category in categories
        if category and category in normalized_query
    ]
    return _rank(hits)[:k]


def token_overlap_retrieval(query: str, categories: list[str], k: int = 4) -> list[RetrievalHit]:
    """Lexical retriever that tolerates spaces/hyphens but not joined-word aliases."""
    query_tokens = set(_tokens(query))
    hits: list[RetrievalHit] = []
    if not query_tokens:
        return hits

    for category in categories:
        category_tokens = set(_tokens(category))
        if not category_tokens:
            continue
        overlap = query_tokens & category_tokens
        if overlap:
            score = len(overlap) / len(category_tokens)
            hits.append(RetrievalHit(category=category, score=score, reason="token_overlap"))
    return _rank(hits)[:k]


def adaptive_category_retrieval(query: str, categories: list[str], k: int = 4) -> list[RetrievalHit]:
    """Deterministic adaptive retrieval with query rewriting and longest-form priority.

    This intentionally stays local and explainable: first try rewritten aliases
    generated from real category names, then fall back to token overlap.
    """
    normalized_query = query.lower()
    compact_query = _compact(query)
    hits_by_category: dict[str, RetrievalHit] = {}

    for category in categories:
        alias_forms = {
            category.lower(),
            category.lower().replace("_", " "),
            category.lower().replace("_", "-"),
            category.lower().replace("_", ""),
        }
        alias_forms.update(alias.lower() for alias in CATEGORY_ALIAS_MAP.get(category, set()))
        if any(alias and alias in normalized_query for alias in alias_forms):
            hits_by_category[category] = RetrievalHit(
                category=category,
                score=10.0 + len(category) / 100.0,
                reason="rewritten_alias",
            )
            continue
        if category.lower().replace("_", "") in compact_query:
            hits_by_category[category] = RetrievalHit(
                category=category,
                score=9.0 + len(category) / 100.0,
                reason="compact_alias",
            )

    for hit in token_overlap_retrieval(query, categories, k=len(categories)):
        if hit.category not in hits_by_category:
            hits_by_category[hit.category] = hit

    return _rank(list(hits_by_category.values()))[:k]


def _tokens(text: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", text.lower().replace("_", " ")) if part]


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _rank(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    return sorted(hits, key=lambda hit: (hit.score, len(hit.category), hit.category), reverse=True)
