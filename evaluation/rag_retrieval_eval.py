from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.olist.catalog import load_orders
from app.olist.retrieval import (
    RetrievalHit,
    adaptive_category_retrieval,
    exact_underscore_retrieval,
    token_overlap_retrieval,
)

Retriever = Callable[[str, list[str], int], list[RetrievalHit]]


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    expected_category: str
    variant: str
    group: str


REALISTIC_ALIASES: dict[str, list[str]] = {
    "health_beauty": ["beauty care", "cosmetics", "美妆个护", "健康美容"],
    "bed_bath_table": ["home bedding", "bath and bedding", "床品家居", "床上用品"],
    "furniture_decor": ["home decor", "furnitures decoration", "家具装饰", "软装"],
    "computers_accessories": ["computer accessories", "pc accessories", "电脑配件", "外设"],
    "sports_leisure": ["fitness gear", "sports outdoor", "运动户外", "健身用品"],
    "watches_gifts": ["watch gift", "watches and presents", "手表礼品", "礼品手表"],
    "toys": ["kids toys", "children toy", "儿童玩具", "玩具"],
    "pet_shop": ["pet supplies", "pet store", "宠物用品", "猫狗用品"],
    "auto": ["car accessories", "automotive parts", "汽车用品", "车品"],
    "office_furniture": ["office chairs", "desk furniture", "办公家具", "工位桌椅"],
}

NOISY_HOLDOUT_ALIASES: dict[str, list[str]] = {
    "health_beauty": ["beautycare", "makeup stuff"],
    "bed_bath_table": ["bedroom bathroom things", "home linen"],
    "furniture_decor": ["home deco", "living room decoration"],
    "computers_accessories": ["computer peripherials", "keyboard mouse cables"],
    "sports_leisure": ["sporting goods", "outdoor exercise"],
    "watches_gifts": ["wristwatch presents", "gift watches"],
    "toys": ["kids stuff", "children play items"],
    "pet_shop": ["pet food and toys", "dog cat supplies"],
    "auto": ["car parts", "vehicle accessories"],
    "office_furniture": ["desk chair", "workstation furniture"],
}


def build_cases(limit_categories: int = 50) -> list[RetrievalCase]:
    categories = []
    seen = set()
    for order in load_orders():
        for product in order["products"]:
            category = str(product["category"])
            if category != "unknown" and category not in seen:
                seen.add(category)
                categories.append(category)

    cases: list[RetrievalCase] = []
    for category in categories[:limit_categories]:
        variants = {
            "canonical": category,
            "space_alias": category.replace("_", " "),
            "hyphen_alias": category.replace("_", "-"),
            "compact_alias": category.replace("_", ""),
        }
        for variant, mention in variants.items():
            cases.append(
                RetrievalCase(
                    case_id=f"{variant}-{category}",
                    query=f"{mention} 类目的订单主要有哪些物流和评价风险？",
                    expected_category=category,
                    variant=variant,
                    group="mechanical_alias",
                )
            )
    for category, aliases in REALISTIC_ALIASES.items():
        for index, mention in enumerate(aliases):
            cases.append(
                RetrievalCase(
                    case_id=f"realistic_alias-{category}-{index}",
                    query=f"{mention} 相关商品最近售后、物流和评价表现怎么样？",
                    expected_category=category,
                    variant="realistic_alias",
                    group="realistic_alias",
                )
            )
    for category, aliases in NOISY_HOLDOUT_ALIASES.items():
        for index, mention in enumerate(aliases):
            cases.append(
                RetrievalCase(
                    case_id=f"noisy_holdout_alias-{category}-{index}",
                    query=f"{mention} 这类商品有没有售后风险？",
                    expected_category=category,
                    variant="noisy_holdout_alias",
                    group="noisy_holdout_alias",
                )
            )
    return cases


def evaluate(name: str, retriever: Retriever, categories: list[str], cases: list[RetrievalCase]) -> dict:
    top1 = 0
    recall3 = 0
    reciprocal_rank = 0.0
    failures: list[str] = []

    for case in cases:
        hits = retriever(case.query, categories, 4)
        ranked = [hit.category for hit in hits]
        if ranked and ranked[0] == case.expected_category:
            top1 += 1
        if case.expected_category in ranked[:3]:
            recall3 += 1
            reciprocal_rank += 1 / (ranked.index(case.expected_category) + 1)
        else:
            failures.append(case.case_id)

    total = len(cases)
    return {
        "strategy": name,
        "cases": total,
        "top1": top1 / total,
        "recall@3": recall3 / total,
        "mrr@3": reciprocal_rank / total,
        "failures": failures,
    }


def evaluate_by_group(
    name: str,
    retriever: Retriever,
    categories: list[str],
    cases: list[RetrievalCase],
) -> list[dict]:
    groups = sorted({case.group for case in cases})
    return [
        {
            "group": group,
            **evaluate(name, retriever, categories, [case for case in cases if case.group == group]),
        }
        for group in groups
    ]


def main() -> None:
    categories = sorted(
        {
            str(product["category"])
            for order in load_orders()
            for product in order["products"]
            if product["category"] != "unknown"
        }
    )
    cases = build_cases()
    strategies: list[tuple[str, Retriever]] = [
        ("exact_underscore", exact_underscore_retrieval),
        ("token_overlap", token_overlap_retrieval),
        ("adaptive_rewrite", adaptive_category_retrieval),
    ]

    print(f"RAG retrieval eval cases: {len(cases)}")
    print("strategy,cases,top1,recall@3,mrr@3,failures")
    for name, retriever in strategies:
        result = evaluate(name, retriever, categories, cases)
        print(
            f"{result['strategy']},{result['cases']},"
            f"{result['top1']:.2%},{result['recall@3']:.2%},{result['mrr@3']:.2%},"
            f"{len(result['failures'])}"
        )
        if result["failures"]:
            print("  sample_failures=" + ",".join(result["failures"][:5]))
        for group_result in evaluate_by_group(name, retriever, categories, cases):
            print(
                "  group={group},cases={cases},top1={top1:.2%},recall@3={recall:.2%},mrr@3={mrr:.2%}".format(
                    group=group_result["group"],
                    cases=group_result["cases"],
                    top1=group_result["top1"],
                    recall=group_result["recall@3"],
                    mrr=group_result["mrr@3"],
                )
            )


if __name__ == "__main__":
    main()
