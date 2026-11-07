from app.olist.retrieval import (
    adaptive_category_retrieval,
    exact_underscore_retrieval,
    token_overlap_retrieval,
)

CATEGORIES = ["telephony", "fixed_telephony", "home_appliances", "home_confort"]


def test_exact_baseline_only_matches_canonical_form() -> None:
    assert exact_underscore_retrieval("fixed_telephony 类目", CATEGORIES, 1)[0].category == "fixed_telephony"
    assert exact_underscore_retrieval("fixed telephony 类目", CATEGORIES, 1)[0].category != "fixed_telephony"


def test_token_overlap_handles_spaces_but_not_compact_alias() -> None:
    assert token_overlap_retrieval("fixed telephony 类目", CATEGORIES, 1)[0].category == "fixed_telephony"
    assert token_overlap_retrieval("fixedtelephony 类目", CATEGORIES, 1) == []


def test_adaptive_retrieval_handles_compact_alias_and_prefers_longer_category() -> None:
    assert adaptive_category_retrieval("fixedtelephony 类目", CATEGORIES, 1)[0].category == "fixed_telephony"
    assert adaptive_category_retrieval("fixed_telephony 类目", CATEGORIES, 1)[0].category == "fixed_telephony"
