from app.olist.knowledge import MarkdownKnowledgeBase


def test_markdown_knowledge_base_searches_policy_sections() -> None:
    hits = MarkdownKnowledgeBase().search("refund coupon compensation human confirmation", k=2)
    assert hits
    assert "human confirmation" in hits[0].text.lower() or "confirmation" in hits[0].text.lower()


def test_markdown_knowledge_base_searches_support_faq() -> None:
    hits = MarkdownKnowledgeBase().search("退款进度是否已经完成", k=3)

    assert hits
    assert any(hit.source_type == "faq" for hit in hits)
    assert any("Refund Status FAQ" == hit.section_title for hit in hits)


def test_markdown_knowledge_base_searches_merchant_rules() -> None:
    hits = MarkdownKnowledgeBase().search("美妆 health beauty 已送达订单能不能取消", k=5)

    assert hits
    assert any(hit.source_type == "merchant_rule" for hit in hits)
    assert any("Health Beauty Merchant Rule" == hit.section_title for hit in hits)
