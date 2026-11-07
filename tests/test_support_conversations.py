from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import support_conversations as store
from app.llm.customer_presenter import present


def test_handoff_and_customer_messages_persist_without_internal_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "conversations.db")
    store.initialize()
    store.append_message("a", "customer", "包裹还没到")
    store.set_human_mode("a", True)
    store.append_message("a", "staff", "我来帮您核查。")
    store.append_message("b", "customer", "另一位客户")
    store.initialize()
    assert store.human_mode("a") is True
    assert store.human_mode("b") is False
    rows = store.messages("a")
    assert [row["content"] for row in rows] == ["包裹还没到", "我来帮您核查。"]
    assert all(set(row) == {"id", "sender", "content", "created_at"} for row in rows)
    store.set_human_mode("a", False)
    assert store.human_mode("a") is False


@pytest.mark.anyio
async def test_customer_presenter_blocks_unexecuted_refund_claim():
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="退款成功，已到账。")))
    answer = await present(llm, "我要退款", {}, {"status": "pending_review"})
    assert "退款成功" not in answer
    assert "已到账" not in answer


@pytest.mark.anyio
async def test_customer_presenter_keeps_plain_grounded_answer():
    reply = SimpleNamespace(content="订单显示已送达，请确认是否收到。")
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=reply))
    answer = await present(llm, "到货了吗", {"final_answer": "delivered"}, None)
    assert answer == "订单显示已送达，请确认是否收到。"
