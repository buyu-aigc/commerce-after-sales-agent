from app.agent.actions import _missing_address_fields


def test_address_change_requires_actionable_recipient_fields() -> None:
    assert _missing_address_fields("请改订单的收货地址") == ["收件人", "联系电话", "详细新地址"]


def test_address_change_accepts_complete_customer_payload() -> None:
    message = "新地址：上海市徐汇区天平路 1 号；收件人张三；电话 13800138000"
    assert _missing_address_fields(message) == []
