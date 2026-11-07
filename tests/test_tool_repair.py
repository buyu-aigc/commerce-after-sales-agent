from app.tools.repair import repair_order_id


def test_repair_order_id_normalizes_spaces_and_case() -> None:
    result = repair_order_id("203096F0 3D82 E0DF FBC41EBC2E2BCFB7")
    assert result.ok is True
    assert result.value == "203096f03d82e0dffbc41ebc2e2bcfb7"


def test_repair_order_id_returns_structured_error() -> None:
    result = repair_order_id("203096f03d82")
    assert result.ok is False
    assert result.error_code == "incomplete_order_id"
