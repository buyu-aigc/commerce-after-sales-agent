"""Shared fixtures for the P0-1 Playwright browser E2E suite.

The suite is opt-in (``RUN_E2E=1``) so the regular unit/integration test run
stays fast and does not require a browser.  Each test gets a disposable copy of
the V2 app with its own SQLite files, a free port, and the offline workflow
(``OPENAI_API_KEY=test-key``).  No V1 process, database, or online service is
touched.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

if os.environ.get("RUN_E2E") != "1":
    pytest.skip(
        "P0 browser E2E is opt-in: run tests/e2e/run.sh or set RUN_E2E=1",
        allow_module_level=True,
    )

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ORDER_DELAYED = "203096f03d82e0dffbc41ebc2e2bcfb7"  # delivered, delayed
ORDER_OTHER_OWNER = "8aec3a066f732dd927ec8fef1752415b"  # owned by demo-customer/customer-demo only
ORDER_HIGH_VALUE = "360787554f41824600bdcba9687cd4f0"  # invoiced, high value -> review
REVIEW_TOKEN = os.environ.get("P0_E2E_REVIEW_TOKEN", "e2e-review-token")
REVIEW_PAYLOAD = {
    "reviewer_id": "e2e-reviewer",
    "role": "after_sales_operator",
    "auth_scopes": ["after_sales:write"],
}
LIVE_LLM = os.environ.get("P0_E2E_LIVE") == "1"
ARTIFACT_DIR = Path(
    os.environ.get("P0_E2E_ARTIFACT_DIR")
    or (ROOT / "reports" / "p0-e2e")
)


@dataclass
class Server:
    base_url: str
    runtime: Path
    log_path: Path
    token: str = REVIEW_TOKEN


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_runtime(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for name in ("app", "frontend", "policies"):
        shutil.copytree(ROOT / name, destination / name, ignore=ignore)
    for name in ("olist_derived", "knowledge_base", "rescommons_derived"):
        shutil.copytree(ROOT / "data" / name, destination / "data" / name, ignore=ignore)
    (destination / "data").mkdir(parents=True, exist_ok=True)
    return destination


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _wait_ready(server: Server, process: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    probe = f"{server.base_url}/runtime/status"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = server.log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"V2 server exited early:\n{tail}")
        try:
            with _opener().open(probe, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - readiness probe
            last_error = exc
        time.sleep(0.2)
    tail = server.log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(f"V2 server did not become ready ({last_error}):\n{tail}")


@pytest.fixture()
def app_server(tmp_path: Path):
    runtime = _build_runtime(tmp_path / "v2-runtime")
    port = _free_port()
    log_path = runtime / "server.log"
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "OPENAI_API_KEY": (
            os.environ.get("OPENAI_API_KEY", "")
            if LIVE_LLM
            else "test-key"
        ),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
        "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", ""),
        "LLM_TIMEOUT_SECONDS": os.environ.get("LLM_TIMEOUT_SECONDS", "60" if LIVE_LLM else "12"),
        "DEMO_MODE": "1",
        "REVIEW_API_TOKEN": REVIEW_TOKEN,
        "RUNTIME_STORE_BACKEND": "memory",
        "PYTHONUNBUFFERED": "1",
    }
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=runtime,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    server = Server(base_url=f"http://127.0.0.1:{port}", runtime=runtime, log_path=log_path)
    _wait_ready(server, process)
    try:
        yield server
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        yield instance
        instance.close()


def new_context(browser: Browser, **kwargs) -> BrowserContext:
    context = browser.new_context(
        viewport={"width": 1440, "height": 1000},
        locale="zh-CN",
        **kwargs,
    )
    context.set_default_timeout(120_000 if LIVE_LLM else 15_000)
    return context


@pytest.fixture()
def page(browser: Browser):
    context = new_context(browser)
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


# ---------------------------------------------------------------------------
# Helpers shared by the browser tests
# ---------------------------------------------------------------------------


def create_case(
    page: Page,
    server: Server,
    *,
    session_id: str,
    action_type: str = "refund_request",
    order_id: str = ORDER_DELAYED,
    customer: str = "demo-customer",
    surface: str = "order_detail",
) -> str:
    response = page.request.post(
        f"{server.base_url}/customer/chat",
        headers={"X-Demo-Customer": customer},
        data={
            "message": _action_message(action_type),
            "session_id": session_id,
            "requested_action": action_type,
            "page_context": {"surface": surface, "selected_order_id": order_id},
        },
    )
    assert response.ok, response.text()
    payload = response.json()
    assert payload.get("case_id"), payload
    return str(payload["case_id"])


def _action_message(action_type: str) -> str:
    return {
        "refund_request": "我想为这笔订单申请退款。",
        "cancel_order": "我想取消这笔订单。",
        "complaint_escalation": "这笔订单遇到了问题，我想申请售后处理。",
        "change_address": "我想修改这笔订单的收货地址。",
        "invoice_request": "我想为这笔订单申请发票。",
        "open_support_case": "这笔订单遇到了问题，我想申请售后处理。",
    }.get(action_type, "我想申请售后处理。")


def review_login(page: Page, server: Server, token: str = REVIEW_TOKEN) -> None:
    page.goto(f"{server.base_url}/review")
    page.fill("#token", token)
    page.click("#login button")


def review_case(page: Page, server: Server, case_id: str) -> dict:
    response = page.request.get(
        f"{server.base_url}/review/cases/{case_id}",
        headers={"X-Review-Token": REVIEW_TOKEN},
    )
    assert response.ok, response.text()
    return response.json()


def decide_case(page: Page, server: Server, case_id: str, decision: str) -> dict:
    response = page.request.post(
        f"{server.base_url}/review/cases/{case_id}/{decision}",
        headers={"X-Review-Token": REVIEW_TOKEN},
        data=REVIEW_PAYLOAD,
    )
    assert response.ok, response.text()
    return response.json()


def customer_progress(page: Page, server: Server, session_id: str, customer: str = "demo-customer") -> dict:
    response = page.request.get(
        f"{server.base_url}/customer/sessions/{session_id}/progress",
        headers={"X-Demo-Customer": customer},
    )
    return {"status": response.status, "body": response.json() if response.ok else response.text()}


def boot_customer(page: Page, server: Server, session_id: str | None = None) -> str:
    page.goto(f"{server.base_url}/customer")
    page.wait_for_selector("#orders article.order")
    if session_id:
        page.evaluate("(value) => sessionStorage.setItem('support-session', value)", session_id)
        page.reload()
        page.wait_for_selector("#orders article.order")
    current = page.evaluate("sessionStorage.getItem('support-session')")
    assert current
    return str(current)


def save_screenshot(page: Page, artifacts_dir: Path, name: str) -> Path:
    path = artifacts_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return path
