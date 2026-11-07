from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app import observability
from app.after_sales import AfterSalesDecisionEngine
from app.olist.knowledge import MarkdownKnowledgeBase
from app.olist.service import (
    InMemoryCaseService,
    OlistService,
    SQLiteCaseService,
    format_order_status,
)
from app.retrieval.hybrid import HybridSupportRetriever


class ToolCallError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ToolCallContext(BaseModel):
    user_id: str = "demo-user"
    role: str = "support_agent"
    tenant_id: str = "olist-demo"
    session_id: str = "unknown"
    auth_scopes: list[str] = Field(default_factory=list)
    allowed_order_ids: list[str] = Field(default_factory=list)


class ToolCallResult(BaseModel):
    tool_name: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    cached: bool = False
    attempts: int = 1
    latency_ms: float = 0.0
    error_code: str | None = None
    error_message: str | None = None


class ToolCacheBackend(Protocol):
    async def get(self, key: str) -> ToolCallResult | None:
        """Return cached tool result, or None on miss/expiry."""

    async def set(self, key: str, result: ToolCallResult, ttl_seconds: float) -> None:
        """Persist a successful read-only tool result with TTL."""


class RateLimitDecision(BaseModel):
    allowed: bool
    key: str
    limit: int
    remaining: int
    reset_after_seconds: int


class RuntimeStore(ToolCacheBackend, Protocol):
    """Short-lived runtime coordination backend.

    Redis is useful for this layer because these records are intentionally
    ephemeral: read-only tool cache, request rate windows, HITL pending TTL, and
    short side-effect locks. Durable trace and final idempotency records still
    belong in SQLite or an append-only event stream.
    """

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        """Increment a fixed-window counter and return whether the call is allowed."""

    async def put_pending_confirmation(
        self,
        session_id: str,
        pending: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Store a HITL pending action with TTL."""

    async def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        """Return HITL pending action while its TTL is still alive."""

    async def clear_pending_confirmation(self, session_id: str) -> None:
        """Remove HITL pending action after confirm/cancel/timeout."""

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        """Return a lock token when acquired, or None when another worker holds it."""

    async def release_lock(self, key: str, token: str) -> None:
        """Release a lock only if its token still matches."""


class InMemoryToolCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, ToolCallResult]] = {}

    async def get(self, key: str) -> ToolCallResult | None:
        cached = self._store.get(key)
        if not cached:
            return None
        expires_at, result = cached
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return result

    async def set(self, key: str, result: ToolCallResult, ttl_seconds: float) -> None:
        self._store[key] = (time.time() + ttl_seconds, result)


class InMemoryRuntimeStore(InMemoryToolCache):
    """In-process runtime store used by tests and single-worker demos."""

    def __init__(self) -> None:
        super().__init__()
        self._counters: dict[str, tuple[float, int]] = {}
        self._pending: dict[str, tuple[float, dict[str, Any]]] = {}
        self._locks: dict[str, tuple[float, str]] = {}

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        now = time.time()
        expires_at, count = self._counters.get(key, (now + window_seconds, 0))
        if expires_at <= now:
            expires_at, count = now + window_seconds, 0
        count += 1
        self._counters[key] = (expires_at, count)
        remaining = max(limit - count, 0)
        return RateLimitDecision(
            allowed=count <= limit,
            key=key,
            limit=limit,
            remaining=remaining,
            reset_after_seconds=max(int(expires_at - now), 0),
        )

    async def put_pending_confirmation(
        self,
        session_id: str,
        pending: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        self._pending[session_id] = (time.time() + max(ttl_seconds, 1), dict(pending))

    async def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        cached = self._pending.get(session_id)
        if not cached:
            return None
        expires_at, pending = cached
        if expires_at <= time.time():
            self._pending.pop(session_id, None)
            return None
        return dict(pending)

    async def clear_pending_confirmation(self, session_id: str) -> None:
        self._pending.pop(session_id, None)

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        now = time.time()
        cached = self._locks.get(key)
        if cached and cached[0] > now:
            return None
        token = uuid.uuid4().hex
        self._locks[key] = (now + max(ttl_seconds, 1), token)
        return token

    async def release_lock(self, key: str, token: str) -> None:
        cached = self._locks.get(key)
        if cached and cached[1] == token:
            self._locks.pop(key, None)


class RedisToolCache:
    """Redis-backed cache for multi-worker tool-call deployments."""

    def __init__(self, client: Any, prefix: str = "olist-agent:tool-cache") -> None:
        self._client = client
        self._prefix = prefix.rstrip(":")

    @classmethod
    def from_url(cls, url: str, prefix: str = "olist-agent:tool-cache") -> RedisToolCache:
        try:
            from redis import asyncio as redis_asyncio
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install redis support with `pip install .[redis]`.") from exc
        return cls(redis_asyncio.from_url(url, decode_responses=True), prefix=prefix)

    async def get(self, key: str) -> ToolCallResult | None:
        raw = await self._client.get(self._namespaced(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return ToolCallResult.model_validate_json(raw)

    async def set(self, key: str, result: ToolCallResult, ttl_seconds: float) -> None:
        await self._client.set(self._namespaced(key), result.model_dump_json(), ex=max(int(ttl_seconds), 1))

    def _namespaced(self, key: str) -> str:
        return f"{self._prefix}:{key}"


class RedisRuntimeStore(RedisToolCache):
    """Redis-backed runtime store for multi-worker deployments."""

    def __init__(self, client: Any, prefix: str = "olist-agent") -> None:
        super().__init__(client, prefix=f"{prefix.rstrip(':')}:tool-cache")
        self._runtime_prefix = prefix.rstrip(":")

    @classmethod
    def from_url(cls, url: str, prefix: str = "olist-agent") -> RedisRuntimeStore:
        try:
            from redis import asyncio as redis_asyncio
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install redis support with `pip install .[redis]`.") from exc
        return cls(redis_asyncio.from_url(url, decode_responses=True), prefix=prefix)

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        namespaced = self._runtime_key("rate", key)
        count = int(await self._client.incr(namespaced))
        if count == 1:
            await self._client.expire(namespaced, max(window_seconds, 1))
            reset_after = window_seconds
        else:
            ttl = int(await self._client.ttl(namespaced))
            reset_after = ttl if ttl > 0 else window_seconds
        return RateLimitDecision(
            allowed=count <= limit,
            key=key,
            limit=limit,
            remaining=max(limit - count, 0),
            reset_after_seconds=max(reset_after, 0),
        )

    async def put_pending_confirmation(
        self,
        session_id: str,
        pending: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        await self._client.set(
            self._runtime_key("hitl", session_id),
            json.dumps(pending, ensure_ascii=False, sort_keys=True),
            ex=max(ttl_seconds, 1),
        )

    async def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._runtime_key("hitl", session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return dict(json.loads(raw))

    async def clear_pending_confirmation(self, session_id: str) -> None:
        await self._client.delete(self._runtime_key("hitl", session_id))

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        token = uuid.uuid4().hex
        acquired = await self._client.set(
            self._runtime_key("lock", key),
            token,
            ex=max(ttl_seconds, 1),
            nx=True,
        )
        return token if acquired else None

    async def release_lock(self, key: str, token: str) -> None:
        redis_key = self._runtime_key("lock", key)
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        end
        return 0
        """
        await self._client.eval(script, 1, redis_key, token)

    def _runtime_key(self, kind: str, key: str) -> str:
        return f"{self._runtime_prefix}:{kind}:{key}"


class OrderStatusArgs(BaseModel):
    order_id: str = Field(..., min_length=32, max_length=64)


class QueryArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class OwnedOrdersArgs(BaseModel):
    filter: str = Field(default="all", pattern="^(all|in_transit|attention)$")


class PolicySearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=10)


class SupportSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=10)


class SideEffectArgs(BaseModel):
    action_type: str = Field(..., min_length=1, max_length=64)
    order_id: str = Field(..., min_length=32, max_length=64)
    message_text: str = Field(..., min_length=1, max_length=4000)
    case_id: str | None = Field(default=None, max_length=128)


class AfterSalesCaseArgs(BaseModel):
    action_type: str = Field(..., min_length=1, max_length=64)
    order_id: str = Field(..., min_length=32, max_length=64)
    user_request: str = Field(..., min_length=1, max_length=4000)
    policy_sections: list[dict[str, Any]] = Field(default_factory=list)


ToolHandler = Callable[[dict[str, Any], ToolCallContext], dict[str, Any] | Awaitable[dict[str, Any]]]
FallbackHandler = Callable[[dict[str, Any], ToolCallContext, Exception], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    allowed_roles: frozenset[str] = field(default_factory=lambda: frozenset({"support_agent"}))
    side_effect: bool = False
    risk_level: str = "read"
    auth_scope: str = "support:read"
    idempotency_required: bool = False
    cache_ttl_seconds: float | None = None
    timeout_seconds: float = 2.0
    retries: int = 0
    backoff_seconds: float = 0.05
    fallback: FallbackHandler | None = None


class ToolCallManager:
    """Governed tool runner for deterministic business tools.

    The manager centralizes the production concerns around tool execution:
    schema validation, role allowlists, cache, async execution,
    timeout/retry/backoff, fallback, normalized output, and audit events.
    """

    def __init__(
        self,
        specs: list[ToolSpec],
        cache_backend: ToolCacheBackend | None = None,
        runtime_store: RuntimeStore | None = None,
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._runtime = runtime_store
        self._cache = cache_backend or runtime_store or InMemoryRuntimeStore()
        self.audit_events: list[dict[str, Any]] = []

    def list_tools(self) -> list[str]:
        return sorted(self._specs)

    def list_tool_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "side_effect": spec.side_effect,
                "risk_level": spec.risk_level,
                "auth_scope": spec.auth_scope,
                "allowed_roles": sorted(spec.allowed_roles),
                "idempotency_required": spec.idempotency_required,
                "cache_ttl_seconds": spec.cache_ttl_seconds,
                "timeout_seconds": spec.timeout_seconds,
                "retries": spec.retries,
            }
            for spec in sorted(self._specs.values(), key=lambda item: item.name)
        ]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolCallContext | None = None,
    ) -> ToolCallResult:
        context = context or ToolCallContext()
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            result = self._failure(name, "unknown_tool", f"Unknown tool: {name}", started)
            self._audit(context, name, arguments, result)
            return result

        try:
            validated = spec.input_model.model_validate(arguments).model_dump()
            self._check_permission(spec, context)
            self._check_customer_order_ownership(validated, context)
        except Exception as exc:
            error_code = exc.error_code if isinstance(exc, ToolCallError) else "schema_validation_failed"
            result = self._failure(name, error_code, str(exc), started)
            self._audit(context, name, arguments, result)
            return result

        cache_key = (
            self._cache_key(name, validated, context.tenant_id)
            if spec.cache_ttl_seconds and not spec.side_effect
            else None
        )
        if cache_key:
            cached = await self._cache.get(cache_key)
            if cached:
                result = cached.model_copy(update={"cached": True})
                self._audit(context, name, validated, result)
                return result

        if spec.side_effect and self._runtime is not None:
            result = await self._execute_with_lock(spec, validated, context, started)
        else:
            result = await self._execute(spec, validated, context, started)
        if cache_key and result.ok:
            await self._cache.set(cache_key, result, float(spec.cache_ttl_seconds))
        self._audit(context, name, validated, result)
        return result

    async def _execute_with_lock(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolCallContext,
        started: float,
    ) -> ToolCallResult:
        lock_key = self._cache_key(spec.name, arguments, context.tenant_id)
        lock_ttl = max(int((spec.timeout_seconds + spec.backoff_seconds) * (spec.retries + 1)) + 1, 3)
        token = await self._runtime.acquire_lock(f"side-effect:{lock_key}", lock_ttl)
        if token is None:
            return self._failure(
                spec.name,
                "side_effect_in_progress",
                "Another worker is already executing the same side-effect request.",
                started,
            )
        try:
            return await self._execute(spec, arguments, context, started)
        finally:
            await self._runtime.release_lock(f"side-effect:{lock_key}", token)

    async def _execute(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: ToolCallContext,
        started: float,
    ) -> ToolCallResult:
        attempts = 0
        last_error: Exception | None = None
        for attempt in range(spec.retries + 1):
            attempts = attempt + 1
            try:
                data = await asyncio.wait_for(
                    _invoke(spec.handler, arguments, context),
                    timeout=spec.timeout_seconds,
                )
                return ToolCallResult(
                    tool_name=spec.name,
                    ok=True,
                    data=data,
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except Exception as exc:
                last_error = exc
                if attempt < spec.retries:
                    await asyncio.sleep(spec.backoff_seconds * (2**attempt))

        assert last_error is not None
        if spec.fallback is not None:
            fallback_data = spec.fallback(arguments, context, last_error)
            return ToolCallResult(
                tool_name=spec.name,
                ok=True,
                data=fallback_data,
                attempts=attempts,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="fallback_used",
                error_message=str(last_error),
            )
        return self._failure(spec.name, _error_code(last_error), str(last_error), started, attempts)

    @staticmethod
    def _check_permission(spec: ToolSpec, context: ToolCallContext) -> None:
        if context.role not in spec.allowed_roles:
            raise ToolCallError(
                "permission_denied",
                f"Role {context.role!r} is not allowed to call {spec.name!r}",
            )
        if not context.auth_scopes and spec.risk_level != "read":
            raise ToolCallError(
                "permission_denied",
                f"Scope {spec.auth_scope!r} is required to call non-read tool {spec.name!r}",
            )
        if (
            context.auth_scopes
            and "*" not in context.auth_scopes
            and spec.auth_scope not in context.auth_scopes
        ):
            raise ToolCallError(
                "permission_denied",
                f"Scope {spec.auth_scope!r} is required to call {spec.name!r}",
            )

    @staticmethod
    def _check_customer_order_ownership(arguments: dict[str, Any], context: ToolCallContext) -> None:
        """Keep model-produced tool arguments inside the authenticated order set."""
        if context.role != "customer" or "order_id" not in arguments:
            return
        order_id = str(arguments["order_id"]).lower()
        if order_id not in {value.lower() for value in context.allowed_order_ids}:
            raise ToolCallError(
                "order_owner_required",
                "The requested order is not available to the current customer.",
            )

    @staticmethod
    def _failure(
        name: str,
        error_code: str,
        error_message: str,
        started: float,
        attempts: int = 1,
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_name=name,
            ok=False,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_code=error_code,
            error_message=error_message,
        )

    def _audit(
        self,
        context: ToolCallContext,
        name: str,
        arguments: dict[str, Any],
        result: ToolCallResult,
    ) -> None:
        redacted_arguments = _redact_args(arguments)
        self.audit_events.append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tenant_id": context.tenant_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "role": context.role,
                "auth_scopes": context.auth_scopes,
                "tool_name": name,
                "risk_level": self._specs[name].risk_level if name in self._specs else "unknown",
                "side_effect": self._specs[name].side_effect if name in self._specs else False,
                "args_hash": self._cache_key(name, arguments),
                "cache_key": self._cache_key(name, arguments, context.tenant_id),
                "redacted_args": redacted_arguments,
                "ok": result.ok,
                "cached": result.cached,
                "attempts": result.attempts,
                "latency_ms": round(result.latency_ms, 3),
                "error_code": result.error_code,
                "result_summary": _result_summary(result.data),
            }
        )
        observability.record_tool_call(
            tool=name,
            latency_ms=result.latency_ms,
            ok=result.ok,
            error_code=result.error_code,
            result_count=_result_count(result.data),
            redacted_args=redacted_arguments,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            session_id=context.session_id,
        )

    @staticmethod
    def _cache_key(name: str, arguments: dict[str, Any], tenant_id: str | None = None) -> str:
        payload = json.dumps(
            {"tenant_id": tenant_id, "tool": name, "args": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _invoke(
    handler: ToolHandler,
    arguments: dict[str, Any],
    context: ToolCallContext,
) -> dict[str, Any]:
    if inspect.iscoroutinefunction(handler):
        return await handler(arguments, context)
    result = await asyncio.to_thread(handler, arguments, context)
    if inspect.isawaitable(result):
        return await result
    return result


def build_business_tool_manager(
    *,
    olist_service: OlistService,
    knowledge_base: MarkdownKnowledgeBase,
    support_retriever: HybridSupportRetriever,
    case_service: InMemoryCaseService | SQLiteCaseService,
    cache_backend: ToolCacheBackend | None = None,
    runtime_store: RuntimeStore | None = None,
) -> ToolCallManager:
    olist_service.set_projection_store(case_service)
    support_roles = frozenset({"customer", "support_agent", "after_sales_operator", "ops_manager", "admin"})
    ops_roles = frozenset({"ops_manager", "admin"})
    after_sales_engine = AfterSalesDecisionEngine()

    return ToolCallManager(
        [
            ToolSpec(
                name="get_order_status",
                description="Return deterministic order, delivery, payment, and review facts.",
                input_model=OrderStatusArgs,
                allowed_roles=support_roles,
                risk_level="read",
                auth_scope="orders:read",
                cache_ttl_seconds=300,
                timeout_seconds=3,
                retries=1,
                fallback=lambda args, ctx, exc: {
                    "answer": "订单事实工具暂时不可用，请稍后重试或转人工核查。",
                    "found": False,
                    "fallback_reason": str(exc),
                },
                handler=lambda args, ctx: _order_status_payload(olist_service, str(args["order_id"])),
            ),
            ToolSpec(
                name="list_owned_orders",
                description="List compact owned-order summaries for a customer-facing order assistant.",
                input_model=OwnedOrdersArgs,
                allowed_roles=support_roles,
                risk_level="read",
                auth_scope="orders:read",
                cache_ttl_seconds=60,
                timeout_seconds=3,
                retries=1,
                handler=lambda args, ctx: {
                    "orders": _owned_order_rows(
                        olist_service,
                        ctx.allowed_order_ids if ctx.role == "customer" else [],
                        str(args["filter"]),
                    )
                },
            ),
            ToolSpec(
                name="search_category_risk",
                description="Retrieve category-level logistics and review risk insights.",
                input_model=QueryArgs,
                allowed_roles=support_roles,
                risk_level="read",
                auth_scope="analytics:read",
                cache_ttl_seconds=300,
                timeout_seconds=3,
                retries=1,
                fallback=lambda args, ctx, exc: {"insights": [], "fallback_reason": str(exc)},
                handler=lambda args, ctx: {"insights": olist_service.category_insights(str(args["query"]))},
            ),
            ToolSpec(
                name="generate_after_sales_priority_report",
                description="Generate a read-only after-sales operations decision report.",
                input_model=QueryArgs,
                allowed_roles=ops_roles,
                risk_level="read",
                auth_scope="after_sales:ops_report",
                cache_ttl_seconds=120,
                timeout_seconds=8,
                retries=1,
                handler=lambda args, ctx: {
                    "report": olist_service.after_sales_priority_report(str(args["query"])),
                },
            ),
            ToolSpec(
                name="search_policy_knowledge",
                description="Retrieve support policy sections from markdown KB.",
                input_model=PolicySearchArgs,
                allowed_roles=support_roles,
                risk_level="read",
                auth_scope="policy:read",
                cache_ttl_seconds=300,
                timeout_seconds=5,
                retries=1,
                fallback=lambda args, ctx, exc: {"sections": [], "fallback_reason": str(exc)},
                handler=lambda args, ctx: {
                    "sections": [
                        {
                            "source": hit.source,
                            "source_type": hit.source_type,
                            "section_title": hit.section_title,
                            "text": hit.text,
                            "score": hit.score,
                        }
                        for hit in knowledge_base.search(str(args["query"]), k=int(args["k"]))
                    ]
                },
            ),
            ToolSpec(
                name="search_support_examples",
                description="Retrieve similar public support conversations for answer style context.",
                input_model=SupportSearchArgs,
                allowed_roles=support_roles,
                risk_level="read",
                auth_scope="support_examples:read",
                cache_ttl_seconds=300,
                timeout_seconds=5,
                retries=1,
                fallback=lambda args, ctx, exc: {"docs": [], "fallback_reason": str(exc)},
                handler=lambda args, ctx: {
                    "docs": [
                        {
                            "doc_id": hit.doc.get("doc_id", ""),
                            "intent": hit.doc.get("intent", ""),
                            "capability": hit.doc.get("capability", ""),
                            "text": hit.doc.get("text", ""),
                            "score": hit.score,
                        }
                        for hit in support_retriever.hybrid_search(str(args["query"]), k=int(args["k"]))
                    ]
                },
            ),
            ToolSpec(
                name="prepare_side_effect",
                description="Draft an after-sales side-effect action before HITL confirmation.",
                input_model=OrderStatusArgs,
                allowed_roles=support_roles,
                risk_level="medium",
                auth_scope="after_sales:draft",
                cache_ttl_seconds=120,
                timeout_seconds=3,
                retries=1,
                handler=lambda args, ctx: {
                    "draft": olist_service.escalation_draft(str(args["order_id"])),
                },
            ),
            ToolSpec(
                name="assess_after_sales_case",
                description=(
                    "Assess an after-sales case using deterministic order facts, "
                    "policy references, verifier checks, and customer-reply draft."
                ),
                input_model=AfterSalesCaseArgs,
                allowed_roles=support_roles,
                risk_level="medium",
                auth_scope="after_sales:assess",
                timeout_seconds=8,
                retries=1,
                handler=lambda args, ctx: _assess_after_sales_case(
                    after_sales_engine,
                    olist_service,
                    args,
                ),
            ),
            ToolSpec(
                name="execute_side_effect",
                description="Execute a confirmed idempotent side-effect action.",
                input_model=SideEffectArgs,
                allowed_roles=frozenset({"after_sales_operator", "admin"}),
                side_effect=True,
                risk_level="critical",
                auth_scope="after_sales:write",
                idempotency_required=True,
                timeout_seconds=2,
                retries=1,
                handler=lambda args, ctx: {
                    "result": case_service.execute_action(
                        action_type=str(args["action_type"]),
                        order_id=str(args["order_id"]),
                        message_text=str(args["message_text"]),
                        case_id=str(args["case_id"]) if args.get("case_id") else None,
                    )
                },
            ),
        ],
        cache_backend=cache_backend,
        runtime_store=runtime_store,
    )


def _assess_after_sales_case(
    engine: AfterSalesDecisionEngine,
    olist_service: OlistService,
    args: dict[str, Any],
) -> dict[str, Any]:
    order = olist_service.get_order_status(str(args["order_id"]))
    if order is None:
        return {
            "found": False,
            "case": None,
            "decision": {
                "outcome": "ask_clarification",
                "reason_code": "order_not_found",
                "confidence": 0.0,
                "requires_human": False,
            },
            "verification": {"passed": True, "flags": [], "required_next_step": "clarify"},
            "customer_reply": "没有找到该订单，请确认完整订单号后再提交售后请求。",
        }
    case = engine.assess(
        action_type=str(args["action_type"]),
        order=order,
        user_request=str(args["user_request"]),
        policy_sections=list(args.get("policy_sections", [])),
    )
    return {
        "found": True,
        "case": case.model_dump(),
        "decision": case.decision.model_dump(),
        "verification": case.verification.model_dump(),
        "customer_reply": case.customer_reply,
    }


def _owned_order_rows(
    olist_service: OlistService,
    allowed_order_ids: list[str],
    filter_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order_id in allowed_order_ids:
        order = olist_service.get_order_status(order_id)
        if order is None:
            continue
        in_transit = order.status in {"approved", "invoiced", "processing", "shipped"}
        needs_attention = bool(
            (isinstance(order.delay_days, int) and order.delay_days > 0)
            or order.refund_status
            or order.address_change_status
        )
        if filter_name == "in_transit" and not in_transit:
            continue
        if filter_name == "attention" and not needs_attention:
            continue
        rows.append(
            {
                "order_id": order.order_id,
                "status": order.status,
                "category": order.category_summary,
                "estimated_delivery_date": order.estimated_delivery_date,
                "delay_days": order.delay_days,
                "payment_value": order.payment_value,
            }
        )
    return sorted(rows, key=lambda item: str(item["order_id"]))


def build_tool_cache_from_env() -> ToolCacheBackend:
    return build_runtime_store_from_env()


def build_runtime_store_from_env() -> RuntimeStore:
    backend = (
        os.environ.get("RUNTIME_STORE_BACKEND")
        or os.environ.get("TOOL_CACHE_BACKEND")
        or "memory"
    ).strip().lower()
    if backend in {"memory", "inmemory", "local", ""}:
        return InMemoryRuntimeStore()
    if backend == "redis":
        url = os.environ.get("REDIS_URL")
        if not url:
            raise RuntimeError("RUNTIME_STORE_BACKEND=redis requires REDIS_URL.")
        prefix = os.environ.get("REDIS_PREFIX") or os.environ.get("TOOL_CACHE_PREFIX", "olist-agent")
        return RedisRuntimeStore.from_url(url, prefix=prefix)
    raise RuntimeError(f"Unsupported RUNTIME_STORE_BACKEND: {backend}")


def _error_code(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, ToolCallError):
        return exc.error_code
    return "tool_execution_failed"


def _redact_args(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for key, value in arguments.items():
        if key == "order_id" and isinstance(value, str) and len(value) > 10:
            redacted[key] = f"{value[:6]}...{value[-4:]}"
        elif "message" in key or "text" in key:
            text = str(value)
            redacted[key] = {"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()[:12]}
        else:
            redacted[key] = value
    return redacted


def _result_summary(data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, list):
            summary[key] = f"list[{len(value)}]"
        elif isinstance(value, dict):
            summary[key] = f"dict[{len(value)}]"
        else:
            summary[key] = value
    return summary


def _result_count(data: dict[str, Any]) -> int:
    """Return a cardinality for dashboards without logging result payloads."""
    if not isinstance(data, dict):
        return 0
    for key in ("orders", "results", "cases", "rows", "hits", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    if "found" in data:
        return 1 if data.get("found") else 0
    return len(data)


def _order_status_payload(service: OlistService, order_id: str) -> dict[str, Any]:
    """Keep the legacy text and add customer-safe facts from the same read."""
    view = service.get_order_status(order_id)
    fields = ("order_id", "status", "purchase_timestamp", "estimated_delivery_date",
              "delivered_customer_date", "delay_days", "payment_value")
    return {"answer": format_order_status(view), "found": view is not None,
            "facts": {key: getattr(view, key, None) for key in fields} if view else {}}
