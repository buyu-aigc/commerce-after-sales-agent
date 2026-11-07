"""Confirmation-scope guard for tau2 retail write actions.

The guard catches a common customer-service failure mode: after the agent asks a
multi-item confirmation question, the simulated user may answer with a broad
"yes/both/all". For irreversible write tools, that is not enough evidence that
the customer intentionally confirmed every item.
"""

from __future__ import annotations

import re

CONFIRMATION_SCOPE_CLARIFICATION = (
    "Before I change the order, please list the exact item names you want me to "
    "process now, for example: 'desk lamp only' or 'water bottle and desk lamp'."
)

_BROAD_SCOPE_RE = re.compile(
    r"\b(yes|yeah|yep|sure|ok|okay|confirmed?|approve|approved|proceed|go ahead|"
    r"both|all|everything)\b",
    re.IGNORECASE,
)
_EXPLICIT_NARROWING_RE = re.compile(
    r"\b(only|just|except|not|instead|rather|desk lamp|water bottle|shirt|jeans|"
    r"jacket|sweater|shoes|bag|watch|speaker|headphones|charger)\b",
    re.IGNORECASE,
)
_MULTI_ITEM_WRITE_CONTEXT_RE = re.compile(
    r"\b(confirm|confirmation|proceed|approve).{0,240}\b(exchange|return|refund|"
    r"cancel|modify|change)\b|\b(exchange|return|refund|cancel|modify|change).{0,240}"
    r"\b(confirm|confirmation|proceed|approve)\b",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_ITEM_SIGNAL_RE = re.compile(
    r"\b(both|all|multiple|items|item names|list exactly|which item|which items)\b",
    re.IGNORECASE,
)


def is_broad_confirmation(text: str | None) -> bool:
    """Return True when the user confirmation does not specify item scope."""

    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return False
    if _EXPLICIT_NARROWING_RE.search(normalized):
        return False
    return bool(_BROAD_SCOPE_RE.search(normalized))


def needs_item_scope_clarification(
    user_text: str | None,
    previous_assistant_text: str | None,
) -> bool:
    """Decide whether a broad confirmation must be narrowed before a write.

    This is intentionally conservative: it only fires when the previous assistant
    turn was about confirming a risky write action and explicitly signaled a
    multi-item scope.
    """

    if not is_broad_confirmation(user_text):
        return False

    previous = previous_assistant_text or ""
    if not _MULTI_ITEM_WRITE_CONTEXT_RE.search(previous):
        return False
    return bool(_MULTI_ITEM_SIGNAL_RE.search(previous))
