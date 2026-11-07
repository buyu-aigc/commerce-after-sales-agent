#!/usr/bin/env python3
"""τ-bench retail adapter for this project.

Run this file inside a separate tau2-bench checkout, because tau2 requires
Python 3.12+ while the main project intentionally stays on Python 3.11.

Example:
    cd /path/to/tau2-bench
    uv sync
    OPENAI_API_KEY=... uv run python /path/to/this/file --num-tasks 5
"""

from __future__ import annotations

import argparse

from pydantic import BaseModel, Field
from tau2_confirmation_guard import (
    CONFIRMATION_SCOPE_CLARIFICATION,
    needs_item_scope_clarification,
)
from tau2_variant_guard import repair_variant_selection

try:
    from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
    from tau2.data_model.message import (
        APICompatibleMessage,
        AssistantMessage,
        Message,
        MultiToolMessage,
        SystemMessage,
    )
    from tau2.data_model.simulation import TextRunConfig
    from tau2.environment.toolkit import Tool
    from tau2.registry import registry
    from tau2.runner import run_domain
    from tau2.utils.llm_utils import generate
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tau2 env
    raise SystemExit(
        "This adapter must be executed inside a tau2-bench environment. "
        "See README.md for setup steps."
    ) from exc


SYSTEM_PROMPT = """\
You are a production-grade retail customer service agent evaluated by tau-bench.

Follow the domain policy exactly.
Use tools for factual customer/order/product information.
Never invent order, user, product, item, payment, address, or refund facts.

Execution policy:
- Decompose the user's request into all required read-only and write tasks.
- Prefer read-only tools before any write tool.
- For write tools, explain the exact planned mutation and wait for explicit
  user confirmation unless the domain policy says no confirmation is needed.
- For multi-item write tools, confirm each item separately in the confirmation
  message. Do not ask "confirm both/all"; ask the customer to list exactly which
  item names they want to proceed with from the proposed list. Execute only the
  item ids that the customer explicitly confirms after that message. If the
  confirmation changes or narrows the scope, use the latest confirmed scope.
- A single retail order may become non-actionable after a write tool changes
  its status. If the customer asks for multiple write actions on the same order
  (for example return plus exchange), do not execute them sequentially. Before
  the first write, explain the conflict, compare the user's stated priority or
  savings preference, and confirm one chosen write action.
- Never exchange an item to itself. The new item id must differ from the
  original item id. If the customer asks for the exact same variant, explain
  that the exchange tool requires a different available variant and offer the
  closest policy-compliant alternative.
- For address changes, after collecting the full new address and identifying
  eligible pending orders, ask for explicit confirmation for the default address
  and each eligible order address. Once confirmed, execute those address updates
  before handling any later reversal request.
- If the customer asks to change their default address and all possible order
  addresses, first update the default address, then inspect all orders, then
  update every eligible pending order address. If the customer later regrets the
  default-address change, only revert the default user address unless they also
  explicitly ask to revert order addresses.
- For footwear variants, preserve the original shoe size by default. Only change
  shoe size when the customer explicitly asks for a different size; otherwise
  optimize price or other preferences within the same size.
- Preserve the customer's original fallback preferences when recommending
  variants. Example: if the customer says they want a clicky full-size RGB
  keyboard, but if that exact variant is unavailable they will take no
  backlight, then choose an available clicky full-size no-backlight keyboard
  rather than inventing a white-backlight compromise.
- When a user references an item in another pending order as the desired variant,
  treat that order only as a lookup source. Do not modify, cancel, or otherwise
  mutate the referenced pending order unless the user explicitly asks to change
  that order too.
- When comparing multiple refund/exchange/cancellation options, always state
  each individual amount and the combined total for any grouped option, using
  the exact arithmetic expression when possible.
- If a required slot is missing or ambiguous, ask a concise clarification.
- If a tool call fails, repair the parameter once when the error is recoverable;
  otherwise explain the limitation and follow the policy fallback.
- Transfer to a human only when policy or missing information prevents safe
  resolution.

This prompt intentionally mirrors the engineering contract of the local Olist
Agent project: LLM handles planning and language, tau2 tools handle facts and
side effects, and write actions require a confirmation boundary.

<domain_policy>
{domain_policy}
</domain_policy>
"""


class RetailBenchmarkState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage] = Field(default_factory=list)
    turns: int = 0


class PolicyAwareRetailAgent(HalfDuplexAgent[RetailBenchmarkState]):
    """A tau2 HalfDuplexAgent with policy instructions and optional guards."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        enable_confirmation_guard: bool = True,
        enable_variant_repair: bool = True,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self.enable_confirmation_guard = enable_confirmation_guard
        self.enable_variant_repair = enable_variant_repair

    def get_init_state(
        self,
        message_history: list[Message] | None = None,
    ) -> RetailBenchmarkState:
        system_prompt = SYSTEM_PROMPT.format(domain_policy=self.domain_policy)
        return RetailBenchmarkState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    def _last_assistant_text(self, state: RetailBenchmarkState) -> str:
        for historic_message in reversed(state.messages):
            if getattr(historic_message, "role", None) == "assistant":
                return str(getattr(historic_message, "content", "") or "")
        return ""

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: RetailBenchmarkState,
    ) -> tuple[AssistantMessage, RetailBenchmarkState]:
        state.turns += 1
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        if (
            self.enable_confirmation_guard
            and not isinstance(message, MultiToolMessage)
            and needs_item_scope_clarification(
                str(getattr(message, "content", "") or ""),
                self._last_assistant_text(state),
            )
        ):
            response = AssistantMessage.text(CONFIRMATION_SCOPE_CLARIFICATION)
            state.messages.append(response)
            return response, state

        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            call_name="olist_policy_aware_retail_agent",
            **self.llm_args,
        )
        if self.enable_variant_repair:
            repair_variant_selection(response, state.messages)
        state.messages.append(response)
        return response, state


def create_policy_aware_retail_agent(tools, domain_policy, **kwargs):
    return PolicyAwareRetailAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args") or {"temperature": 0},
    )


def create_prompt_only_retail_agent(tools, domain_policy, **kwargs):
    """Ablation: retain the policy prompt but remove deterministic guards."""
    return PolicyAwareRetailAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args") or {"temperature": 0},
        enable_confirmation_guard=False,
        enable_variant_repair=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tau2 retail variants with this adapter.")
    parser.add_argument(
        "--variant",
        choices=("baseline", "prompt_only", "full"),
        default="full",
        help=(
            "baseline uses tau2's stock llm_agent; prompt_only uses this "
            "project's prompt without guards; full enables prompt and guards."
        ),
    )
    parser.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--judge-llm",
        default=None,
        help="LLM for tau2 natural-language assertion review; defaults to --user-llm.",
    )
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--save-to", default="olist_agent_tau2_retail_subset")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    judge_llm = args.judge_llm or args.user_llm

    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge_llm

    agent_name = "llm_agent"
    if args.variant == "prompt_only":
        agent_name = "olist_prompt_only_retail"
        registry.register_agent_factory(create_prompt_only_retail_agent, agent_name)
    elif args.variant == "full":
        agent_name = "olist_policy_aware_retail"
        registry.register_agent_factory(create_policy_aware_retail_agent, agent_name)
    results = run_domain(
        TextRunConfig(
            domain="retail",
            agent=agent_name,
            llm_agent=args.agent_llm,
            llm_user=args.user_llm,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            seed=args.seed,
            save_to=args.save_to,
            max_concurrency=args.max_concurrency,
            task_ids=args.task_id or None,
            task_split_name=args.task_split_name,
            timeout=args.timeout,
            auto_resume=args.auto_resume,
            review_model=judge_llm,
            hallucination_retries=0,
        )
    )
    rewards = [run.reward_info.reward for run in results.simulations if run.reward_info]
    if rewards:
        print(
            f"tau2 retail {args.variant} pass@1: "
            f"{sum(rewards) / len(rewards):.2%} ({sum(rewards)}/{len(rewards)})"
        )


if __name__ == "__main__":
    main()
