# Demo After-Sales Policy Pack

This pack is for the learning/demo V2 product. It is **not** an official marketplace or legal policy.

## What it contains

- `demo_preship_cancel_auto_v1`: pre-shipment, low-value cancel → auto.
- `demo_delay_refund_auto_v1`: delivered, delay ≥10 days, amount ≤ R$100, no prior compensation/repeat claim, risk not high → auto.
- `demo_delay_refund_review_v1`: delayed refund outside the auto limits → human review.
- `demo_repeat_claim_review_v1`: repeat claim → human review.
- `demo_high_risk_user_review_v1`: high-risk user → human review.
- `demo_refund_failure_review_v1`: refund failure/abnormal → human review.
- `demo_default_review_v1`: no matching policy → human review.

## Versioning and effective windows

- `policies/demo_after_sales.json` is the active pack (`version: 1.1.0`).
- `policies/archive/demo_after_sales_v1.0.0.json` is the frozen previous version used by
  the replay/rollback drill.
- `effective_from` / `effective_to` are optional per policy and inherited from the pack.
  The window is **right-open**: a policy is inactive at `effective_to` exactly.
- Any new pack must keep the previous version in `policies/archive/`, add a golden
  case for a new auto rule, and pass the 600-row replay gate.

## Validation, replay, and rollback

```bash
# Schema + effective-window + conflict validation
python scripts/validate_policy_pack.py

# 600-row versioned replay (archive vs current, fixed as_of)
python -m evaluation.policy_versioned_replay \
  --output reports/policy_versioned_replay_report.json

# Auto-approval spot check + rollback / negative drills
python -m evaluation.policy_hardening_drill \
  --output reports/policy_hardening_drill_report.json
```

A pack that fails schema validation, contains a same-priority conflict, auto-approves
a non-control stratum, or changes decisions on rollback must not be deployed.

## Runtime guardrails

`PolicyEngine` applies these guards even if a pack is malformed:

- `max_amount <= 0` or `amount > max_amount` → downgrade to review;
- `repeat_claim`, `refund_failure`, `prior_compensation` true → review;
- `risk_level == "high"` → review;
- no matching policy or no active policy → the default review policy.

Every decision records `policy_version`, `pack_hash`, `evaluated_at`, and the effective
window for audit/replay.

## How it is used

- `PolicyEngine` loads this JSON and returns `auto` or `review`, `policy_id`, `max_amount`, reasons.
- The customer Agent uses it for policy-backed decisions.
- Low-risk auto decisions can execute within the configured limit; high-value/repeat cases go to the existing HITL workflow.
- Every Case keeps the policy decision in its draft payload for audit.

## Important limits

- The amount thresholds are demo assumptions, not business policy.
- No official legal/compliance review.
- Replace this pack with tenant-owned rules before any real deployment.
- Do not expose policy ids or internal reasons to customers; use customer-safe copy.
- The 600-row fixture set uses real OList order rows but synthetic risk/repeat/refund-failure
  labels; it is not 600 real customers or reviewers.

## Inspiration (not copied rules)

Public marketplace help pages and consumer-law summaries were used only to understand common patterns such as cancellation windows, return/refund evidence, and review escalation. The concrete thresholds here are product choices for this demo.
