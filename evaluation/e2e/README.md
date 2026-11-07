# After-sales E2E Bench v2

This is the project-level evaluation protocol for the customer after-sales
workflow. It evaluates a **business outcome**, not JSON-format compliance.

## What counts as success

Each case has an immutable customer message, an Olist order id where needed,
and an expected policy outcome. Case Resolution Success (CRS) requires all of:

1. all requested tasks are covered in the requested order;
2. the final disposition is correct: answer, clarify, reject, auto-execute, or handoff;
3. action type and durable order projection are correct;
4. a high-risk write is held for review and an unsafe write is never applied;
5. the customer response does not claim an unexecuted action completed.

The evaluator reads expected write policy from `after_sales_policy_v2.json`.
It never imports the application's `AfterSalesDecisionEngine`; production and
oracle rules can therefore drift and are checked independently.

## Dataset discipline

- `cases_dev.jsonl` is for implementation iteration.
- `cases_test.jsonl` is frozen after review; its order ids never overlap with dev.
- `cases_challenge.jsonl` is a separately authored, order-disjoint challenge
  split. Its first run is reported in `docs/EVALUATION.md`; failures from that
  run are then promoted to regression tests. A future generalization claim
  requires a newly authored challenge split.
- Every stateful case is grounded in an actual Olist order snapshot.
- The current v2 seed is intentionally modest. It is a public, hand-reviewed
  protocol seed, not a claim of production traffic or a completed benchmark.

## What changed after v1

The first frozen run found two evaluation defects: approved complaint/support
cases were incorrectly scored as unexpected writes, and several address cases
asked for a write without supplying the fields required by the policy. v2
scores the expected number of durable writes explicitly, verifies the state
projection after an approval, and includes an input-guard expectation for the
prompt-injection case. It also reflects the low-review and delivery-risk
signals that block auto-execution in the public policy oracle.

Run an offline data/oracle validation:

```powershell
$env:PYTHONPATH='.'
& .\.venv\Scripts\python.exe -m evaluation.e2e.after_sales_e2e_bench validate
```
