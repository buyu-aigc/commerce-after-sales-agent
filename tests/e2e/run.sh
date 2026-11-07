#!/usr/bin/env bash
# Run the opt-in Playwright P0-1 suite against a disposable V2 instance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ENV="${AIPM_RUNTIME_ENV:-$ROOT/runtime-env.sh}"
if [[ -f "$RUNTIME_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
fi

export RUN_E2E=1
export P0_E2E_REVIEW_TOKEN="${P0_E2E_REVIEW_TOKEN:-e2e-review-token}"
export P0_E2E_ARTIFACT_DIR="${P0_E2E_ARTIFACT_DIR:-$ROOT/reports/p0-e2e}"

mkdir -p "$P0_E2E_ARTIFACT_DIR"

cd "$ROOT"
python -m pytest tests/e2e -q --tb=short "$@"
