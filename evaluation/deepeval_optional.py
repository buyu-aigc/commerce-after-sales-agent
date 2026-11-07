from __future__ import annotations

import importlib.util


def main() -> None:
    if importlib.util.find_spec("deepeval") is None:
        print(
            "DeepEval is not installed. Export cases with `python -m evaluation.deepeval_export`, "
            "then install the optional eval extra when you want model-judged metrics."
        )
        return

    print(
        "DeepEval is available. Use data/olist_derived/deepeval_rag_cases.jsonl as the test source "
        "for contextual precision/recall, faithfulness, tool correctness, and agent trajectory metrics."
    )


if __name__ == "__main__":
    main()
