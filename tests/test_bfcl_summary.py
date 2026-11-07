import json

from scripts.summarize_bfcl_results import summarize_score_root


def test_bfcl_summary_reads_category_json(tmp_path) -> None:
    score_dir = tmp_path / "score"
    model_dir = score_dir / "demo-model"
    model_dir.mkdir(parents=True)
    (model_dir / "BFCL_v3_simple_python_score.json").write_text(
        json.dumps(
            {
                "test_category": "simple_python",
                "accuracy": 0.75,
                "total": 4,
                "correct": 3,
            }
        ),
        encoding="utf-8",
    )

    result = summarize_score_root(score_dir)

    assert result["json_score_count"] == 1
    assert result["category_scores"][0]["category"] == "simple_python"
    assert result["category_scores"][0]["accuracy"] == 0.75


def test_bfcl_summary_reads_leaderboard_csv(tmp_path) -> None:
    score_dir = tmp_path / "score"
    score_dir.mkdir()
    (score_dir / "data_overall.csv").write_text(
        "Model,Overall Acc,Cost ($),P95\nDemoModel,80.5,0.12,3.4\nOther,1,2,3\n",
        encoding="utf-8",
    )

    result = summarize_score_root(score_dir, "Demo")

    assert result["csv_row_count"] == 1
    assert result["leaderboard_rows"][0]["model"] == "DemoModel"
    assert result["leaderboard_rows"][0]["overall_acc"] == "80.5"
