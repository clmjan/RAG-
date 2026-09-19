import json
import subprocess
import sys

from app.config import Settings
from app.db import Database
from app.evaluation import load_test_cases, run_ablation, run_evaluation


def test_evaluation_dataset_has_required_size_and_categories():
    cases = load_test_cases()
    assert len(cases) >= 50
    assert sum(not case.get("should_refuse") for case in cases) >= 30
    assert sum(bool(case.get("should_refuse")) for case in cases) >= 15
    assert sum(bool(case.get("hard_negative")) for case in cases) >= 8
    assert sum(any("\u4e00" <= char <= "\u9fff" for char in case["question"]) for case in cases) >= 10
    assert sum(any(char.isascii() and char.isalpha() for char in case["question"]) for case in cases) >= 10
    assert sum(bool(case.get("multi_source")) for case in cases) >= 4
    assert sum(bool(case.get("cross_chapter")) for case in cases) >= 3
    assert {case.get("split") for case in cases} == {"validation", "test"}


def test_evaluation_saves_reproducible_config_snapshot(tmp_path):
    database = Database(tmp_path / "evaluation.db")
    database.init()
    result = run_evaluation(database, test_cases=[
        {"id": "a", "split": "validation", "question": "unknown", "source": [], "should_refuse": True},
        {"id": "b", "split": "test", "question": "unknown", "source": [], "should_refuse": True},
    ], config=Settings(chunk_size=123, chunk_overlap=12, hybrid_lexical_weight=0.4,
                       hybrid_semantic_weight=0.6))
    assert result["config"]["chunk_size"] == 123
    assert result["config"]["chunk_overlap"] == 12
    assert result["config"]["hybrid_lexical_weight"] == 0.4
    saved = database.list_evaluation_runs()[0]
    assert saved["config"] == result["config"]
    assert "embedding_api_key" not in json.dumps(saved["config"])


def test_evaluate_cli_runs_real_method(tmp_path):
    # Use a temporary empty database: the CLI still executes the selected method and emits real zeros.
    env = dict(__import__("os").environ)
    env["DATABASE_PATH"] = str(tmp_path / "cli.db")
    completed = subprocess.run(
        [sys.executable, "-m", "app.evaluate", "--method", "bm25"],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        env=env, capture_output=True, text=True, check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["retrieval_method"] == "bm25"
    assert "recall_at_1" in payload["metrics"]
    assert payload["config"]["retrieval_method"] == "bm25"


def test_ablation_persists_each_real_variant(tmp_path):
    database = Database(tmp_path / "ablation.db")
    database.init()
    runs = run_ablation(database, "hybrid", config=Settings())
    assert len(runs) >= 10
    assert len(database.list_evaluation_runs(limit=100)) == len(runs)
    assert {run["ablation"] for run in runs} >= {
        "baseline", "chunk_size_350", "chunk_overlap_0", "hybrid_0.2_0.8",
    }
    assert all("evaluation_dataset_sha256" in run["config"] for run in runs)
