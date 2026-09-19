"""Command-line offline evaluation runner.

Examples:
    python -m app.evaluate --method bm25
    python -m app.evaluate --method hybrid
    python -m app.evaluate --compare
"""

import argparse
import json

from .config import settings
from .db import Database
from .evaluation import run_ablation, run_evaluation


METHODS = ("tfidf", "bm25", "embedding", "hybrid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real offline RAG evaluations")
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--compare", action="store_true", help="run all retrieval methods")
    parser.add_argument("--ablation", action="store_true", help="run real configuration ablations")
    args = parser.parse_args(argv)
    if not args.compare and not args.method:
        parser.error("provide --method or --compare")

    database = Database(settings.database_path)
    database.init()
    methods = METHODS if args.compare else (args.method,)
    output = []
    for method in methods:
        if args.ablation:
            output.extend(run_ablation(database, method, config=settings))
        else:
            output.append(run_evaluation(database, method=method, config=settings))
    print(json.dumps(output[0] if len(output) == 1 else {"runs": output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
