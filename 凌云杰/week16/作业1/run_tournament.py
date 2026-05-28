from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine import WerewolfGame
from src.evaluation import evaluate_games, review_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Werewolf Agent Team tournament.")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="logs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    for idx in range(args.games):
        game = WerewolfGame(seed=args.seed + idx)
        log = game.run()
        logs.append(log)
        (out_dir / f"game_{idx + 1:03d}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = logs[-1]
    summary = {
        "evaluation": evaluate_games(logs),
        "latest_review": review_game(latest),
        "games": [{"seed": log["seed"], "winner": log["winner"], "reason": log["reason"]} for log in logs],
    }
    (out_dir / "latest_game.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "tournament.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
