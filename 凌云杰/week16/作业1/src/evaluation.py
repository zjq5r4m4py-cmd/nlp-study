from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List


def evaluate_games(game_logs: Iterable[dict]) -> dict:
    logs = list(game_logs)
    winners = Counter(log["winner"] for log in logs)
    role_survival = defaultdict(lambda: {"alive": 0, "total": 0})
    avg_days = 0
    exile_hits = 0
    exiles = 0

    for log in logs:
        days = max((event.get("day", 0) for event in log["events"]), default=0)
        avg_days += days
        alive = set(log["final_alive"])
        for pid, role in log["roles"].items():
            role_survival[role]["total"] += 1
            if int(pid) in alive:
                role_survival[role]["alive"] += 1
        for event in log["events"]:
            if event["type"] == "exile":
                exiles += 1
                if event.get("role") == "werewolf":
                    exile_hits += 1

    total = max(len(logs), 1)
    return {
        "games": len(logs),
        "win_rate": {side: count / total for side, count in winners.items()},
        "avg_days": avg_days / total,
        "wolf_exile_precision": exile_hits / exiles if exiles else 0,
        "role_survival": {
            role: item["alive"] / item["total"] if item["total"] else 0
            for role, item in sorted(role_survival.items())
        },
    }


def review_game(log: dict) -> List[str]:
    notes = []
    seer_checks = [e for e in log["events"] if e["type"] == "private_action" and e.get("action") == "seer_check"]
    wolf_kills = [e for e in log["events"] if e["type"] == "private_action" and e.get("action") == "wolf_kill"]
    exiles = [e for e in log["events"] if e["type"] == "exile"]

    if seer_checks:
        last = seer_checks[-1]
        notes.append(f"预言家共验人 {len(seer_checks)} 次，最后一次查验 {last['target']} 号。")
    else:
        notes.append("预言家较早出局或未完成有效查验，好人信息源不足。")

    wolf_power_hits = 0
    power_roles = {"seer", "witch", "hunter"}
    roles: Dict[str, str] = {str(k): v for k, v in log["roles"].items()}
    for event in wolf_kills:
        target = str(event.get("target"))
        if roles.get(target) in power_roles:
            wolf_power_hits += 1
    notes.append(f"狼人夜晚刀中神职 {wolf_power_hits}/{len(wolf_kills)} 次，体现了隐藏身份识别能力。")

    wolf_exiles = sum(1 for event in exiles if event.get("role") == "werewolf")
    notes.append(f"白天放逐狼人 {wolf_exiles}/{len(exiles)} 次，反映公共讨论与投票质量。")
    notes.append(f"最终胜方：{log['winner']}，原因：{log['reason']}")
    return notes
