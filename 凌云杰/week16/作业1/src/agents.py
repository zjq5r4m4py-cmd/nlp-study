from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


GOOD_ROLES = {"seer", "witch", "hunter", "villager"}
WOLF_ROLES = {"werewolf"}


ROLE_NAMES = {
    "werewolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}


@dataclass
class Memory:
    checked: Dict[int, str] = field(default_factory=dict)
    suspicion: Dict[int, float] = field(default_factory=dict)
    trusted: Set[int] = field(default_factory=set)
    private_notes: List[str] = field(default_factory=list)


@dataclass
class Observation:
    day: int
    phase: str
    player_id: int
    role: str
    alive: List[int]
    public_events: List[dict]
    private_events: List[dict]
    known_wolves: List[int] = field(default_factory=list)


class BaseAgent:
    """规则型 Agent 基类。可替换为 LLM Agent，但接口保持稳定。"""

    def __init__(self, player_id: int, role: str, rng: random.Random):
        self.player_id = player_id
        self.role = role
        self.rng = rng
        self.memory = Memory()

    @property
    def side(self) -> str:
        return "wolves" if self.role in WOLF_ROLES else "good"

    def observe(self, obs: Observation) -> None:
        for event in obs.private_events:
            if event["type"] == "seer_result":
                target = event["target"]
                side = event["side"]
                self.memory.checked[target] = side
                if side == "wolves":
                    self.memory.suspicion[target] = self.memory.suspicion.get(target, 0) + 4
                else:
                    self.memory.trusted.add(target)
            if event["type"] == "wolf_team":
                self.memory.private_notes.append(f"狼队友: {event['wolves']}")

        for event in obs.public_events[-12:]:
            if event["type"] == "vote":
                voter, target = event["actor"], event["target"]
                if target == self.player_id and voter != self.player_id:
                    self.memory.suspicion[voter] = self.memory.suspicion.get(voter, 0) + 0.3
            if event["type"] == "exile" and event.get("role") == "werewolf":
                voters = event.get("voters", [])
                for voter in voters:
                    if voter != self.player_id:
                        self.memory.trusted.add(voter)

    def speak(self, obs: Observation) -> str:
        if self.role == "werewolf":
            target = self._least_trusted(obs.alive, exclude=obs.known_wolves + [self.player_id])
            return f"我倾向于怀疑 {target} 号，白天发言比较保守，可能在隐藏视角。"
        if self.role == "seer" and self.memory.checked:
            target, side = next(reversed(self.memory.checked.items()))
            label = "狼人阵营" if side == "wolves" else "好人阵营"
            return f"我是预言家视角，昨晚验到 {target} 号是{label}，建议大家围绕这个信息投票。"
        if self.role == "witch":
            return "我会重点看投票一致性和夜晚死亡信息，今天先听预言家和被怀疑位的解释。"
        if self.role == "hunter":
            return "我会站边信息更完整的人，如果被强推，我会优先带走最高嫌疑位。"
        target = self._most_suspicious(obs.alive)
        return f"目前我更关注 {target} 号的行为，建议大家把发言和投票逻辑对齐。"

    def vote(self, obs: Observation) -> int:
        candidates = [p for p in obs.alive if p != self.player_id]
        if not candidates:
            return self.player_id
        if self.role == "werewolf":
            good_candidates = [p for p in candidates if p not in obs.known_wolves]
            return self._least_trusted(good_candidates or candidates)
        return self._most_suspicious(candidates)

    def choose_night_kill(self, obs: Observation) -> Optional[int]:
        candidates = [p for p in obs.alive if p not in obs.known_wolves]
        if not candidates:
            return None
        priority_roles = self._infer_public_power_roles(obs)
        for target in priority_roles:
            if target in candidates:
                return target
        return self._least_trusted(candidates)

    def choose_seer_check(self, obs: Observation) -> Optional[int]:
        candidates = [p for p in obs.alive if p != self.player_id and p not in self.memory.checked]
        if not candidates:
            return None
        return self._most_suspicious(candidates)

    def choose_witch_save(self, obs: Observation, victim: int, has_antidote: bool) -> bool:
        if not has_antidote:
            return False
        if victim == self.player_id:
            return True
        return self.rng.random() < 0.25

    def choose_witch_poison(self, obs: Observation, has_poison: bool, night_victim: Optional[int]) -> Optional[int]:
        if not has_poison:
            return None
        candidates = [p for p in obs.alive if p != self.player_id and p != night_victim]
        if not candidates:
            return None
        target = self._most_suspicious(candidates)
        if self.memory.suspicion.get(target, 0) >= 3.5 or self.rng.random() < 0.12:
            return target
        return None

    def choose_hunter_shot(self, obs: Observation) -> Optional[int]:
        candidates = [p for p in obs.alive if p != self.player_id]
        return self._most_suspicious(candidates) if candidates else None

    def _most_suspicious(self, candidates: Sequence[int]) -> int:
        return max(candidates, key=lambda p: (self.memory.suspicion.get(p, 0), self.rng.random()))

    def _least_trusted(self, candidates: Sequence[int], exclude: Optional[Iterable[int]] = None) -> int:
        excluded = set(exclude or [])
        pool = [p for p in candidates if p not in excluded]
        if not pool:
            pool = list(candidates)
        return min(pool, key=lambda p: (p in self.memory.trusted, self.rng.random()))

    def _infer_public_power_roles(self, obs: Observation) -> List[int]:
        targets = []
        for event in obs.public_events:
            if event["type"] == "speech" and ("预言家" in event["text"] or "验到" in event["text"]):
                targets.append(event["actor"])
        return targets


class WerewolfAgent(BaseAgent):
    pass


class SeerAgent(BaseAgent):
    pass


class WitchAgent(BaseAgent):
    pass


class HunterAgent(BaseAgent):
    pass


class VillagerAgent(BaseAgent):
    pass


def build_agent(player_id: int, role: str, rng: random.Random) -> BaseAgent:
    agent_cls = {
        "werewolf": WerewolfAgent,
        "seer": SeerAgent,
        "witch": WitchAgent,
        "hunter": HunterAgent,
        "villager": VillagerAgent,
    }[role]
    return agent_cls(player_id, role, rng)
