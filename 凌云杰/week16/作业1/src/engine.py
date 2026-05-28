from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .agents import GOOD_ROLES, ROLE_NAMES, WOLF_ROLES, Observation, build_agent


@dataclass
class Player:
    id: int
    role: str
    alive: bool = True

    @property
    def side(self) -> str:
        return "wolves" if self.role in WOLF_ROLES else "good"


class WerewolfGame:
    def __init__(self, seed: int = 1, roles: Optional[List[str]] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.roles = roles or ["werewolf", "werewolf", "seer", "witch", "hunter", "villager", "villager", "villager"]
        shuffled = list(self.roles)
        self.rng.shuffle(shuffled)
        self.players = {i + 1: Player(i + 1, role) for i, role in enumerate(shuffled)}
        self.agents = {pid: build_agent(pid, player.role, self.rng) for pid, player in self.players.items()}
        self.day = 0
        self.events: List[dict] = []
        self.private_events: Dict[int, List[dict]] = {pid: [] for pid in self.players}
        self.witch_antidote = True
        self.witch_poison = True
        self.winner: Optional[str] = None
        self.reason = ""
        self._init_private_info()

    def run(self, max_days: int = 8) -> dict:
        self._log("setup", "engine", text="角色分配完成，私有身份已下发。")
        while not self.winner and self.day < max_days:
            self.day += 1
            self._night()
            self._check_win()
            if self.winner:
                break
            self._day_discussion()
            self._vote_and_exile()
            self._check_win()
        if not self.winner:
            self.winner = "good"
            self.reason = "达到最大天数，好人阵营仍未被击穿。"
        self._log("game_over", "engine", winner=self.winner, reason=self.reason)
        return self.to_log()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_log(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_log(self) -> dict:
        return {
            "seed": self.seed,
            "roles": {pid: player.role for pid, player in self.players.items()},
            "winner": self.winner,
            "reason": self.reason,
            "events": self.events,
            "private_events": self.private_events,
            "final_alive": self.alive_ids(),
        }

    def alive_ids(self) -> List[int]:
        return [pid for pid, p in self.players.items() if p.alive]

    def _night(self) -> None:
        self._log("phase", "engine", phase="night", day=self.day, text=f"第 {self.day} 夜开始。")
        self._seer_action()
        victim = self._wolf_action()
        saved = self._witch_save_action(victim) if victim else False
        poisoned = self._witch_poison_action(victim if saved else None)

        deaths = []
        if victim and not saved:
            deaths.append(victim)
        if poisoned and poisoned not in deaths:
            deaths.append(poisoned)
        for pid in deaths:
            self._kill(pid, cause="night")
        self._log("night_result", "engine", victim=victim, saved=saved, poisoned=poisoned, deaths=deaths)
        self._hunter_revenge(deaths, phase="night")

    def _day_discussion(self) -> None:
        self._log("phase", "engine", phase="day", day=self.day, text=f"第 {self.day} 天发言开始。")
        for pid in self.alive_ids():
            agent = self.agents[pid]
            obs = self._observation(pid, "day")
            agent.observe(obs)
            text = agent.speak(obs)
            self._log("speech", pid, text=text)

    def _vote_and_exile(self) -> None:
        votes = {}
        for pid in self.alive_ids():
            agent = self.agents[pid]
            obs = self._observation(pid, "vote")
            agent.observe(obs)
            target = agent.vote(obs)
            if target not in self.alive_ids() or target == pid:
                choices = [p for p in self.alive_ids() if p != pid]
                target = self.rng.choice(choices) if choices else pid
            votes[pid] = target
            self._log("vote", pid, target=target)
        if not votes:
            return
        counter = Counter(votes.values())
        top_count = max(counter.values())
        tied = [pid for pid, count in counter.items() if count == top_count]
        exiled = self.rng.choice(tied)
        voters = [voter for voter, target in votes.items() if target == exiled]
        role = self.players[exiled].role
        self._kill(exiled, cause="exile")
        self._log("exile", "engine", target=exiled, role=role, voters=voters, text=f"{exiled} 号被放逐，身份为{ROLE_NAMES[role]}。")
        self._hunter_revenge([exiled], phase="day")

    def _seer_action(self) -> None:
        seers = [p for p in self.alive_ids() if self.players[p].role == "seer"]
        if not seers:
            return
        seer = seers[0]
        obs = self._observation(seer, "night")
        target = self.agents[seer].choose_seer_check(obs)
        if not target:
            return
        side = self.players[target].side
        event = {"type": "seer_result", "day": self.day, "actor": seer, "target": target, "side": side}
        self.private_events[seer].append(event)
        self._log("private_action", seer, action="seer_check", target=target, result=side)

    def _wolf_action(self) -> Optional[int]:
        wolves = [p for p in self.alive_ids() if self.players[p].role == "werewolf"]
        if not wolves:
            return None
        proposals = []
        for wolf in wolves:
            obs = self._observation(wolf, "night")
            proposals.append(self.agents[wolf].choose_night_kill(obs))
        proposals = [p for p in proposals if p in self.alive_ids() and self.players[p].role != "werewolf"]
        if not proposals:
            return None
        victim = Counter(proposals).most_common(1)[0][0]
        self._log("private_action", "wolves", action="wolf_kill", target=victim, actors=wolves)
        return victim

    def _witch_save_action(self, victim: int) -> bool:
        witches = [p for p in self.alive_ids() if self.players[p].role == "witch"]
        if not witches:
            return False
        witch = witches[0]
        obs = self._observation(witch, "night")
        save = self.agents[witch].choose_witch_save(obs, victim, self.witch_antidote)
        if save and self.witch_antidote:
            self.witch_antidote = False
            self._log("private_action", witch, action="witch_save", target=victim)
            return True
        return False

    def _witch_poison_action(self, night_victim: Optional[int]) -> Optional[int]:
        witches = [p for p in self.alive_ids() if self.players[p].role == "witch"]
        if not witches:
            return None
        witch = witches[0]
        obs = self._observation(witch, "night")
        target = self.agents[witch].choose_witch_poison(obs, self.witch_poison, night_victim)
        if target and target in self.alive_ids() and target != witch and self.witch_poison:
            self.witch_poison = False
            self._log("private_action", witch, action="witch_poison", target=target)
            return target
        return None

    def _hunter_revenge(self, deaths: List[int], phase: str) -> None:
        for pid in deaths:
            if self.players[pid].role != "hunter":
                continue
            obs = self._observation(pid, phase)
            target = self.agents[pid].choose_hunter_shot(obs)
            if target and self.players.get(target) and self.players[target].alive:
                self._kill(target, cause="hunter")
                self._log("hunter_shot", pid, target=target, text=f"猎人 {pid} 号带走 {target} 号。")

    def _kill(self, pid: int, cause: str) -> None:
        if self.players[pid].alive:
            self.players[pid].alive = False
            self._log("death", "engine", target=pid, role=self.players[pid].role, cause=cause)

    def _check_win(self) -> None:
        alive = [self.players[pid] for pid in self.alive_ids()]
        wolves = [p for p in alive if p.role == "werewolf"]
        good = [p for p in alive if p.role in GOOD_ROLES]
        if not wolves:
            self.winner = "good"
            self.reason = "所有狼人出局。"
        elif len(wolves) >= len(good):
            self.winner = "wolves"
            self.reason = "狼人数量不少于好人数量。"

    def _observation(self, pid: int, phase: str) -> Observation:
        role = self.players[pid].role
        wolves = [p for p in self.players if self.players[p].role == "werewolf"] if role == "werewolf" else []
        visible_private = list(self.private_events[pid])
        return Observation(
            day=self.day,
            phase=phase,
            player_id=pid,
            role=role,
            alive=self.alive_ids(),
            public_events=list(self.events),
            private_events=visible_private,
            known_wolves=wolves,
        )

    def _init_private_info(self) -> None:
        wolves = [pid for pid, player in self.players.items() if player.role == "werewolf"]
        for wolf in wolves:
            self.private_events[wolf].append({"type": "wolf_team", "wolves": wolves})

    def _log(self, event_type: str, actor, **payload) -> None:
        event = {"type": event_type, "day": self.day, "actor": actor, **payload}
        self.events.append(event)
