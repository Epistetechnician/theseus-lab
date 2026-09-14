"""Small in-memory terminal prototype for the Theseus identity question."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable


BEHAVIOR_MIN = 0.95
MECHANISM_MIN = 0.80


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


@dataclass(frozen=True)
class State:
    step: int
    components: dict[str, float]
    behavior: tuple[float, ...]

    @property
    def state_digest(self) -> str:
        return digest({
            "step": self.step,
            "components": self.components,
            "behavior": self.behavior,
        })


@dataclass(frozen=True)
class Transition:
    before: str
    after: str
    lineage: tuple[tuple[str, str, float], ...]


@dataclass
class Lab:
    initial: State = field(default_factory=lambda: State(0, {"safe": 1.0, "task": 1.0}, (1.0, 0.8)))
    states: list[State] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.states:
            self.states.append(self.initial)

    @property
    def current(self) -> State:
        return self.states[-1]

    def reset(self) -> None:
        self.states[:] = [self.initial]
        self.transitions.clear()
        self.emit("reset")

    def transform(self, components: dict[str, float], lineage: Iterable[tuple[str, str, float]], action: str) -> None:
        previous = self.current
        next_state = State(previous.step + 1, dict(sorted(components.items())), previous.behavior)
        edge_list = tuple(lineage)
        self.states.append(next_state)
        self.transitions.append(Transition(previous.state_digest, next_state.state_digest, edge_list))
        self.emit(action)

    def preserve(self, name: str) -> None:
        current = self.current
        if name not in current.components:
            raise ValueError(f"unknown component: {name}")
        components = dict(current.components)
        components[name] *= 0.98
        lineage = tuple((component, component, 1.0) for component in current.components)
        self.transform(components, lineage, f"preserve {name}")

    def replace(self, old: str, new: str) -> None:
        current = self.current
        if old not in current.components:
            raise ValueError(f"unknown component: {old}")
        components = dict(current.components)
        del components[old]
        components[new] = 1.0
        self.transform(components, (), f"replace {old} -> {new}")

    def split(self, old: str, first: str, second: str) -> None:
        current = self.current
        if old not in current.components:
            raise ValueError(f"unknown component: {old}")
        components = dict(current.components)
        del components[old]
        components[first] = 0.5
        components[second] = 0.5
        self.transform(components, ((old, first, 0.5), (old, second, 0.5)), f"split {old} -> {first},{second}")

    def report(self) -> dict[str, object]:
        if len(self.states) < 2:
            return {"status": "NO_TRANSITION", "state": self.current.state_digest}
        before, after = self.states[-2:]
        transition = self.transitions[-1]
        source_mass = {name: 0.0 for name in before.components}
        target_mass = {name: 0.0 for name in after.components}
        for source, target, mass in transition.lineage:
            if source in source_mass and target in target_mass:
                source_mass[source] += mass
                target_mass[target] += mass
        source_preservation = sum(source_mass.values()) / len(source_mass)
        target_coverage = sum(target_mass.values()) / len(target_mass)
        mechanism = (source_preservation + target_coverage) / 2.0
        behavior = self._behavior_similarity(before.behavior, after.behavior)
        drift = behavior >= BEHAVIOR_MIN and mechanism < MECHANISM_MIN
        return {
            "behavior_continuity": round(behavior, 4),
            "mechanism_continuity": round(mechanism, 4),
            "source_preservation": round(source_preservation, 4),
            "target_coverage": round(target_coverage, 4),
            "behaviorally_equivalent_mechanistic_drift": drift,
            "transition_digest": digest({
                "before": transition.before,
                "after": transition.after,
                "lineage": transition.lineage,
            }),
        }

    @staticmethod
    def _behavior_similarity(before: tuple[float, ...], after: tuple[float, ...]) -> float:
        if len(before) != len(after):
            raise ValueError("behavior dimensions differ")
        differences = []
        for left, right in zip(before, after):
            denominator = abs(left) + abs(right)
            differences.append(0.0 if denominator == 0.0 else abs(left - right) / denominator)
        return 1.0 - sum(differences) / len(differences)

    def emit(self, action: str) -> None:
        print(json.dumps({
            "action": action,
            "state": {
                "step": self.current.step,
                "components": self.current.components,
                "behavior": self.current.behavior,
                "state_digest": self.current.state_digest,
            },
            "report": self.report(),
        }, sort_keys=True))


def run_demo() -> None:
    scenarios = (
        ("preserved", lambda lab: lab.preserve("safe")),
        ("split", lambda lab: lab.split("safe", "safe-a", "safe-b")),
        ("drift", lambda lab: lab.replace("safe", "replacement")),
    )
    for name, action in scenarios:
        print(json.dumps({"scenario": name}, sort_keys=True))
        lab = Lab()
        lab.emit("initial")
        action(lab)
        print(json.dumps({"scenario": name, "verdict": lab.report()}, sort_keys=True))


def interactive() -> None:
    lab = Lab()
    lab.emit("initial")
    while True:
        try:
            command = input("theseus> ").strip().split()
        except EOFError:
            print()
            return
        if not command:
            continue
        try:
            if command[0] == "show":
                lab.emit("show")
            elif command[0] == "preserve" and len(command) == 2:
                lab.preserve(command[1])
            elif command[0] == "replace" and len(command) == 3:
                lab.replace(command[1], command[2])
            elif command[0] == "split" and len(command) == 4:
                lab.split(command[1], command[2], command[3])
            elif command[0] == "probe":
                print(json.dumps(lab.report(), sort_keys=True))
            elif command[0] == "reset":
                lab.reset()
            elif command[0] == "help":
                print("show | preserve OLD | replace OLD NEW | split OLD NEW_A NEW_B | probe | reset | quit")
            elif command[0] in {"quit", "exit"}:
                return
            else:
                print("invalid command; use help")
        except ValueError as error:
            print(json.dumps({"error": str(error)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the fixed scenarios and exit")
    args = parser.parse_args()
    if args.demo:
        run_demo()
    else:
        interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
