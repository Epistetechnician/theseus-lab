"""Deterministic terminal prototype for the Theseus identity question."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable


BEHAVIOR_MIN = 0.95
HELDOUT_MIN = 0.95
MECHANISM_MIN = 0.80
VISIBLE_PROBES = (0, 1)
HELDOUT_PROBES = (2, 3)

# The first two probe dimensions are visible during the transformation. The
# last two are held out and expose whether visible equivalence was accidental.
SAFE_PROFILE = (0.4, 0.6, 0.9, -0.2)
TASK_PROFILE = (0.6, 0.2, 0.3, 0.5)


def evaluate(
    components: dict[str, float],
    mechanisms: dict[str, tuple[float, ...]],
    probe_indexes: tuple[int, ...],
) -> tuple[float, ...]:
    outputs = [0.0 for _ in probe_indexes]
    for name, mass in sorted(components.items()):
        profile = mechanisms[name]
        for output_index, profile_index in enumerate(probe_indexes):
            outputs[output_index] += mass * profile[profile_index]
    return tuple(round(value, 10) for value in outputs)


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


@dataclass(frozen=True)
class State:
    step: int
    components: dict[str, float]
    mechanisms: dict[str, tuple[float, ...]]

    @property
    def behavior(self) -> tuple[float, ...]:
        return evaluate(self.components, self.mechanisms, VISIBLE_PROBES)

    @property
    def heldout_behavior(self) -> tuple[float, ...]:
        return evaluate(self.components, self.mechanisms, HELDOUT_PROBES)

    @property
    def state_digest(self) -> str:
        return digest({
            "step": self.step,
            "components": self.components,
            "mechanisms": self.mechanisms,
            "behavior": self.behavior,
            "heldout_behavior": self.heldout_behavior,
        })


@dataclass(frozen=True)
class Transition:
    before: str
    after: str
    lineage: tuple[tuple[str, str, float], ...]


@dataclass
class Lab:
    initial: State = field(default_factory=lambda: State(
        0,
        {"safe": 1.0, "task": 1.0},
        {"safe": SAFE_PROFILE, "task": TASK_PROFILE},
    ))
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

    def transform(
        self,
        components: dict[str, float],
        lineage: Iterable[tuple[str, str, float]],
        action: str,
        mechanisms: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        previous = self.current
        next_mechanisms = previous.mechanisms if mechanisms is None else mechanisms
        next_state = State(
            previous.step + 1,
            dict(sorted(components.items())),
            dict(sorted(next_mechanisms.items())),
        )
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
        mechanisms = dict(current.mechanisms)
        del components[old]
        del mechanisms[old]
        components[new] = 1.0
        old_profile = current.mechanisms[old]
        # Replacement is deliberately equivalent on visible probes but
        # different on held-out probes: this is a probe-coverage adversary.
        mechanisms[new] = (old_profile[0], old_profile[1], -0.4, 0.8)
        self.transform(components, (), f"replace {old} -> {new}", mechanisms)

    def split(self, old: str, first: str, second: str) -> None:
        current = self.current
        if old not in current.components:
            raise ValueError(f"unknown component: {old}")
        components = dict(current.components)
        mechanisms = dict(current.mechanisms)
        del components[old]
        components[first] = 0.5
        components[second] = 0.5
        old_profile = mechanisms.pop(old)
        mechanisms[first] = old_profile
        mechanisms[second] = old_profile
        self.transform(
            components,
            ((old, first, 0.5), (old, second, 0.5)),
            f"split {old} -> {first},{second}",
            mechanisms,
        )

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
        visible_behavior = self._behavior_similarity(before.behavior, after.behavior)
        heldout_behavior = self._behavior_similarity(before.heldout_behavior, after.heldout_behavior)
        causal_continuity = self._causal_intervention_continuity(before, after)
        visible_drift = visible_behavior >= BEHAVIOR_MIN and mechanism < MECHANISM_MIN
        heldout_failure = visible_behavior >= BEHAVIOR_MIN and heldout_behavior < HELDOUT_MIN
        structural_counterexample = (
            visible_behavior >= BEHAVIOR_MIN
            and heldout_behavior >= HELDOUT_MIN
            and mechanism < MECHANISM_MIN
        )
        return {
            "behavior_continuity": round(visible_behavior, 4),
            "visible_behavior_continuity": round(visible_behavior, 4),
            "heldout_behavior_continuity": round(heldout_behavior, 4),
            "mechanism_continuity": round(mechanism, 4),
            "causal_intervention_continuity": round(causal_continuity, 4),
            "component_set_jaccard": round(self._component_set_jaccard(before, after), 4),
            "source_preservation": round(source_preservation, 4),
            "target_coverage": round(target_coverage, 4),
            "behaviorally_equivalent_mechanistic_drift": visible_drift,
            "visible_equivalence_hidden_failure": heldout_failure,
            "structure_only_theseus_counterexample": structural_counterexample,
            "transition_digest": digest({
                "before": transition.before,
                "after": transition.after,
                "lineage": transition.lineage,
            }),
        }

    @staticmethod
    def _component_set_jaccard(before: State, after: State) -> float:
        before_names = set(before.components)
        after_names = set(after.components)
        union = before_names | after_names
        return 1.0 if not union else len(before_names & after_names) / len(union)

    @classmethod
    def _causal_intervention_continuity(cls, before: State, after: State) -> float:
        before_effects = cls._intervention_effects(before)
        after_effects = cls._intervention_effects(after)
        common = sorted(set(before_effects) & set(after_effects))
        coverage = len(common) / max(len(before_effects), len(after_effects))
        if not common:
            return 0.0
        effect_similarity = sum(
            cls._behavior_similarity(before_effects[name], after_effects[name])
            for name in common
        ) / len(common)
        return coverage * effect_similarity

    @staticmethod
    def _intervention_effects(state: State) -> dict[str, tuple[float, ...]]:
        return {
            name: tuple(state.components[name] * value for value in state.mechanisms[name])
            for name in state.components
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
                "mechanisms": self.current.mechanisms,
                "behavior": self.current.behavior,
                "heldout_behavior": self.current.heldout_behavior,
                "state_digest": self.current.state_digest,
            },
            "report": self.report(),
        }, sort_keys=True))


def run_demo() -> None:
    scenarios = (
        ("preserved", lambda lab: lab.preserve("safe")),
        ("structural_split", lambda lab: lab.split("safe", "safe-a", "safe-b")),
        ("probe_coverage_adversary", lambda lab: lab.replace("safe", "replacement")),
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
