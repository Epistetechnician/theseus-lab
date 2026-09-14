"""Prototype-only local checkpoint adapter with no filesystem or network I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


PROBE_IDS = (
    "visible_safety",
    "visible_task",
    "heldout_counterfactual_safety",
    "heldout_counterfactual_composition",
)
SAFE_PROFILE = (0.4, 0.6, 0.9, -0.2)
TASK_PROFILE = (0.6, 0.2, 0.3, 0.5)


def full_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointSummary:
    """Caller-owned metadata identifying the deterministic prototype checkpoint."""

    model_id: str
    revision: str
    runtime_id: str
    weights_digest: str
    component_profiles: dict[str, tuple[float, ...]]

    @property
    def summary_digest(self) -> str:
        return full_digest(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "runtime_id": self.runtime_id,
            "weights_digest": self.weights_digest,
            "component_profiles": {
                name: list(profile)
                for name, profile in sorted(self.component_profiles.items())
            },
        }

    @classmethod
    def toy(cls) -> "CheckpointSummary":
        profiles = {"safe": SAFE_PROFILE, "task": TASK_PROFILE}
        return cls(
            model_id="theseus-linear-toy",
            revision="synthetic-checkpoint-v1",
            runtime_id="python-deterministic-arithmetic-v1",
            weights_digest=full_digest({"component_profiles": profiles}),
            component_profiles=profiles,
        )


class LocalCheckpointAdapter:
    """Run a checkpoint summary through deterministic probe arithmetic."""

    def __init__(self, summary: CheckpointSummary | None = None) -> None:
        self.summary = summary or CheckpointSummary.toy()

    @property
    def checkpoint_digest(self) -> str:
        return self.summary.summary_digest

    @property
    def initial_mechanisms(self) -> dict[str, tuple[float, ...]]:
        return dict(self.summary.component_profiles)

    def evaluate(
        self,
        components: dict[str, float],
        mechanisms: dict[str, tuple[float, ...]],
    ) -> tuple[float, ...]:
        if set(components) != set(mechanisms):
            raise ValueError("component and mechanism summaries differ")
        outputs = [0.0 for _ in PROBE_IDS]
        for name, mass in sorted(components.items()):
            profile = mechanisms[name]
            if len(profile) != len(PROBE_IDS):
                raise ValueError(f"profile for {name} has the wrong probe dimension")
            for index, value in enumerate(profile):
                outputs[index] += mass * value
        return tuple(round(value, 10) for value in outputs)
