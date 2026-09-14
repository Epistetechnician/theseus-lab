"""Fresh-probe SmolLM2 replication of the local Theseus protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smollm_reference_check import run as run_smollm


FRESH_VISIBLE_PROMPTS = (
    "A thermometer measures",
    "A horse is a",
    "Two plus two equals",
    "A cube has six",
    "The color made by red and white is",
    "A person who studies rocks is a",
    "Seeds need water to",
    "The opposite of empty is",
)

FRESH_HELDOUT_PROMPTS = (
    "The color of the daytime sky is",
    "A rectangle has four",
    "The opposite of north is",
    "The shape with five sides is a",
    "A robin is a kind of",
    "A whale is a",
    "The organ used for breathing is the",
    "The largest ocean is the",
    "A compass has a",
    "The season after spring is",
    "The season before winter is",
    "A bicycle has two",
    "A piano is a musical",
    "Honey comes from",
    "A rainbow appears after",
    "A seed contains a baby",
    "A doctor treats",
    "A farmer grows",
    "The Earth has one natural",
    "The planet known for rings is",
    "The closest star to Earth is the",
    "The moon reflects light from the",
    "A square has four equal",
    "A circle has no",
    "A kilogram weighs more than a",
    "A centimeter is shorter than a",
    "An hour contains sixty",
    "A calendar year usually has twelve",
    "The first day of the week is",
    "A pair means",
    "A trio contains",
    "The number after nine is",
    "The number before one hundred is",
    "A dozen eggs contains",
    "A triangle with equal sides is",
    "A liquid takes the shape of its",
    "Ice melts into",
    "Steam is water in its",
    "A shadow forms when light is",
    "A mirror can reflect",
    "A battery stores",
    "A magnet can attract",
    "Wood usually floats on",
    "A thermometer can measure",
    "A ruler measures",
    "A microscope makes small objects appear",
    "A telescope makes distant objects appear",
    "A library is a place with",
    "A museum displays",
    "A kitchen is used for",
    "A bed is used for",
    "A coat keeps a person",
    "Shoes protect the",
    "A helmet protects the",
    "A key can open a",
    "A clock tells the",
    "A map represents a",
    "A bridge crosses a",
    "A tunnel passes under or through",
    "A boat travels on",
    "An airplane travels through",
    "A train runs on",
    "A letter is part of a",
    "A sentence contains words and",
)


def run(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    return run_smollm(
        model_dir,
        model_id or "smollm2-135m",
        visible_prompts=FRESH_VISIBLE_PROMPTS,
        heldout_prompts=FRESH_HELDOUT_PROMPTS,
        probe_suite_name="smollm2_fresh_v1",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
