"""V3 binding for the independent numeric factorial closeout.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v3,
global-experiment-assurance-lifecycle-v1.
"""
import importlib
import json
import hashlib

import theseus_v39_fit_factorial_v3 as protocol

closeout = importlib.import_module("theseus_v39_fit_factorial_v1_closeout")
closeout.PROTOCOL = protocol.PROTOCOL
closeout.ROOT = protocol.BASE
closeout.MEMBERS = protocol.MEMBERS


def verify_freeze() -> dict:
    value = json.loads(protocol.FREEZE.read_text())
    rows = []
    for member in protocol.MEMBERS:
        source = protocol.REPO / member
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append({"path": member, "bytes": source.stat().st_size, "sha256": digest})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if value["members"] != rows or hashlib.sha256(canonical).hexdigest() != value["packet_sha256"]:
        raise ValueError("V3 frozen source members differ")
    return value


closeout.verify_freeze = verify_freeze


def main() -> dict:
    return closeout.main()


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
