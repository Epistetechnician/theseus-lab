"""Real-model smoke test for the transcoder baseline glue (DRAFT, unfrozen, not a result).

Checks only: (1) identity replacement reproduces logits on the real Gemma 2 2B,
(2) variance explained of the public transcoders at the chosen layers under the
model-card convention (plus input-scale diagnostics, never gating), (3) delta LM
loss. Prints JSON to stdout. Fixed short text, so numbers are smoke values only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import transcoder_baseline_core as core
import transcoder_baseline_model as tm

LAYERS = {5: 87, 12: 111, 20: 88}  # closest average_l0 to 100, ties smaller (draft 4c)
TEXT = [
    "The committee reviewed the proposal for the new bridge and concluded that the cost estimates were unrealistic given the current price of steel.",
    "In 1998 the research group published a short paper describing how small perturbations in the input can change the output of a network.",
    "Rainfall across the northern valleys was well above average this spring, which delayed planting but improved the outlook for the summer harvest.",
    "She opened the old notebook and found a list of names, each followed by a date and a few words about a place she had never visited.",
    "The compiler reports an error when two functions share the same name and signature, even if their bodies are identical in every respect.",
    "Several economists argued that the central bank should wait for more data before changing its policy rate again this year.",
    "A careful measurement of a small effect requires an instrument whose own noise is much smaller than the effect being measured.",
    "The museum's collection includes pottery from three different periods, along with tools and ornaments recovered from a single burial site.",
]


def main(model_dir: str, params_root: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to("cuda").eval()
    enc = [tok(t)["input_ids"] for t in TEXT]  # includes BOS
    n = min(len(e) for e in enc)  # truncate to the shortest so no padding enters the statistics
    ids = torch.tensor([e[:n] for e in enc]).to("cuda")
    out: dict = {"n_tokens": int(ids.numel()), "layers": {}}
    base_logits = tm.logits(model, ids)
    for layer, l0 in LAYERS.items():
        path = Path(params_root) / f"layer_{layer}/width_16k/average_l0_{l0}/params.npz"
        raw = np.load(path)
        params = {k: torch.from_numpy(raw[k]).to("cuda") for k in core.GEMMA_SCOPE_KEYS}
        x, y = tm.capture_mlp_io(model, layer, ids)
        d = x.shape[-1]
        row: dict = {"l0": l0, "params_file": str(path)}
        with tm.replace_mlp_block(model, layer, tm.original_block_fn(model, layer)):
            row["identity_max_abs_logit_diff"] = float((tm.logits(model, ids) - base_logits).abs().max())
        fn = lambda t, p=params: core.jumprelu_transcoder(p, t)
        row["variance_explained_model_card"] = tm.variance_explained(fn(x), y)
        row["diagnostic_variance_explained_input_scale"] = {
            "sqrt_d": tm.variance_explained(fn(x * d ** 0.5), y),
            "inv_sqrt_d": tm.variance_explained(fn(x / d ** 0.5), y),
        }
        row["delta_lm_loss"] = tm.delta_lm_loss(model, ids, layer, fn)
        row["gate_variance_explained_ge_0.8"] = bool(row["variance_explained_model_card"] >= 0.8)
        out["layers"][str(layer)] = row
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
