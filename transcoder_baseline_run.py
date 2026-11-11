"""Runner for the transcoder baseline (DRAFT, unfrozen; spec: docs/transcoder-baseline-literature-and-draft.md 3, 4e, 4i).

`run_layer` is model-agnostic (any Gemma2ForCausalLM pair in float32 / float64) so it is unit-tested on a tiny
random model; `main` wires the real Gemma 2 2B, the public Gemma Scope transcoders and a text corpus.
Nothing here is a result until the packet is frozen and the run is approved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import transcoder_baseline_core as core
import transcoder_baseline_model as tm

STRATA = {"scale_1e-3": 1e-3, "scale_1e-1": 1e-1, "scale_1": 1.0}
N_CASES = 24
N_NOISE_CASES = 4
SEQ = 64
SEED = 0
BATCH = 8


def _direction(d: int, seed: int) -> torch.Tensor:
    u = torch.randn(d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return u / u.pow(2).mean().sqrt()  # unit RMS, like the un-gained MLP input


def _intervened(fn: Callable, pos: int, delta: torch.Tensor) -> Callable:
    def g(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, pos] = x[:, pos] + delta.to(x.dtype).to(x.device)
        return fn(x)
    return g


def _noisy(fn: Callable, gen: torch.Generator, rel: float) -> Callable:
    def g(x: torch.Tensor) -> torch.Tensor:
        noise = (torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device) * 2 - 1) * rel
        return fn(x * (1 + noise))
    return g


def effect(model, layer: int, ids: torch.Tensor, fn: Callable, pos: int, delta: torch.Tensor,
           wrap: Callable[[Callable], Callable] = lambda f: f) -> torch.Tensor:
    """logits(with intervention) - logits(without), both through `fn` at `layer` (flattened, float64, CPU)."""

    with tm.replace_mlp_block(model, layer, wrap(fn)):
        base = tm.logits(model, ids)
    with tm.replace_mlp_block(model, layer, wrap(_intervened(fn, pos, delta))):
        pert = tm.logits(model, ids)
    return (pert - base).double().reshape(-1).cpu()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return core.cosine(a.numpy(), b.numpy())


def _med(v: list[float]) -> float:
    return float(np.median(v)) if v else float("nan")


def convention(model32, layer: int, eval_ids: torch.Tensor, fn: Callable) -> dict[str, Any]:
    """Amended gate (4i): BOS (position 0) excluded; VE reported not gated."""

    x, y = tm.capture_mlp_io(model32, layer, eval_ids)
    d = x.shape[-1]
    nb = lambda t: t[:, 1:]
    ve = tm.variance_explained(fn(x)[:, 1:], nb(y))
    alts = {"sqrt_d": tm.variance_explained(fn(x * d ** 0.5)[:, 1:], nb(y)),
            "inv_sqrt_d": tm.variance_explained(fn(x / d ** 0.5)[:, 1:], nb(y))}
    cos = core.cosine(fn(x)[:, 1:].double().cpu().numpy(), nb(y).double().cpu().numpy())
    out = core.convention_check(ve, alts, cos)
    out.update({"variance_explained_nonbos": ve, "alternative_variance_explained": alts, "nonbos_cosine": cos})
    return out


def run_layer(model32, model64, layer: int, params: dict[str, torch.Tensor], fit_ids: torch.Tensor, eval_ids: torch.Tensor) -> dict[str, Any]:
    dev = next(model32.parameters()).device
    d = model32.config.hidden_size
    tc = lambda p: (lambda x: core.jumprelu_transcoder(p, x))
    conv = convention(model32, layer, eval_ids, tc(params))
    row: dict[str, Any] = {"convention": conv}
    if not conv["passes"]:
        row["status"] = "NOT_RUN: convention unverified"
        return row

    x, y = tm.capture_mlp_io(model32, layer, fit_ids)
    w, b = core.ridge_affine(x[:, 1:].reshape(-1, d), y[:, 1:].reshape(-1, d))  # BOS excluded from the fit
    affine = lambda xx: (xx.double() @ w + b).to(xx.dtype)
    arms = {"transcoder": tc(params), "affine": affine, "control_blend": tc(core.blend_with_random(params))}
    orig32, orig64 = tm.original_block_fn(model32, layer), tm.original_block_fn(model64, layer)

    # fixed cases, identical across arms and precisions
    n = eval_ids.shape[0]
    cases = [(i % n, 1 + (i * 7) % (SEQ - 2), _direction(d, SEED * 1000 + i)) for i in range(N_CASES)]
    base64_norm = float(tm.logits(model64, eval_ids[:1].to(dev)).double().norm())

    results: dict[str, Any] = {}
    for sname, scale in STRATA.items():
        eff_orig = []
        for (doc, pos, u) in cases:
            ids = eval_ids[doc:doc + 1].to(dev)
            eff_orig.append((effect(model32, layer, ids, orig32, pos, scale * u),
                             effect(model64, layer, ids, orig64, pos, scale * u)))
        eps = _med([core.effect_in_eps(e64.numpy(), base64_norm) for _, e64 in eff_orig])
        for arm, fn in arms.items():
            c32, c64 = [], []
            for (doc, pos, u), (o32, o64) in zip(cases, eff_orig):
                ids = eval_ids[doc:doc + 1].to(dev)
                a32 = effect(model32, layer, ids, fn, pos, scale * u)
                a64 = effect(model64, layer, ids, fn, pos, scale * u)
                c32.append(_cos(o32, a32)); c64.append(_cos(o64, a64))
            results.setdefault(arm, {})[sname] = {
                "median_cos_float32": _med(c32), "median_cos_float64": _med(c64),
                "gap": _med(c64) - _med(c32), "effect_eps": eps}

    # stochastic-perturbation noise floor (4e): float32 runs with last-bit input noise vs the float64 effect
    gen = torch.Generator(device=dev).manual_seed(SEED)
    for arm in ("transcoder", "affine"):
        fn = arms[arm]
        for sname, scale in STRATA.items():
            cs, spread = [], []
            for (doc, pos, u) in cases[:N_NOISE_CASES]:
                ids = eval_ids[doc:doc + 1].to(dev)
                ref = effect(model64, layer, ids, fn, pos, scale * u)
                runs = [effect(model32, layer, ids, fn, pos, scale * u, wrap=lambda f: _noisy(f, gen, core.NOISE_REL))
                        for _ in range(core.NOISE_K)]
                cs.extend(_cos(r, ref) for r in runs)
                sc = float(np.median([r.norm() for r in runs]))
                spread.append(max(float((runs[i] - runs[j]).norm()) for i in range(len(runs)) for j in range(i + 1, len(runs))) / sc if sc else float("nan"))
            results[arm][sname]["noise_floor"] = {"median_cos_vs_float64": _med(cs), "min_cos_vs_float64": float(np.nanmin(cs)),
                                                  "median_relative_spread": _med(spread)}

    gates = {arm: core.case_gate({s: v["median_cos_float64"] for s, v in results[arm].items()}) for arm in arms}
    losses = {}
    for arm, fn in arms.items():
        parts = [tm.delta_lm_loss(model32, eval_ids[i:i + BATCH].to(dev), layer, fn) for i in range(0, eval_ids.shape[0], BATCH)]
        losses[arm] = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
    row.update({"status": "RUN", "arms": results, "gate_float64": gates, "delta_lm_loss": losses})
    return row


def summarize(layers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ran = {k: v for k, v in layers.items() if v.get("status") == "RUN"}
    p1 = [{"layer": k, "transcoder_min_cos": min(s["median_cos_float64"] for s in v["arms"]["transcoder"].values()),
           "affine_min_cos": min(s["median_cos_float64"] for s in v["arms"]["affine"].values())} for k, v in ran.items()]
    p2 = [{"effect_eps": s["effect_eps"], "gap": s["gap"]} for v in ran.values() for s in v["arms"]["transcoder"].values()]
    control_fails = {k: not v["gate_float64"]["control_blend"]["passes"] for k, v in ran.items()}
    return {"layers_run": sorted(ran), "layers_not_run": sorted(set(layers) - set(ran)),
            "P1_transcoder_beats_affine": core.p1_transcoder_beats_affine(p1) if p1 else None,
            "P2_gap_follows_effect_size": core.p2_gap_follows_effect_size(p2) if p2 else None,
            "P3": "not evaluable: no published delta-loss figure located (draft 4i)",
            "negative_control_fails_gate_per_layer": control_fails,
            "instrument_valid": bool(control_fails) and all(control_fails.values())}


def _ids(tok, text_path: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    raw = Path(text_path).read_bytes()
    docs = [l for l in raw.decode().splitlines() if l.strip()]
    enc = [tok(t)["input_ids"][:SEQ] for t in docs]  # tokenizer prepends BOS
    enc = [e for e in enc if len(e) == SEQ]
    cut = int(len(enc) * 0.7)
    t = lambda xs: torch.tensor(xs, dtype=torch.long)
    return t(enc[:cut]), t(enc[cut:]), hashlib.sha256(raw).hexdigest()


def main(model_dir: str, params_root: str, corpus: str, out_path: str, layers: dict[int, int] | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    layers = layers or {5: 87, 12: 111, 20: 88}
    tok = AutoTokenizer.from_pretrained(model_dir)
    m32 = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to("cuda").eval()
    m64 = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float64).to("cuda").eval()
    fit, ev, corpus_sha = _ids(tok, corpus)
    out: dict[str, Any] = {"corpus_sha256": corpus_sha, "n_fit_docs": int(fit.shape[0]), "n_eval_docs": int(ev.shape[0]), "layers": {}}
    for layer, l0 in layers.items():
        path = Path(params_root) / f"layer_{layer}/width_16k/average_l0_{l0}/params.npz"
        raw = np.load(path)
        params = {k: torch.from_numpy(raw[k]).to("cuda") for k in core.GEMMA_SCOPE_KEYS}
        out["layers"][str(layer)] = {"l0": l0, "params_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     **run_layer(m32, m64, layer, params, fit.to("cuda"), ev)}
    out["summary"] = summarize(out["layers"])
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
