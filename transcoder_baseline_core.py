"""Model-free core for the transcoder baseline (DRAFT, unfrozen).

Pure tensor functions that the not-yet-written model runner will call: a JumpReLU
transcoder forward (Gemma Scope params), closed-form ridge affine fit, effect
metrics in float32 epsilon units, a stochastic-perturbation noise-floor estimate,
and the preregistered gate. Nothing here loads a model or touches custody.
Spec: docs/transcoder-baseline-literature-and-draft.md.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

COSINE_MIN = 0.90
RIDGE = 1e-6
FP32_EPS = float(np.finfo(np.float32).eps)
# Preregistered prediction constants (draft section 3).
P2_LARGE_EFFECT_EPS, P2_LARGE_EFFECT_MAX_GAP = 1e4, 0.02
P2_SMALL_EFFECT_EPS, P2_SMALL_EFFECT_MIN_GAP = 1e2, 0.05
NOISE_K = 8
NOISE_REL = 2.0 ** -23
GEMMA_SCOPE_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")


def jumprelu_transcoder(params: Mapping[str, Any], x: torch.Tensor) -> torch.Tensor:
    """Gemma Scope JumpReLU: pre = x W_enc + b_enc; acts = pre * (pre > threshold); out = acts W_dec + b_dec."""

    missing = [k for k in GEMMA_SCOPE_KEYS if k not in params]
    if missing:
        raise KeyError(f"transcoder params missing {missing}")
    dtype = x.dtype
    w_enc, w_dec, b_enc, b_dec, thr = (torch.as_tensor(params[k]).to(dtype) for k in GEMMA_SCOPE_KEYS)
    pre = x @ w_enc + b_enc
    return (pre * (pre > thr)) @ w_dec + b_dec


def ridge_affine(x: torch.Tensor, y: torch.Tensor, ridge: float = RIDGE) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge fit of y ~ x W + b in float64 (centered, intercept unpenalised)."""

    x64, y64 = x.double(), y.double()
    xm, ym = x64.mean(0), y64.mean(0)
    xc, yc = x64 - xm, y64 - ym
    gram = xc.T @ xc + ridge * torch.eye(xc.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(gram, xc.T @ yc)
    return w, ym - xm @ w


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float("nan") if na == 0 or nb == 0 else float(a @ b / (na * nb))


def effect_in_eps(effect: np.ndarray, logit_norm: float) -> float:
    """Effect-vector norm in units of float32 epsilon times the logit norm (below ~1e2, float32 cosine is rounding-dominated)."""

    return float(np.linalg.norm(np.asarray(effect, np.float64)) / (FP32_EPS * logit_norm))


def perturbation_noise_floor(fn, x: torch.Tensor, k: int = NOISE_K, rel: float = NOISE_REL, seed: int = 0) -> dict[str, float]:
    """Spread of fn(x) under k independent last-bit relative perturbations of x (float32).

    A deliberately crude substitute for stochastic arithmetic: perturbs only the input.
    Returns max pairwise deviation relative to the median output norm.
    """

    g = torch.Generator().manual_seed(seed)
    outs = []
    for _ in range(k):
        noise = (torch.rand(x.shape, generator=g, dtype=x.dtype) * 2 - 1) * rel
        outs.append(fn(x * (1 + noise)).detach().double().numpy())
    stack = np.stack(outs)
    scale = float(np.median(np.linalg.norm(stack.reshape(k, -1), axis=1)))
    spread = float(max(np.linalg.norm(stack[i] - stack[j]) for i in range(k) for j in range(i + 1, k)))
    return {"max_pairwise_deviation": spread, "scale": scale, "relative": spread / scale if scale else float("nan")}


def case_gate(strata_cosines: Mapping[str, float | None]) -> dict[str, Any]:
    ok = bool(strata_cosines) and all(c is not None and not np.isnan(c) and c >= COSINE_MIN for c in strata_cosines.values())
    return {"cosine_ok": ok, "passes": ok}


def p1_transcoder_beats_affine(rows: list[dict[str, Any]]) -> bool:
    """Per layer, transcoder float64 min-stratum cosine exceeds the affine arm's. rows: {layer, transcoder_min_cos, affine_min_cos}."""

    return bool(rows) and all(r["transcoder_min_cos"] > r["affine_min_cos"] for r in rows)


def p2_gap_follows_effect_size(rows: list[dict[str, Any]]) -> bool:
    """rows: {effect_eps, gap}. Large effects must have gap <= 0.02, small effects >= 0.05; a set with neither regime is not evaluable (False)."""

    large = [r for r in rows if r["effect_eps"] >= P2_LARGE_EFFECT_EPS]
    small = [r for r in rows if r["effect_eps"] <= P2_SMALL_EFFECT_EPS]
    if not large and not small:
        return False
    return all(r["gap"] <= P2_LARGE_EFFECT_MAX_GAP for r in large) and all(r["gap"] >= P2_SMALL_EFFECT_MIN_GAP for r in small)
