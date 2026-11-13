"""Gemma 2 MLP-boundary glue for the transcoder baseline (DRAFT, unfrozen).

Boundary per the Gemma Scope transcoder card: input = activations just after the
pre-MLP RMSNorm with the norm gain folded into the MLP input matrices, i.e. the
RMS-normalised residual WITHOUT the gain; target = MLP output AFTER the post-MLP
RMSNorm. HF Gemma2RMSNorm multiplies by (1 + weight), so the un-gained input is
the hooked output divided by that factor. Whether the paper's extra "divide by
RMS" dataset scaling applies on top is unresolved; `variance_explained` on real
data is the check (see docs), and this module applies no extra scaling.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


def _gain(norm: torch.nn.Module) -> torch.Tensor:
    return 1.0 + norm.weight.detach().to(torch.float64)


@torch.no_grad()
def capture_mlp_io(model, layer: int, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_nogain, y_target), each [batch, seq, d_model] in the model dtype."""

    blk = model.model.layers[layer]
    store: dict[str, torch.Tensor] = {}
    gain = _gain(blk.pre_feedforward_layernorm)
    h1 = blk.pre_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: store.__setitem__("x", (o.double() / gain).to(o.dtype)))
    h2 = blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o: store.__setitem__("y", o.detach()))
    try:
        model(input_ids=input_ids)
    finally:
        h1.remove(); h2.remove()
    return store["x"], store["y"]


@contextmanager
def replace_mlp_block(model, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Replace post_norm(MLP(pre_norm(x))) by fn(x_nogain) for the duration of the context."""

    blk = model.model.layers[layer]
    store: dict[str, torch.Tensor] = {}
    gain = _gain(blk.pre_feedforward_layernorm)
    h1 = blk.pre_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: store.__setitem__("x", (o.double() / gain).to(o.dtype)))
    h2 = blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o: fn(store["x"]).to(o.dtype))
    try:
        yield
    finally:
        h1.remove(); h2.remove()


@torch.no_grad()
def logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids).logits


@torch.no_grad()
def original_block_fn(model, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """The unmodified block as a function of the un-gained input (identity-replacement test)."""

    blk = model.model.layers[layer]
    gain = blk.pre_feedforward_layernorm.weight.detach()

    def fn(x: torch.Tensor) -> torch.Tensor:
        # .forward bypasses the replacement hook on the post-MLP norm (else infinite recursion)
        return blk.post_feedforward_layernorm.forward(blk.mlp(x * (1.0 + gain).to(x.dtype)))
    return fn


def variance_explained(pred: torch.Tensor, target: torch.Tensor) -> float:
    p, t = pred.double().reshape(-1, pred.shape[-1]), target.double().reshape(-1, target.shape[-1])
    return float(1.0 - ((t - p) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def delta_lm_loss(model, input_ids: torch.Tensor, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, float]:
    """Mean next-token cross-entropy, baseline vs replaced (the field-standard delta loss)."""

    def ce(lg: torch.Tensor) -> float:
        return float(torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]).double(), input_ids[:, 1:].reshape(-1)))
    base = ce(logits(model, input_ids))
    with replace_mlp_block(model, layer, fn):
        rep = ce(logits(model, input_ids))
    return {"base_ce": base, "replaced_ce": rep, "delta": rep - base}
