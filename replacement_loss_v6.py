"""Downstream-aware projection fitting for replacement-loss-v6.

State slice: theseus-lab-sandbox-v1. The base model is frozen; only a fresh
affine candidate is trained from externally custodied adaptation prompts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import replacement_loss_v3_runner as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v6-downstream-aware"
TRAIN_EPOCHS = 8
TRAIN_BATCH_SIZE = 4
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 0.0
BEHAVIOR_WEIGHT = 1.0
TEMPERATURE = 1.0


def _digest_state(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def train_behavior_aware_replacement(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    layer: int,
    records: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int = TRAIN_EPOCHS,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    """Train a new affine replacement on activation and next-token KL loss."""

    if not records:
        raise ValueError("v6 adaptation records cannot be empty")
    if epochs < 1 or epochs > TRAIN_EPOCHS:
        raise ValueError("epoch count must be within the frozen v6 schedule")
    rows = v3._token_rows(tokenizer, records, with_continuation=False)
    if any(int(row["prompt_len"]) < 1 for row in rows):
        raise ValueError("each adaptation prompt must contain at least one token")

    target_projection = v3._projection(model, family, layer)
    in_features, out_features, use_bias = v3._module_dimensions(target_projection)
    cpu_rng = torch.random.get_rng_state()
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    try:
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        candidate = torch.nn.Linear(
            in_features, out_features, bias=use_bias, dtype=torch.float32, device="cpu"
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            candidate.parameters(),
            lr=TRAIN_LEARNING_RATE,
            weight_decay=TRAIN_WEIGHT_DECAY,
        )
        order_rng = np.random.default_rng(seed ^ 0x6A09E667)
        losses: list[float] = []
        activation_losses: list[float] = []
        behavior_losses: list[float] = []
        update_count = 0

        for epoch in range(epochs):
            order = order_rng.permutation(len(rows)).tolist()
            shuffled = [rows[index] for index in order]
            for group, inputs in v3._batches(shuffled, tokenizer.pad_token_id, batch_size=TRAIN_BATCH_SIZE):
                with torch.no_grad():
                    teacher_logits, _teacher_inputs, teacher_outputs = v3._run_projection(
                        model, family, layer, inputs, positions=None
                    )
                if teacher_logits is None or teacher_outputs is None:
                    raise RuntimeError("teacher pass did not capture projection outputs and logits")

                selected_positions = torch.tensor(
                    [int(row["prompt_len"]) - 1 for row in group], dtype=torch.long
                )
                batch_indices = torch.arange(len(group), dtype=torch.long)
                teacher_next = teacher_logits[batch_indices, selected_positions].detach().float()
                mask = inputs["attention_mask"].bool()
                teacher_outputs = teacher_outputs.detach().float()
                hooked: dict[str, Any] = {"count": 0, "output": None}

                def replace_output(
                    _module: torch.nn.Module,
                    args: tuple[torch.Tensor, ...],
                    _output: torch.Tensor,
                ) -> torch.Tensor:
                    hooked["count"] += 1
                    output = candidate(args[0].detach().float())
                    hooked["output"] = output
                    return output

                handle = target_projection.register_forward_hook(replace_output)
                try:
                    student_logits = model(**inputs, use_cache=False, return_dict=True).logits
                finally:
                    handle.remove()
                if hooked["count"] != 1:
                    raise RuntimeError("candidate replacement hook was not called exactly once")

                student_next = student_logits[batch_indices, selected_positions].float()
                student_outputs = hooked["output"]
                if not isinstance(student_outputs, torch.Tensor):
                    raise RuntimeError("candidate hook did not retain a differentiable boundary output")
                with torch.no_grad():
                    target_norm = (teacher_outputs.square() * mask.unsqueeze(-1)).sum().clamp_min(1e-12)
                activation_loss = (
                    ((student_outputs - teacher_outputs).square() * mask.unsqueeze(-1)).sum()
                    / target_norm
                )
                behavior_loss = F.kl_div(
                    F.log_softmax(student_next / TEMPERATURE, dim=-1),
                    F.softmax(teacher_next / TEMPERATURE, dim=-1),
                    reduction="batchmean",
                ) * (TEMPERATURE**2)
                total_loss = activation_loss + BEHAVIOR_WEIGHT * behavior_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("v6 treatment loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                losses.append(float(total_loss.detach().cpu()))
                activation_losses.append(float(activation_loss.detach().cpu()))
                behavior_losses.append(float(behavior_loss.detach().cpu()))
                update_count += 1

        candidate.eval()
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "training_method": "fresh_affine_adamw_activation_plus_next_token_kl",
            "seed": int(seed),
            "epochs": int(epochs),
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": TRAIN_LEARNING_RATE,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "activation_weight": 1.0,
            "behavior_weight": BEHAVIOR_WEIGHT,
            "temperature": TEMPERATURE,
            "adaptation_prompt_count": len(records),
            "optimizer_update_count": update_count,
            "mean_training_loss": float(np.mean(losses)),
            "mean_activation_loss": float(np.mean(activation_losses)),
            "mean_behavior_loss": float(np.mean(behavior_losses)),
            "candidate_state_sha256": _digest_state(candidate),
        }
        return candidate, receipt
    finally:
        for parameter, originally_trainable in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(originally_trainable)
        model.train(was_training)
        torch.use_deterministic_algorithms(was_deterministic)
        torch.random.set_rng_state(cpu_rng)
