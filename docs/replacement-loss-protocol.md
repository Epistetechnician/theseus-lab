# Trained-model replacement/loss protocol

State slice: `theseus-lab-sandbox-v1`.

Protocol: `theseus-replacement-loss-v2`.

This experiment tests a structural replacement in two owner-custodied local
checkpoints: OPT-125M (ReLU MLP) and SmolLM2-135M (gated MLP). It uses a new
8-visible/64-held-out probe contract and three locked seeds per checkpoint.
The target layer and target channel are selected from visible activation traces
only. Held-out outputs are not read until the transformation and thresholds are
locked in the aggregate receipt.

## Treatment

The selected trained-model MLP channel is replaced by two newly materialized
child channels. Each child receives the original incoming channel computation;
each receives half of the original outgoing projection. The original channel
component therefore disappears as an exact parameterized component, the MLP
width increases by one, and the summed treatment contribution is algebraically
equal to the original contribution. This is a structural replacement/loss,
not a permutation and not independently retrained weights.

The mechanism gate requires all of the following:

- the layer width increases by exactly one;
- the canonical sorted component-set digest changes;
- the original target component digest is absent after replacement; and
- ablating both children together matches the original target ablation with
  mean relative error at most `1e-8`; and
- ablating either child separately produces the same effect as ablating the
  other child, with mean relative error at most `1e-8`.

## Locked behavior gates

The treatment must have held-out top-1 agreement of `1.0` and mean relative
logit error at most `1e-8`. The wrong-source and seeded-random controls must
diverge on held-out data by mean relative logit error at least `1e-4` or lose
exact top-1 agreement.

The claim ceiling is
`LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_OBSERVATION_ONLY`. A pass would be
evidence that a trained-model computational component can be structurally
removed and replaced while held-out behavior is preserved under this exact
contract. It would not establish independent retraining, architecture-general
validity, mechanistic understanding, consciousness, alignment, production
readiness, or independent scientific review.
