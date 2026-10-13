# Exact structural split for additive modules

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-additive-module-validity-v1`.

## Theorem

Let the output of a module be

```text
y(x) = Σᵢ fᵢ(x) vᵢ
```

where each `fᵢ(x)` is a scalar, each `vᵢ` is a fixed vector in the output
space, and the sum and scalar multiplication use the same field. For any
component `j` and any scalar `α`, replacing its contribution by

```text
fⱼ(x) vⱼ  →  fⱼ(x) (α vⱼ) + fⱼ(x) ((1 − α) vⱼ)
```

leaves `y(x)` exactly unchanged for every input `x`.

**Proof.** By distributivity,
`fⱼ(x)(αvⱼ) + fⱼ(x)((1−α)vⱼ) = fⱼ(x)(α + 1−α)vⱼ = fⱼ(x)vⱼ`.
Every other term is unchanged, so the full sum is unchanged pointwise. ∎

If a deterministic downstream computation receives only `y(x)`, its output
is also unchanged. For a two-layer MLP channel,
`fⱼ(x)` can be the channel activation and `vⱼ` its outgoing projection
column. In a gated MLP, `fⱼ(x)` can be the product of the gate and up
projections after the gate activation, with the down-projection column as
`vⱼ`.

## Scope and limits

This proves exact structural non-uniqueness for the stated additive-module
class. It does not prove that every component in an arbitrary model has this
form, that two trained systems are mechanistically equivalent, or that the
replacement can be learned independently. A candidate subsystem qualifies
only when its actual execution path can be written in the theorem's form at a
fixed insertion point.

The identity is over exact field arithmetic. Floating-point summation order,
quantization, clipping, child-specific normalization, dropout, stateful hooks,
or altered routing can introduce differences. Those effects must be measured
in implementation checks; they do not refute the algebraic theorem. The
theorem says nothing about subjective experience, alignment, safety,
production readiness, or scientific acceptance.

## Architecture-validation packet

The mathematical result is architecture-general for additive modules. The
following finite validation checks whether separate model adapters implement
that theorem correctly. It must not be described as proof about all
architectures.

Before selecting or loading held-out checkpoints, freeze this model-class
matrix and the adapter rules:

| Stratum | Required additive component |
| --- | --- |
| OPT/ReLU decoder | One hidden MLP channel and its outgoing projection column |
| GPT-2/GELU decoder | One MLP channel and its outgoing projection column |
| Llama/SiLU-gated decoder | One gated MLP channel and its down-projection column |
| Encoder-decoder | One qualifying additive MLP channel in an encoder or decoder block |
| Recurrent/state-space | One qualifying additive output-projection channel inside a Mamba mixer |

All five checkpoint identities are chosen and pinned by the custodian only
after the protocol and implementation are frozen. Select fresh
architecture-held-out checkpoints that were not used to design the protocol,
choose thresholds, debug adapters, or tune the experiment. Record their
identities and exact artifact digests in the owner-custodied packet before
execution. Existing GPT-2, OPT, Llama-family, and SmolLM2 experiment
checkpoints are not eligible as fresh holdouts if they informed protocol
design.

### Proof obligations

For every stratum and checkpoint:

1. Document the inspected computation graph and show that the selected local
   contribution is `fⱼ(x)vⱼ` at the stated insertion point.
2. Implement the transformation independently in PyTorch and NumPy code paths.
   Neither implementation may call the other to produce the transformed
   output or expected result.
3. Check the local component sum directly in float64 with maximum normalized
   error `≤ 1e-12`. Check full-model logits in float32 with maximum normalized
   error `≤ 1e-5` on the frozen held-out prompts. Report top-1 agreement as a
   diagnostic, implementation-to-implementation error, and output digests;
   top-1 alone is not a gate. These numerical tolerances account for
   finite-precision execution; the theorem itself remains exact over field
   arithmetic.
4. Run the frozen packet in a clean second runtime with a recorded runtime,
   dependency, source, checkpoint, fixture, and output digest. This same-owner
   reproduction can earn `REPRODUCED_LOCAL`; it is not independent review.
5. Retain failures and unsupported adapter cases as explicit
   `INCONCLUSIVE` outcomes. Do not substitute a different checkpoint or
   architecture after seeing a failure without issuing a new protocol and
   packet.

The local acceptance target is all five strata passing in both implementations
in the primary runtime and in a distinct clean second runtime with identical
packet and checkpoint digests. The runner
[`additive_module_validity.py`](../additive_module_validity.py) loads only
local checkpoints from an external owner-custody root, requires full revision
SHAs and an after-freeze checkpoint-selection attestation, and emits no
owner-only checkpoint path. It refuses model runs unless NumPy, Safetensors,
Tokenizers, PyTorch, and Transformers match the frozen `requirements.txt`.
Only then report `REPRODUCED_LOCAL` for this
predeclared finite model class. A separate blind internal reviewer may then
provide `SEGREGATED_INTERNAL_REVIEW`; this is not `INDEPENDENT_ACCEPT`. The
claim remains exact structural validity for additive modules plus successful
adapter reproduction across the named finite class, not “all architectures.”

### Local runner contract

The owner-custodied manifest is not checked into this repository. It has this
shape, with all five checkpoint identities chosen only after the packet is
frozen:

```json
{
  "protocol_id": "theseus-additive-module-validity-v1",
  "checkpoints": [
    {
      "family": "opt_relu",
      "model_id": "<public model repository>",
      "revision": "<40-character commit SHA>",
      "path": "model/<custodied-checkpoint-directory>",
      "selected_after_protocol_freeze": true
    }
  ]
}
```

Repeat one entry for each family in the matrix. `path` is relative to the
owner-custody root and must remain outside the repository. The runner verifies
the five-family set, full revision SHA format, local-only loading, checkpoint
tree digest, and freshness attestation. It does not fetch weights. The
primary and clean second runtimes each write aggregate-safe JSON receipts to
external custody. Compare them with `--compare-first` and `--compare-second`;
the comparator requires the same packet, source, checkpoint and tokenized
fixture identities, distinct runtime fingerprints, matching complete installed
dependency manifests, passing gates, and reproduction metrics within tolerance
before reporting `REPRODUCED_LOCAL`.

The eight passing unit checks use random tiny architecture configurations to
exercise projection mapping, PyTorch and NumPy code paths, and the T5 and
Mamba forward paths. They validate adapter wiring only; they are not checkpoint
replication and do not earn architecture-level assurance.

## Required report fields

Every run report records the exact source revision and dirty-tree identity,
state slice, protocol and artifact digests, assurance level, reviewer status,
reproduction status, checkpoint and fixture custody, runtime/toolchain,
controls, resource use, known limitations, claim ceiling, nonclaims, and the
next blocked authority gate. Model weights, raw prompts, traces, and
owner-only paths remain in approved external custody; this repository stores
protocol and aggregate-safe results only.

Current status: the theorem above is proved algebraically. The finite
architecture-validation packet and adapter implementation have local
synthetic smoke checks, but no fresh checkpoint has been evaluated and no
architecture replication assurance level is earned yet. The existing v2
structural split is an `INTERNAL_PASS` only for its local
structural-replacement scope.
