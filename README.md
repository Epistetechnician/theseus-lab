# Theseus Lab

An isolated terminal prototype for testing the Ship of Theseus thesis:
behavior can remain stable while the mechanism changes, splits, or disappears.

The base demo is an in-memory logic sandbox. It does not load model weights,
call a provider, read assessment data, or make production claims. The separate
local-checkpoint experiment is deliberately explicit about its external model
directory and claim ceiling.

The toy adapter consumes caller-owned checkpoint metadata and performs
deterministic arithmetic only. Neither path calls a provider, reads an
assessment corpus, or makes production claims.

Run the real local checkpoint experiment after installing the pinned local
runtime from [requirements.txt](requirements.txt):

```sh
python gpt2_experiment.py --model-dir /path/to/caller-owned/gpt2
```

The real experiment verifies the seven GPT-2 bundle artifacts, loads 160
Safetensors tensors, runs NumPy GPT-2 inference, and compares a layer-0 MLP
channel permutation on the fixed visible and held-out prompts. The model
directory is never copied into this repository.

Run the out-of-family OPT adapter with:

```sh
python opt_reference_check.py \
  --model-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/opt-125m \
  --model-id opt-125m
```

The OPT path verifies the pinned PyTorch state dictionary, loads 197 tensors,
computes logits with an independent NumPy OPT decoder, and compares those
logits against local Transformers output. OPT channel selection uses only the
eight visible probes; aligned interventions are then evaluated on 16 fixed
held-out probes.

Run the topology-diverse SmolLM-135M adapter with:

```sh
python smollm_reference_check.py \
  --model-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm-135m \
  --model-id smollm-135m
```

The SmolLM path verifies the pinned Llama-family bundle, reconstructs tied
output embeddings from the checked-in input embedding, and compares an
independent NumPy implementation against local Transformers output. It tests
gated-MLP channel permutations and complete grouped-query-attention (GQA)
group permutations across all 30 layers. The model uses rotary position
encoding, RMSNorm, SiLU gating, nine query heads, and three key/value groups.

Run the independent SmolLM2-135M instance with its fresh probe contract:

```sh
python smollm2_reference_check.py \
  --model-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm2-135m \
  --model-id smollm2-135m
```

SmolLM2 uses the same Llama/GQA implementation family but a distinct pinned
checkpoint and RoPE base. Its qualification uses a new 8-visible/64-held-out
probe suite, so the report records a separate contract rather than reusing the
SmolLM probe digest.

Run the preregistered held-out causal prediction, activation-patching, and
activation-rescue experiment on SmolLM2 with:

```sh
.venv/bin/python causal_prediction_check.py \
  --model-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm2-135m \
  --model-id smollm2-135m
```

This protocol selects three MLP channels per layer from the eight visible
probes at layers 0, 14, and 29, evaluates 576 held-out MLP intervention
observations and 576 held-out GQA-group observations, and compares canonical,
raw-coordinate, and fixed-seed random mappings. It then evaluates 576
activation-patching observations and 192 full-activation rescue observations.
The compact report passed as
`CAUSAL_PREDICTION_AND_RESCUE_PASS`: canonical MLP and GQA prediction error
was `0.0` with 100% canonical wins against both baselines; raw-coordinate
relative error was `1.5952721936` for MLP and `5.8210934364` for GQA. Aligned
patching error was `0.0` versus `12.0264040945` for raw patching. Aligned
rescue error was `3.298e-7` versus `0.5280996787` for raw rescue, with 100%
canonical wins in both cases. The causal target-bound probe digest was
`f56f43ffdb106bd3a17bc40824299aef6db17286a80adbec5e9a3ba1738cd461`.
For every selected MLP and GQA layer, the operational mapping was inferred
from exact composite component signatures; the recorded synthetic shift was
used only to construct the transformed tensor case. Each mapping had a raw
digest difference and matching canonical digests.

This result is local checkpoint evidence that the tested causal effects are
recoverable under the exact validated symmetry mapping. It does not establish
mechanistic understanding, architecture-general proof, alignment validation,
independent review, or production readiness; the report records provider
calls as false and assessment status as local synthetic probes only.

Run the fresh OPT-125M causal prediction, activation-patching, and
activation-rescue protocol:

```sh
python opt_causal_prediction_check.py \
  --model-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/opt-125m \
  --model-id opt-125m
```

This protocol uses a new 8-visible/64-held-out probe contract, selects three
active ReLU channels per layer from the visible probes, and evaluates layers
0, 5, and 11 with three MLP channels and three attention heads per layer. It
checks blind component-signature mapping, 576 MLP and 576 attention causal
predictions, 576 activation patches, and 192 full-activation rescues. The
Gman CPU-8 execution loaded the pinned 197-tensor OPT checkpoint and completed
with `exit_code: 0`. MLP prediction, patching, and rescue passed, but the
locked attention random-control win-rate gate was `0.6667` (95% CI
`[0.6267, 0.7049]`) against the required `0.8`, so the aggregate status is
`INCONCLUSIVE`. The result is a falsifying protocol observation, not a reason
to relax the threshold; provider calls, real assessment data, independent
review, mechanistic understanding, alignment validation, and production
readiness remain unclaimed.

Run the independent local reference, identity criterion, causal atlas, and
Jacobian-lens proxy with:

```sh
python reference_check.py --model-dir /path/to/caller-owned/gpt2
```

The accepted checkpoint IDs and exact artifact hashes are pinned in
[checkpoint_manifests.json](checkpoint_manifests.json). To run the same
qualification protocol on both locally custodied checkpoints:

```sh
python replication_check.py \
  --gpt2-dir /path/to/caller-owned/gpt2 \
  --distilgpt2-dir /path/to/caller-owned/distilgpt2 \
  --opt-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/opt-125m \
  --smollm-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm-135m \
  --smollm2-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm2-135m
```

With the optional checkpoint directories, this cross-checks the NumPy logits against
PyTorch/Transformers on 72 fixed prompts for the GPT-2-family checkpoints,
OPT-125M, and, when supplied, SmolLM-135M. It then tests function-preserving
MLP-channel and attention-family permutations. A symmetry case is accepted
only when raw component digests differ, blind canonical digests match, the
inferred mapping matches the exact transformed components, visible and
held-out behavior matches, and the aligned intervention matches. Ablation,
pruning, and replacement retain distinct canonical digests and are treated as
non-symmetry controls.

For each checkpoint, the causal atlas covers three MLP channels and three
attention components at every layer, with deterministic 95% bootstrap
intervals. The SmolLM atlas has 180 entries (three gated-MLP channels and
three GQA groups at each of 30 layers).
The local Jacobian-lens proxy fits an average derivative from eight visible
probes to four fixed target-token logits, then evaluates 64 held-out probes in
post-activation MLP space. It is an operational diagnostic for this
checkpoint, not a full J-space implementation, mechanistic proof, or
consciousness claim.

The cross-check covers the pinned GPT-2, distilgpt2, and OPT-125M checkpoints
when `--opt-dir` is supplied, and adds the pinned SmolLM-135M topology when
`--smollm-dir` is supplied. `--smollm2-dir` adds a second Llama-family model
instance with its explicitly fresh probe contract. All selected checkpoints
must pass the complete local protocol. A run including SmolLM reports
`CROSS_TOPOLOGY_REPLICATION_PASS`; a run including SmolLM2 reports
`CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS` and separately verifies the
shared legacy contract plus the fresh SmolLM2 contract.

The five-checkpoint topology replay passed. GPT-2 passed 48/48 symmetry cases,
distilgpt2 passed 24/24, OPT-125M passed 48/48, SmolLM-135M passed 120/120,
and SmolLM2-135M passed 120/120; every blind mapping was recovered. Their
control counts were 14, 8, 14, 32, and 32 respectively, with canonical digest
differences in every control. Every checkpoint passed its local reference and
Jacobian-proxy gates. The four checkpoints on the shared suite used digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf` and
the fresh SmolLM2 suite used digest
`8b90443749e884dbd83a0f2e983c5445a1dd6439e8d79f9a5a13fcb88d1bd52b`.
The aggregate reported `CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS`.

This is local cross-topology replication evidence for the tested
transformations. It is not architecture-general proof, mechanistic
understanding, alignment validation, or production readiness.

The formal identity criterion is:

```text
raw digest differs + canonical digest matches + behavior matches
+ aligned interventions match
```

The stronger falsification gate additionally requires blind mapping recovery;
its controls must produce canonical digest differences. The forward path uses
float32 for broad coverage and float64 only for the intervention comparison,
so the intervention threshold remains fixed rather than being relaxed for
accumulated roundoff.

Run the full demonstration:

```sh
python3 theseus_lab.py --demo
```

Run the custody-aware adapter regression tests in the pinned runtime:

```sh
.venv/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

The tests exercise the public OPT and SmolLM adapters against local
Transformers plus their function-preserving channel/group permutations. They
skip when the caller-owned custody bundle or pinned runtime is unavailable.

Run an interactive session:

```sh
python3 theseus_lab.py
```

Commands are `show`, `preserve OLD`, `replace OLD NEW`, `split OLD NEW_A
NEW_B`, `probe`, `reset`, `help`, and `quit`.

The durable question and current verdict live in [NOTES.md](NOTES.md).
