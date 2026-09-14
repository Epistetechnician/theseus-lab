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
  --smollm-dir /Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm-135m
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
`--smollm-dir` is supplied. All selected checkpoints must pass the complete
local protocol and share the probe digest. A run including SmolLM reports
`CROSS_TOPOLOGY_REPLICATION_PASS`.

The four-checkpoint topology replay passed. GPT-2 passed 48/48 symmetry cases,
distilgpt2 passed 24/24, OPT-125M passed 48/48, and SmolLM-135M passed 120/120;
all four recovered every blind mapping. Their control counts were 14, 8, 14,
and 32 respectively, with canonical digest differences in every control.
Every checkpoint passed its local reference and Jacobian-proxy gates, and the
combined command used probe digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf` and
reported `CROSS_TOPOLOGY_REPLICATION_PASS`.

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
/Users/shaanp/.codex/venvs/theseus-lab-31111/bin/python \
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
