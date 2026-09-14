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
  --distilgpt2-dir /path/to/caller-owned/distilgpt2
```

This cross-checks the NumPy logits against PyTorch/Transformers on 72 fixed
prompts, then tests 48 function-preserving MLP-channel and attention-head
permutations across all 12 layers. A symmetry case is accepted only when raw
component digests differ, blind canonical digests match, the inferred mapping
matches the exact transformed components, visible and held-out behavior
matches, and the aligned intervention matches. Ablation, pruning, and
replacement retain distinct canonical digests and are treated as non-symmetry
controls.

The causal atlas covers three MLP channels and three attention heads at each
of all 12 layers, for 72 entries, with deterministic 95% bootstrap intervals.
The local Jacobian-lens proxy fits an average derivative from eight visible
probes to four fixed target-token logits, then evaluates 64 held-out probes in
post-GELU MLP activation space. It is an operational diagnostic for this
checkpoint, not a full J-space implementation, mechanistic proof, or
consciousness claim.

The cross-check currently covers the pinned GPT-2 and distilgpt2 checkpoints.
Both must pass the complete local protocol for the replication command to
report `CROSS_CHECKPOINT_REPLICATION_PASS`.

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

Run an interactive session:

```sh
python3 theseus_lab.py
```

Commands are `show`, `preserve OLD`, `replace OLD NEW`, `split OLD NEW_A
NEW_B`, `probe`, `reset`, `help`, and `quit`.

The durable question and current verdict live in [NOTES.md](NOTES.md).
