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
channel permutation on visible and held-out prompts. The model directory is
never copied into this repository.

Run the independent reference and layer/head sweep with:

```sh
python reference_check.py --model-dir /path/to/caller-owned/gpt2
```

This cross-checks the NumPy logits against PyTorch/Transformers on 16 fixed
prompts, then tests function-preserving MLP-channel and attention-head
permutations at layers 0, 5, and 11. It also measures non-symmetry sensitivity
for channel ablation, block pruning, and channel replacement. It reports both
scale-free and absolute intervention differences.

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
