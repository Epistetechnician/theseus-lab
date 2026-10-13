# Replacement-loss-v6 results

State slice: `theseus-lab-sandbox-v1`.

Status: `INCONCLUSIVE`. The v6 packet and checkpoint identities were frozen,
but the primary run stopped during the first candidate-fitting case. No
candidate freeze, challenge fixture, held-out prediction, or outcome was
produced.

The frozen packet SHA-256 is
`9b7b669195bb4fb9153e872d42f3cb4fd235e9784fdc978ec936037eb176bc5a`.
The exact packet files and manifest are archived in external owner custody.

The pinned v3 evaluator packet is
`d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83`.
The post-freeze checkpoint manifest SHA-256 is
`fcf0b2f66f1be4444a715f185271e010e9f80187f62ce36cf72370926e66c067`.
The three selected checkpoints passed local architecture and parameter-count
preflight:

| Family | Frozen checkpoint and revision | Parameters |
| --- | --- | ---: |
| OPT/ReLU | `babylm/opt-125m-strict-2023@cdc1b7349c61df949df94a03bdd7b43be5313cec` | 125,239,296 |
| GPT-2/GELU | `BabyLM-community/babylm-baseline-100m-gpt2@b26625ffec47c1b215d6174d6ce21370cf8f8209` | 98,425,344 |
| Llama/SiLU-gated | `babylm/babyllama-100m-2024@9c1aeb3459c892ad28fd56ece6672a223c43ba9c` | 58,343,936 |

The runner stopped in `opt_relu`, seed index 0. The projection hook supplied a
flattened batch-by-sequence output while the teacher activation was
batch-by-sequence-by-hidden, causing an activation-loss shape error. Candidate
preparation did not complete. The partial fixtures, capture, and candidate
files remain in owner custody; their inventory digest is
`5c364f706867659d42890c4280bb7200b346f922b4054485e5a114a58c2dcffa` across
10 files. The external abort receipt digest is
`bab0068eff879a55d0d6d95270b591031a96d340a68cba63b53e497de8cce90b`.

Assurance is `INCONCLUSIVE`; reproducibility is `NOT_RUN`; reviewer status is
`EXTERNAL_REVIEW_UNAVAILABLE`. No external review task was submitted. The
frozen packet was not edited. The three selected checkpoint identities are
now excluded from a continuation run so it can use fresh targets.

This file will record aggregate outcomes, exact packet and checkpoint
identities, local reproduction status, custody, limitations, claim ceiling,
and the next authority gate. It will not contain model weights, prompts,
traces, row-level predictions, or private custody paths.
