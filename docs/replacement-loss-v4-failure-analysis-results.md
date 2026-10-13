# Replacement-loss-v4 adversarial failure diagnosis results

State slice: `theseus-lab-sandbox-v1`.

Assurance: `REPRODUCED_LOCAL` for the digest-frozen post-hoc aggregate in two
distinct local Python environments. The v4 experiment remains `FAIL`.

## Frozen identities and custody

- V4 experiment packet: `1006d4a861aa442a18307fb9e1e6e5ea6429df57b0c9fd57e2d766bdb4822be5`.
- Diagnosis packet digest: `49928db7c35e5890fd948e85c832633f1598e6f86b7b5be7aa32dd030b906311`.
- Aggregate report digest: `b704c0033c10ce447006da283ff661ce76a6ecc4bf9b8d010567deec12dbff0f`.
- Aggregate report file SHA-256: `4b7517666b56fdd3e544406d022f2f4284ebd1d5533714c7a1e3e6574e54fd14`.
- Reconciliation digest: `f6a07ca9b50987b455624bb1009fa8b7fefbe7ab237ef3cc8b9d534cc285a677`.

The nine digest-bound v4 prediction, outcome, and report inputs remain under
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v4` and
its `-replicator` directory. The diagnostic report contains aggregate values
only. Prompt text, IDs, token data, activations, logits, and per-row errors
were not emitted or copied to the repository.

## Findings

The predeclared diagnostic classified all three compositional-augmented
adversarial cases as `BROAD_GUARD_MISS`: 14/20, 4/20, and 15/20 rows exceeded
the frozen normalized next-logit ceiling of `0.05`. The corresponding median
errors were `0.0633`, `0.0296`, and `0.0671`; maxima were `0.1328`, `0.0708`,
and `0.1389`. The four violations in the middle seed remain a measured miss
under the preregistered upper-tail rule of at most two violations.

Within-seed Spearman correlations between adversarial activation and logit
error for the compositional-augmented candidate were `0.003`, `0.262`, and
`0.123`. Paired compositional-augmented minus ordinary-only median logit-error
deltas were `+0.0075`, `-0.0374`, and `-0.0180`; the direction varied by seed.
The compositional candidate reduced adversarial median activation error in all
three seeds, but those activation changes did not consistently predict the
behavior effect.

This is post-hoc descriptive evidence. It does not revise v4's `FAIL`, establish
causality, or show that activation reconstruction is a reliable behavior
proxy. The next experiment is the separately frozen v5 balanced
composition/adversarial adaptation intervention.

## Reproduction and claim ceiling

The frozen aggregate report digest matched exactly in the primary and clean
second local runtimes, with distinct runtime prefixes. This is
`REPRODUCED_LOCAL`, not independent acceptance. Reviewer status remains
`EXTERNAL_REVIEW_UNAVAILABLE`; no review task was submitted.

Claim ceiling: `POSTHOC_LOCAL_GPT2_V4_FAILURE_DIAGNOSIS` for the same GPT-2
checkpoint, layer 3 projection, v4 synthetic challenge set, and three seeds.
This diagnosis does not establish architecture generality, mechanistic
understanding, independent science, alignment, consciousness, or production
authorization. Monetary cost and energy were not measured.
