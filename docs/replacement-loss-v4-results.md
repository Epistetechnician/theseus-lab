# Replacement-loss-v4 results

State slice: `theseus-lab-sandbox-v1`.

Assurance: `REPRODUCED_LOCAL` for the exact v4 packet and aggregate outputs in
a distinct clean local runtime. The frozen evaluation outcome is `FAIL`.

## Frozen identities and custody

- Protocol packet SHA-256:
  `1006d4a861aa442a18307fb9e1e6e5ea6429df57b0c9fd57e2d766bdb4822be5`
- Reconciliation SHA-256:
  `46b15b4ea27160fa1150e30decbbe096a5c710947fcf802faebd5583d3969c18`
- Primary report SHA-256:
  `b176c567b8a06790e38a5639c7e03ab3d3a933febaa25e69212eff9078be54b2`
- Replica report SHA-256:
  `a22e0eb28ca546e33340ad5a6c9e7cd8f86ee91aeec22d0e7365ede741cb18dd`
- Replication receipt SHA-256:
  `31eb9602a3b0a38a02b9ecc62b3a8082409cdd48a434ae594038fc05de78300b`

Reports, prompts, model weights, activation captures, candidates, predictions,
and row-level outcomes remain under external owner custody at
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v4` and
the corresponding `-v4-replicator` directory. The exact prompt text and
row-level diagnostics are not stored in the repository.

## Findings

The compositional-augmented arm passed its compositional activation threshold
and paired-improvement threshold in all three seeds. Median normalized error
fell from `0.798–0.856` with ordinary-only adaptation to `0.058–0.065` with
compositional adaptation. Every compositional challenge row improved in all
three seeds. The combined median error over ordinary and adversarial rows stayed
within the predeclared non-inferiority bound relative to ordinary-only
adaptation.

The complete protocol still failed: the augmented arm exceeded the frozen
adversarial next-logit error ceiling `0.05` in every seed (`0.133`, `0.071`,
`0.139`). These failures remain in the result; no threshold was changed. The
activation effect is specific to this checkpoint, target projection, synthetic
prompt generator, and three adaptation seeds.

## Reproduction and gates

The clean second runtime reconstructed the same candidate states and reproduced
all prediction and outcome array digests. Source identity, dependency inventory,
and challenge fixture identities matched. This is `REPRODUCED_LOCAL`, not
independent acceptance. Reviewer status is `EXTERNAL_REVIEW_UNAVAILABLE`; no
external review task was submitted.

The protocol does not establish fresh-checkpoint generalization, all-architecture
validity, independent science, alignment, consciousness, or production
authorization. Monetary cost and energy were not measured. The next scientific
question is why the same adaptation change that sharply improves compositional
activation reconstruction still leaves adversarial behavior outside the frozen
guard.
