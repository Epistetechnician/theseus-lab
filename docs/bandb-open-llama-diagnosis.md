# Band B diagnosis: why open_llama_7b_v2 fails while opt-6.7b passes

Source: `docs/replacement-loss-v8-scale-bandb-{N256,N512}-report.json` only (no new run). Outcomes and gates are unchanged; this note is analysis, not a result, and every explanation below is a hypothesis unless marked measured.

## Measured: two different failure signatures
Median logit-effect cosine per stratum (gate 0.90), 3 seeds, and module-boundary cosine (how well the replacement reproduces the module's own output effect):

| arm | model | adversarial | compositional | ordinary | module-boundary cos |
|---|---|---:|---:|---:|---:|
| N256 | opt-6.7b | 0.98 | 0.79-0.81 | 0.79-0.79 | 0.880-0.888 |
| N512 | opt-6.7b | 1.00 | 0.99 | 0.99 | 0.99994 |
| N256 | open_llama_7b_v2 | 0.98 | 0.76-0.78 | 0.85-0.88 | 0.99993-0.99996 |
| N512 | open_llama_7b_v2 | 0.98-0.99 | 0.80-0.85 | 0.91-0.93 | 0.99996-0.99997 |
| N512 | gpt2-xl | 0.975 | 0.995 | 0.996 | 0.9999 |

- **opt-6.7b at 256 is a coverage failure.** The fit does not reproduce the module (boundary cosine 0.88), and doubling the adaptation prompts fixes it (0.99994). This matches the Band A story.
- **open_llama_7b_v2 is a different failure.** The fit already reproduces the module almost perfectly at 256 (boundary cosine 0.99993+), and doubling prompts improves the logit cosines only partly (compositional 0.77 -> 0.83, ordinary 0.88 -> 0.92, still below 0.90 for compositional on all 3 seeds). More prompts does not close the gap the way it did for opt-6.7b.
- Direction accuracy is 1.00 and top-1 agreement is 1.00 in every case: the sign and the behavior are right; only the cosine of small effect vectors is low.

## Observation worth testing
Logit-effect `median_normalized_error` is about 3e-7 to 4.5e-7 in all three models, a small multiple of float32 machine epsilon (1.19e-7). If the compositional/ordinary logit effects themselves are only a few times above the float32 rounding floor of the logits, their cosine can be low from rounding noise alone, regardless of how good the replacement is. TF32 was off (runner sets `allow_tf32 = False`), so this is plain float32, not TF32.

## Hypotheses (untested)
1. **Float32 resolution floor on the effect vectors** (favored by the boundary-cosine / error-magnitude pattern; consistent with the r7 finding that r6's equivalence misses were float32 rounding). Prediction: recomputing the logit effects for open_llama in float64 raises the compositional and ordinary cosines to about 0.99.
2. **Depth amplification.** Layer 7 of a 32-layer model leaves 24 layers to amplify a small module residual; opt-6.7b's selected layer is 30 and gpt2-xl's 45 (late layers, short downstream path). Prediction: a later layer in open_llama fits better at the logits.
3. **Architecture.** Llama's gated MLP / RMSNorm may make an affine output projection a worse local model for compositional edits. Prediction: float64 does not help and the gap persists.

Hypotheses 1 and 3 are distinguishable with one preregistered float64 run of open_llama_7b_v2 (about 56 GB of weights, fits one 80 GB H100); hypothesis 2 needs a layer sweep. Neither has been run.

## Claim ceiling
Nothing here changes the recorded Band B outcome: open_llama_7b_v2 `FAIL` at 256 and 512; opt-6.7b `FAIL` at 256, `PASS` at 512. The pooled cosine passes (0.90-0.94) at 512 for open_llama; it fails the per-stratum gate.
