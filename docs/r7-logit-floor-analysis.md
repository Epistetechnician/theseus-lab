# r7 logit-effect floor: analysis (hypothesis)

State slice `theseus-lab-sandbox-v1`. Scope: code reading of `replacement_loss_v8s7_runner.py` and `replacement_loss_v3_runner.py`, plus a numpy toy (`scripts/r7_logit_floor_toy.py`, tested by `tests/test_r7_logit_floor_toy.py`). No model was run, no weights were loaded, no torch was installed. Status: **hypothesis**. The source of the floor in the real SmolLM2-1.7B run is not identified.

## The open item

`docs/replacement-loss-v8-scale-r7-results.md` reports that the float64 logit-effect error is 3.7e-6..6.6e-6 across six cases, while the float64 module error is 1e-13..4e-12. P2 (>=100x reduction) fails only on the logit path. A pure summation-order effect would be expected to vanish in float64 as well, so something in the logit path is still float32-limited or the comparison is dominated by a fixed-precision term.

## What the code path does (read from source)

1. `diagnose_case` loads the deltas with `np.load(...)["deltas"].astype(np.float32)` (line 173), then runs both `float32` and `float64` passes. The float64 pass casts those already-rounded deltas with `.to(dtype)`. So the float64 pass uses float32-rounded deltas.
2. `_predictions` forms, per arm, `_next_logits(chg_l) - _next_logits(base_l)`, where `_next_logits` (v3 runner, line 623) indexes the final logits at `prompt_len - 1`, detaches, and moves to CPU. The logit effect is compared between the treatment arm and the mapping-aligned permutation arm.
3. `_run_projection` (v3 runner, line 454) applies the deltas inside a forward pre-hook at the projection input, then reads the post-hook output. The positional logits come from the model's own forward pass.

Checked and ruled out as a source of the t-vs-a gap:

- **The float32 delta cast.** Both arms add the same rounded delta vector. For the mapping-aligned arm the delta is permuted with `index_select`, and a permutation of float32 values is exact. So the rounding is common to both arms and cancels in `t - a`. The toy confirms this (`f64_delta32` matches `f64`, both about 1e-11).
- **The module itself.** Module error is 1e-13..4e-12 in float64, so a module-driven logit error of 5e-6 would need an amplification of about 1e6 through the rest of the network. Nothing in the code path shows that. Across the six r7 cases the float64 module error varies about 45x (9.1e-14 to 4.5e-12) while the logit floor varies only about 1.8x (3.7e-6 to 6.6e-6), so the floor does not track module error.

## Candidate sources (not tested on the real model)

- **C1. A float32 rounding step in the logit path.** If any op between the final hidden state and the logits (the output head, a norm, or a cast in the installed transformers version) returns float32 even when the model is float64, the two arms round to the nearest float32 and differ by about one ulp at the logit magnitude. This is the strongest candidate on structure. Its magnitude depends on the logit size, which the r7 report does not record. The toy gives about 4.8e-7 for logits of magnitude about 30 (ulp 1.9e-6). The ulp is 3.8e-6 for magnitudes 32-64 and 7.6e-6 for 64-128. The six observed floors (3.7e-6, 4.9e-6, 5.2e-6, 5.9e-6, 5.9e-6, 6.6e-6) are not clean integer multiples of one ulp at any single magnitude, so a single rounding step does not fit the numbers cleanly. C1 is possible but unverified. **Update (iteration 19, addendum below): the "10x below" figure was a magnitude artifact of the single |logit|~30 run; at |logit| ~ 10^2-10^3 the toy reproduces the observed order.**
- **C2. Float32 rotary tables.** Rotary cos/sin buffers computed in float32 at construction and cast to float64 by `model.to(float64)` carry a float32 rounding error that is shared by both arms. Shared errors cancel in `t - a` unless the permutation changes which positions use which table entry, so this is weaker than C1. **Toy-tested (iteration 4, `scripts/r7_rotary_float32_toy.py`): same-index lookup cancels (t-vs-a shift ~2e-14); a permuted lookup leaves a rounding residue of ~3.4e-7..4.7e-7 (seeds 0-2), about 10x below the r7 floor. C2 is therefore not a confirmed source. Hypothesis only.**
- **C3. A float32 cast of the intervention in the hook.** `_run_projection` adds `local_deltas[index].to(dtype=changed.dtype)`. The dtype is set by the model, so this is float64 in the float64 pass, but any float32 hop in the installed library would reintroduce C1.

## Toy result (`scripts/r7_logit_floor_toy.py`, seeds 0-2)

| config | logit effect error (seed 0 / 1 / 2) | module error | max abs logit | fp32 ulp at max logit |
|---|---|---|---|---|
| f64 (control) | 1.2e-11 / 1.3e-11 / 9.3e-12 | 9e-12..1e-11 | 26-32 | 1.9e-6 |
| f64_logit32 (C1) | 4.8e-7 / 4.8e-7 / 4.8e-7 | 9e-12..1e-11 | 26-32 | 1.9e-6 |
| f64_delta32 (C3-like) | 1.2e-11 / 1.3e-11 / 9.3e-12 | 9e-12..1e-11 | 26-32 | 1.9e-6 |
| f32 | 0 | 0 | 26-32 | 1.9e-6 |

Reading:

- The float64 control shows no floor. A float32 logit step (C1) produces an error of the right family (ulp scale), but about 4.8e-7, which is roughly one order of magnitude below the observed 3.7e-6..6.6e-6. **The toy does not reproduce the r7 magnitude**, so C1 is not confirmed at the real scale.
- The float32 delta cast (`f64_delta32`) does not change the result, consistent with the cancellation argument above.
- The toy's float32 module error is 0 because the toy's activations are small and its perturbation (1e-13) is below float32 resolution. It does not reproduce the r7 float32 module error (1e-5..6e-4, which needs the real activation scale of about 1.6e3). The toy therefore tests only the logit-path mechanism.

## Addendum (iteration 5): code-reading check of the arm construction

Question left open by iteration 4: do the treatment and aligned arms read the same rotary position indices? Answer from source, no model run:

- **Positions are identical across arms.** `_predictions` (v8s7 runner, line 134) runs both arms through `v3._model_forward_variant` with the same `group` and `inputs`. The only per-arm difference is `input_permutation`. Inside `_run_projection` (v3 runner, lines 482-484) that argument is applied with `changed.index_select(-1, perm)`, which permutes the **feature (channel) axis** of the projection input, not the sequence axis. Token positions, and therefore the rotary row each position reads, are the same for both arms.
- **The permuted module is an exact copy.** `_make_variant_modules` (v3 runner, lines 575-578) builds `raw.weight = treatment.weight[:, perm]`, an index copy with no arithmetic, and the saved file is that copy (line 746 reuses it as `mapping_aligned_permutation`). The aligned input is permuted with the same index.
- **Consequence for C2.** The "permuted rotary lookup" scenario in the C2 toy does not occur in this pipeline. C2 is therefore not reachable as a source of the t-vs-a gap. This is a code-reading result, not a numerical one.
- **Consequence for the rest.** Every float32-rounded quantity in the float64 pass is either shared by both arms (deltas, rotary buffers built in float32) or an exact copy (module weights). The only per-arm difference left is the summation order inside the projection matmul over a permuted channel axis. In float64 that is about 1e-15 relative, far below the 3.7e-6..6.6e-6 floor, so the floor is still not explained by anything in the repository's code.
- **Remaining candidate, owner-env only.** A float32 operation inside the installed transformers forward (a buffer not converted by `model.to(float64)`, or a kernel that upcasts an intermediate) would be per-arm only if it depends on the permuted channel order or on the module values. This cannot be checked here; it needs the dtype assertions listed below.

Status after iteration 5: C1 unconfirmed, C2 ruled out as reachable in this pipeline, C3 unchanged, floor source still unexplained.

## Addendum (iteration 19): logit-magnitude sweep and frozen-report pairing

Two new measurements, both read-only (`scripts/r7_logit_floor_sweep.py`, tested by `tests/test_r7_logit_floor_sweep.py`):

### A. The earlier "one order below" result was a magnitude artifact

The iteration-3 toy ran only at |logit| ~ 26-32. Sweeping the same C1 configuration over magnitudes:

| target logit | realized max abs logit (seed range) | fp32 ulp at max logit | max C1 error over seeds 0-2 | err/ulp | f64 control |
|---|---|---|---|---|---|
| 2 | 2.1 | 2.4e-7 | 3.0e-8 | 0.125 | 1.0e-12 |
| 8 | 8.5 | 9.5e-7 | 1.2e-7 | 0.125 | 4.0e-12 |
| 25 | 32.0 | 1.9e-6 | 4.8e-7 | 0.250 | 1.3e-11 |
| 64 | 67.8 | 7.6e-6 | 9.5e-7 | 0.125 | 3.2e-11 |
| 200 | 255.8 | 1.5e-5 | **3.8e-6** | 0.250 | 1.0e-10 |
| 500 | 529.6 | 6.1e-5 | **7.6e-6** | 0.125 | 2.5e-10 |
| 1500 | 1918.5 | 1.2e-4 | **6.1e-5** | 0.500 | 7.5e-10 |

At |logit| ~ 210-530 the toy C1 error is 3.8e-6..7.6e-6 — **the same order as the observed r7 floor (3.7e-6, 4.9e-6, 5.2e-6, 5.9e-6, 5.9e-6, 6.6e-6)**. The iteration-3 "about 10x below" conclusion assumed the r7 logits are of magnitude ~30; at larger magnitudes C1 reaches the observed band.

However, one caveat is now *measured*: err/ulp is **not** a stable fraction of ulp(max|logit|). It is lattice-valued (0.001 to 0.5 across magnitudes and seeds, err ∈ {integer multiple of ~ulp/16} because a difference of float32-rounded values lands on grid boundaries). So a single float32 rounding step cannot be *confirmed* from magnitudes alone: it explains the floor if and only if some r7 logit value sits near a float32 boundary in a magnitude band with ulp ≥ the floor, i.e. the deciding quantity is the real max |logit| distribution, which no frozen r7 report records.

Narrowed C1 condition (hypothesis): per the table, a floor F is reachable by C1 at magnitude M whenever ulp(M) exceeds F and some logit sits near a float32 boundary (the measured err/ulp reaches 0.5, not 1.0, at this sample count; the 1.9e-6-ulp band around |logit|~30 can therefore produce only up to ~9.5e-7, and the 7.6e-6-ulp band around |logit|~64 produces up to ~3.8e-6). The observed floors 3.7e-6..6.6e-6 are all comfortably produced only for max |logit| in the ~10^2..10^3 range, and the deciding quantity is the real max |logit| distribution, which no frozen r7 report records. This sharpens the owner-env check: record max |logit| per case; if it is below about 16-30, C1 is quantitatively dead; if it is in the 10^2 range, C1 is the default explanation.

### B. Frozen-report pairing: the floor does not track boundary-output magnitude

Pairing the six cases (read from `docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json`):

| case | max_abs_boundary_output | f64 logit floor | f32 logit floor | f64 module err |
|---|---|---|---|---|
| NN256 s0 | 1614 | 5.19e-6 | 4.77e-5 | 2.3e-13 |
| NN256 s1 | 30.0 | 5.88e-6 | 4.39e-5 | 9.9e-14 |
| NN256 s2 | 1956 | 6.62e-6 | 4.01e-5 | 9.1e-13 |
| NN512 s0 | 1592 | 5.91e-6 | 4.29e-5 | 4.5e-12 |
| NN512 s1 | 1684 | 3.72e-6 | 3.62e-5 | 4.3e-12 |
| NN512 s2 | 27.8 | 4.93e-6 | 4.48e-5 | 9.5e-14 |

Reading:

- The f64 floor is essentially *uncorrelated* with the boundary-output magnitude (the boundary-30 case has a higher floor than two boundary-1600 cases). So the floor is not driven by outlier activation magnitudes; if C1 holds, the deciding magnitude is that of the **final logits**, which is decorrelated from outlier residual dims (rogue dimensions typically do not dominate the unembed), and must be recorded in the owner env.
- The f32 logit floor is 3.6e-5..4.8e-5 across all six cases — flat within 1.4x while the boundary magnitude spans 70x — and the f64/f32 ratio is 1/8..1/11.5. Any complete account must produce both levels with one mechanism. C1's f64 level needs max |logit| ~ 10^2 (sweep row 200-500); the f32 level could come from the same rounding at larger |logit| plus float32 forward accumulation, but that combination is not toy-tested.

Status after iteration 19: C1 is consistent with the observed floor **iff** r7 max |logit| is of order 10^2 or larger; the toy reproduces a same-order floor at those magnitudes (3.8e-6..7.6e-6). It remains a hypothesis because the real max |logit| is unrecorded. C2 stays ruled out as unreachable. The owner-env decision procedure is unchanged and now has a concrete threshold.

## Addendum (iteration 20): the two-level account from one mechanism

Open item from iteration 19: can one mechanism produce BOTH the float64-pass floor (3.7e-6..6.6e-6) and the flat float32-pass floor (3.6e-5..4.8e-5, ratio 1/8..1/11.5)? New toy (`scripts/r7_two_level_toy.py`, tested by `tests/test_r7_two_level_toy.py`, 5 tests): one mechanism — **the logit path is float32 in both passes**.

- float64 pass: everything float64 with an r7-scale module difference (1e-13), logits cast to float32 once (C1).
- float32 pass: module and readout in float32 arithmetic; the only difference between the treatment and mapping-aligned arms is summation order in the permuted-channel float32 matmul (a mathematically identical sum in permuted order, exactly the real aligned-arm construction). No noise is injected and no constant is tuned.

Measured structure (seeds 0-15, common magnitudes):

| finding | toy result |
|---|---|
| float64-pass error / ulp(max logit) | a per-seed constant on a power-of-two lattice {0, 1/256 .. 1/2}, invariant across magnitudes 128..256 |
| float32-pass error / max logit | 1.6e-7..3.0e-7 (relative), same family as the real r7 float32 module relative error (7.6e-8..7.6e-7), which emerges from the toy without any tuned constant |
| float64 control (no float32 step) | 5e-11 (floor-free) |
| simultaneous in-band cells | at max logit ~160: e.g. seed 8 gives err_f64 = 3.815e-6 (inside 3.7e-6..6.6e-6) and err_f32 = 3.815e-5 (inside 3.6e-5..4.8e-5), ratio 10.0 (inside the observed 8..11.5) |

Reading: the single mechanism reproduces both observed floors at a common logit magnitude L ~ 130-200, with in-band ratios straddling the observed 1/8..1/11.5. So the two-level structure no longer requires two separate mechanisms: a float32 logit path produces the small ulp-lattice error in the float64 pass and the larger relative error in the float32 pass, with the ratio set by (float32 accumulation steps ~ 2e7 relative) vs (single rounding ~ 2 ulp).

This sharpens (not confirms) C1 into a testable conjunction, all decidable in the owner environment:
1. r7 max |logit| should be roughly 10^2..2x10^2 (if much smaller, the float64 floor is unreachable by one float32 cast; if much larger, the float32 floor overshoots).
2. err_f64/err_f32 should sit at ~1/8..1/12 — it does in the frozen reports (1/8..1/11.5).
3. The toy predicts err_f32 grows linearly with logit magnitude; the observed flat 3.6e-5..4.8e-5 band across cases therefore implies similar max |logit| across the six cases, which is also checkable once the owner env records it.

Status after iteration 20: C1-family mechanism reproduces both levels in the toy at a common magnitude; still hypothesis because the real max |logit| and logit dtype are unrecorded in any frozen r7 report.

## Addendum (iteration 21): joint six-case feasibility scan of the two-level account

Open item from iteration 20: the single-mechanism two-level account was demonstrated at ONE toy cell (seed 8, |logit|~160, inside the collective bands) and was never tested against all six frozen case pairs simultaneously. New measurement (`scripts/r7_joint_fit_scan.py`, tested by `tests/test_r7_joint_fit_scan.py`; read-only over the frozen reports):

- All 2,112 (seed 0-63, L = 64..320 step 8) cells were measured by running `scripts/r7_two_level_toy.py` directly. A pre-check of an extrapolated per-seed law (err_f64 = k(seed)·ulp_f32(L) with k magnitude-invariant) FAILS across binades (seed 0: 1.9e-6 at L~100 vs 3e-8 at L~128 — the rounding-boundary hit statistic is data-scale dependent, not seed-invariant), so no law is assumed anywhere in the result.
- Predeclared per-case match (fixed before running): |toy - case| / case <= 0.25 on BOTH the float64 and float32 floors.

| case | e64 (frozen) | e32 (frozen) | result at ±25% (primary) | result at ±50% (sensitivity) |
|---|---|---|---|---|
| NN256-case0 | 5.194e-6 | 4.768e-5 | NO match on grid | feasible, L 88-320 |
| NN256-case1 | 5.882e-6 | 4.387e-5 | NO match on grid | feasible, L 80-320 |
| NN256-case2 | 6.619e-6 | 4.005e-5 | feasible 16 cells, L 128-280 | feasible, L 80-304 |
| NN512-case0 | 5.910e-6 | 4.292e-5 | NO match on grid | feasible, L 80-320 |
| NN512-case1 | 3.718e-6 | 3.624e-5 | feasible 16 cells, L 88-216 | feasible, L 64-312 |
| NN512-case2 | 4.934e-6 | 4.482e-5 | feasible 18 cells, L 128-288 | feasible, L 80-320 |

- **Joint (all six simultaneously): common window is EMPTY at ±25%; at ±50% sensitivity it is L in [88, 288] (25 grid magnitudes).** Doubling the tolerance exactly rescues the fit, which shows the mechanism is constrained per-case by the discreteness of rounding-boundary draws, not by any band edge: the three unmatched pairs (f64/f32 ratios 0.109, 0.134, 0.138) sit in ratio gaps between the matched ones (0.103, 0.110, 0.165).
- Iteration 20's demonstration cell reproduced at the per-case level: seed 8 at L=160 matches NN512-case1 (+2.6% in e64, +5.3% in e32), NOT a generic band position (it is -42% in e64 vs NN256-case2).
- New grid statistic: err/ulp reaches 1.0 in some cells of the 2,112-cell grid, so iteration 19's "measured err/ulp reaches 0.5, not 1.0" was a smaller-sample observation, not a structural cap. (No cell has a zero float64 error; err_f32 spans 1.1e-5..9.2e-5 over this magnitude range.)

Reading (hypothesis): the two-level float32-logit-path account reproduces the collective band ranges of the six frozen cases but fails to reproduce the six case PAIRS jointly at the ±25% per-case slack; with the slack doubled to ±50% it fits all six at a common L in [88, 288]. C1 therefore stays unconfirmed and now carries a measured constraint sharper than before: the owner-env check is no longer just "record max |logit|" but "per-case max |logit| and rounding-boundary statistics must reproduce the six ratio values e64/e32 at a common magnitude" — if the real r7 logits are in [88, 288] and their boundary-hit draws had ~2x the toy's spread, or if the sample counts of the real cases exceed the toy's 256 samples, the per-case gaps can close; otherwise C1 is falsified as the sole source by the recorded scan.

## Addendum (iteration 22): sample-count probe — the gaps are NOT closable by n

Open item from iteration 21: "if the sample counts of the real cases exceed the toy's 256 samples, the per-case gaps can close". New measurement (`scripts/r7_samplecount_probe.py`, tested by `tests/test_r7_samplecount_probe.py`; read-only over the frozen reports, imports the iteration-21 match rule and frozen-case reader):

- Predeclared config (fixed before the recorded run): n in {256, 1024, 4096}, seeds 0..15, the same grid L = 64..320 step 8, the same ±25% primary / ±50% sensitivity match rule. 1,584 cells measured by running the two-level toy directly.

| metric | n=256 | n=1024 | n=4096 |
|---|---|---|---|
| cases matched at ±25% | 3/6 | 3/6 | 3/6 |
| cases matched (names) | NN256-case2, NN512-case1, NN512-case2 | identical | identical |
| ±25% common window | EMPTY | EMPTY | EMPTY |
| ±50% common window | L in [88, 264] | L in [104, 264] | L in [88, 248] |
| distinct err_f64/ulp levels over cells | 15 | 7 | 5 |
| err_f32 range | 1.1e-5..9.2e-5 | 1.1e-5..9.2e-5 | 1.3e-5..9.2e-5 |
| float64 control max | 1.6e-10 | 1.7e-10 | 1.4e-10 |

Raising n by 16x does not rescue any of the three unmatched cases. The recorded prediction's "more samples" half is REJECTED as measured.

- **New structural finding (measured, 528+32 cells across the grid and on 8 fresh seeds at n=4096): the toy's err_f64/ulp(max logit) statistic takes ONLY power-of-two values {.., 2^-4, 2^-3, .., 1.0} (0 = trivially exempt), never 1.5x a level.** Each cell's error is an integer multiple of the ulp of the boundary-straddling logit, and the max-over-samples statistic never mixes binades on this sample.
- Decision rule that follows, magnitude- and sample-INDEPENDENT: a frozen case is matchable at ±X% iff the relative distance of its e64 to the nearest power-of-two level is <= X%. Distances: NN512-case1 2.60%, NN256-case2 15.26%, NN512-case2 22.69% (matchable); NN256-case0 26.56%, NN512-case0 29.09%, NN256-case1 29.70% (NOT matchable at ±25% at any (seed, L, n)). This reproduces exactly the iteration-21 3/6 split, from the mantissa of e64 alone.

Reading (hypothesis): within the C1-as-toy-formulated mechanism, the per-case gaps are not closable by sampling, magnitude, or seed choice; they are closed only if the real logit path's float32 rounding sites produce a DENSER achievable set than pure power-of-two multiples of one ulp (e.g. several float32 steps with different ulp scales summing across binades), or, failing that, C1 is falsified as the sole source for the six cases jointly. The sharp owner-env prediction is unchanged in substance (record per-case max |logit| and logit dtype) and gains a decision rule: given a recorded max |logit| L*, the predicted e64 must equal an exact power-of-two multiple of ulp_f32(L*) within the tolerance — a direct, data-only test.

## Addendum (iteration 23): two-site variant — a hidden-state f32 cast does NOT densify the achievable set

Open item from iteration 22: the three unmatched cases (mantissa distances 26.6%, 29.1%, 29.7% from the nearest power-of-two level) can only be saved by float32 rounding sites whose contributions land BETWEEN the power-of-two lattice positions. New measurement (`scripts/r7_two_site_toy.py`, tested by `tests/test_r7_two_site_toy.py`; imports the toy's arm construction and the iteration-21 frozen-case reader so nothing drifts): mechanism variant M2 = C1 + one extra f32 cast of the hidden state h per arm, before the readout matmul — the ledger's own example of a second site at a different ulp scale. Arm differences are the recorded-scale f64 module differences (toy eps ~1e-13; r7 f64 module-effect 1e-13..4e-12), measured on the predeclared grid (seeds 0..31, L = 64..320 step 16; 544 cells).

| metric | single-site (C1, iteration 22 mechanism) | two-site (M2) |
|---|---|---|
| nonzero err_f64 cells | 544/544 | 153/544 |
| cells where M2 ERASES a nonzero C1 floor (quench) | n/a | 391/544 (72%) |
| cells nonzero where C1 was zero | n/a | 0/544 |
| err_f64 range (nonzero cells) | (same cells) | 1.16e-10 .. 1.53e-5 |
| hidden-site arm flips per cell | n/a | 0 in 306 cells, 1 in 170, 2 in 68 |
| mantissa of err_two_site/ulp(max logit) | power-of-two-valued (iteration 22) | STILL power-of-two-valued (all 153 nonzero cells) |
| min rel. distance to the three unmatched e64 | 0.2656 / 0.2970 / 0.2909 | identical: 0.2656 / 0.2970 / 0.2909 |
| cases matched at +-25% (f64 side) | 3/6 | 3/6, same three cases |

Three measured regularities, no model run:

1. QUENCHING: the hidden-state f32 cast erases the C1 f64 floor to exactly 0 in 391/544 cells and never creates an error where C1 had none. Mechanism: the arm hidden values differ by ~1e-12, far below the f32 ulp at hidden magnitudes (~1e-7), so both arms round h identically in 306 cells (zero flips) and the C1 logit-boundary flips that used to straddle are destroyed downstream of the cast.
2. Where the hidden site does flip (238 cells, 0-2 flips each), the arm-dependent contribution exists (two-site errs from 1.16e-10 up to 1.5e-5), but the max-over-samples statistic STILL lands only on power-of-two multiples of the logit ulp — the achievable mantissa set does not densify.
3. Consequently the reachable set of e64 values is unchanged: the three unmatched cases stay at exactly the iteration-22 mantissa distances, and no probed cell falls inside their +-25% window (asserted by test over the probe subset, measured over the full predeclared grid).

Reading (hypothesis): the ledger's proposed gap-filling variant is REJECTED as measured. A second float32 site fed by the f64-pass arm differences either quenches the floor (dominant) or, when it flips, still yields power-of-two-lattice max statistics, so it cannot explain the three unmatched cases and is moreover INCOMPATIBLE with r7's nonzero f64 floors when it quenches. Within arm-difference scales of the recorded size (1e-13..4e-12 relative), no float32 cast at hidden-state scale can fill the mantissa gaps; if the three unmatched cases are to be explained jointly with the three matched ones, the residual candidates are (a) arm differences at a much larger scale than the recorded module-effect numbers in the real r7 arms (unrecorded), or (b) a mechanism outside the single/two-site f32-cast family (e.g. summation-order effects in the f32 pass entering the f64 comparison, not the logit cast). The owner-env decision rule from iteration 22 is unchanged and now sharper: a recorded max |logit| L* for which e64 is NOT a near-power-of-two multiple of ulp_f32(L*) falsifies the cast-lattice family, not just the single-site form.

## What would settle it (owner environment, not run here)

1. Record max |logit| over the vocabulary and the logit-path dtype for the r7 SmolLM2-1.7B cases. If max |logit| is 64-128 or more, C1 predicts the observed floor in magnitude.
2. In the installed transformers version, check whether the output head or final norm returns float32 under a float64 model. Confirm by a dtype assertion on the logits tensor in the float64 pass.
3. If C1 is confirmed, a float64 logit path (no intermediate cast) should drive the logit floor to about 1e-12, which would also meet P2. This would be a diagnostic re-run, not a change to the frozen r6 outcome.

## Limits

- Toy only. It mirrors the structure of the comparison, not the model.
- Reading of code is by hand; the installed transformers version is not known to this repo and no dtype was checked.
- No change to any frozen packet, gate, constant, or r6/r7 result. The r6 outcome stays `FAIL`, and the r7 outcome stays `PARTIAL`.

## Addendum (iteration 24): summation-order sites densify the achievable set, but no joint account

Iteration 23 left candidate (b): summation-order effects in the f32 pass entering the f64 comparison. Measured directly (`scripts/r7_summ_order_toy.py`, tested by `tests/test_r7_summ_order_toy.py`; arm construction reuses the two-level toy's `_build` unchanged and the match rule reuses the iteration-21 reader, so nothing drifts): mechanism M4 = the readout reduction in the FLOAT64 pass is accumulated in float32 (every partial sum rounds to f32, no logit cast anywhere), with the aligned arm adding the same term values in permuted channel order. The hidden state and module matmuls stay exact f64, and the module arm difference stays at the recorded r7 scale (eps ~1e-13). Predeclared reading fixed before the run: LIVE iff (P1) some nonzero err_f64 has err_f64/ulp off the power-of-two lattice by >2%, (P2) at least one of the three unmatched cases is f64-matched at +-25% on the predeclared grid, (P3) the exact-f64 readout control stays <=1e-9. Grid: seeds 0..15, L in [8, 320] step 8 (640 cells, ~90 s); L starts below the prior grids because M4's expected errors track the logit magnitude.

| metric | before: cast-family achievable set (iterations 21-23) | after: M4 (this probe) |
|---|---|---|
| nonzero err_f64 cells | 153-544/544+ (variant-dependent; C1 always nonzero) | 640/640 |
| err_f64/ulp mantissa | power-of-two-valued (closed set; dense MATCHES impossible) | DENSE: 557/640 cells >2% off the nearest power-of-two (rel dist 0.0625..0.3125) |
| unmatched cases f64-matchable at +-25% | 0/3 (rel distances 26.6-29.7%, closed by nothing) | 3/3: best deviations 0.010 / 0.108 / 0.112 |
| all six f64-matchable at +-25% | 3/6 | 6/6 (best devs 0.010-0.153) |
| where the f64 matches sit | L 88-288 (joint, +-50%) | L = 8 only (grid edge); sensitivity L in [2,7] (8 seeds): window EMPTY, NN256-case2 unmatched |
| exact-f64 control | ~1e-11 | max 1.6e-10 (<= 1e-9 bound; P3 holds) |
| err_f64 range (nonzero) | up to 1.5e-5 (two-site) | 3.8e-6 .. 3.1e-4 |

Predeclared reading outcome: P1, P2, P3 all hold, so the summation-order family is LIVE as a mantissa densifier. This is the first measured mechanism whose achievable f64 set is dense, breaking the iteration-22 power-of-two-lattice ceiling that closed the three unmatched cases at +-25%.

REJECTED as a joint account, measured: the both-sides joint window is EMPTY on the full grid. At the only L where e64 matches (L~8), the unchanged f32 pass errs are 1.7e-6..2.4e-6 (seeds 0..3), 15-30x below the e32 band (3.6e-5..4.8e-5), and the in-cell ratio err_f64/err_f32 is 2.2-2.7, INVERTED by ~15-30x against r7's observed e64/e32 ~ 1/8..1/12. Both cast-based (iterations 21-23) and accumulation-based (this) probes match one side's scale at a magnitude where the other side's is far off; no tested mechanism produces both floors at one common magnitude.

New measured fact about the frozen data (read-only; recorded because any future lattice reading must respect it): all six frozen e32 are EXACT integer multiples of 2^-21 = 4.768e-7 (n = 76, 84, 90, 92, 94, 100, zero remainder); the six e64 are not (adjacent-integer fractional deviations 7-40%). Under either mechanism family the difference statistic spans logit-magnitude binades; e32 sitting exactly on the 2^-21 lattice implies the smallest-magnitude term involved in each f32-leg comparison chain lies in the [4,8) binade or finer multiples, and the mantissas 76..100 are integer counts of that spacing; e64 being OFF that lattice says the f64-leg differences involve a term of spacing FINER than 2^-21. Hypothesis label: binade membership of the real chains is unrecorded, so this constrains rather than identifies the mechanism.

Open questions / next iterations:
- Owner env (unchanged, now sharper): record per-case max |logit| and logit dtype. Under M4's family the e64-matching regime requires |logit| ~ 8 or below, while the f32 leg requires ~10^2; under C1 both live at ~10^2. Either recorded regime decides one family; the joint 1/8-ratio remains produced by NO tested mechanism.
- Pure-Python: a variant where the f64 pass accumulates in f32 but the f32 pass is the FULL-f32 arithmetic from a single magnitude regime is what the joint empty window measures next (e.g. reach e32 ~4e-5 at L~8 via a different f32-pass construction) — untested.

## Addendum (iteration 25): the joint window is NOT empty — chained-f32 + M4 produces both floors at one magnitude

Iteration 24's declared open experiment (the joint-empty-window measurement half) measured directly (`scripts/r7_chained_f32_toy.py`, tested by `tests/test_r7_chained_f32_toy.py`; the M4 f64 leg is reused unchanged by import, and the case reader / +-25% match rule are imported from the iteration-21 scan, so nothing drifts). Mechanism M5: the f32 leg is a CHAINED float32 pipeline — K=24 tanh-MLP layers of width d=512 (recorded num_hidden_layers=24; hidden 2048 scaled 1/4 for runtime; no tuned constants), the replaced module at chain layer 0 built exactly like the recorded construction (alignment = permuted module input + exact weight-row index copies; residual path canonical), both arms recomputed independently through the whole chain as the real two-pass diagnostic does. The chained leg's noise is the depth-effect the 64-wide single-readout f32 pass of iterations 20-24 lacked: relative logit noise rel ~ 3.4e-6..4.7e-6, roughly magnitude-flat (grid ratio 1.38), so e32 ~ rel x L reaches the e32 band (3.6e-5..4.8e-5) at L ~ 8-14 instead of only at L ~ 10^2.

Predeclared reading (fixed before the recorded run): Q1 = per-case f32 matches at +-25% on the grid; Q2 = a common JOINT window (both legs, all six cases, +-25%) exists; Q3 = chained rel noise magnitude-flat; Q4 = M4 exact-f64 readout control <= 1e-9; Q5 = exact 2^-21-lattice membership of chained e32 reported per magnitude, not asserted either way.

| metric | before (iteration 24: M4 + 64-wide single-readout f32 leg) | after (M5: M4 f64 leg + chained f32 leg) |
|---|---|---|
| f64-side per-case matches at +-25% | 6/6 (L ~ 8, grid edge) | 6/6, L 6..14 |
| f32-side per-case matches at +-25% | 0/6 at L ~ 8 (errs 15-30x below the band) | 6/6, L 6..14; best devs 0.000-0.042 |
| common window f64-only | L = 8 | L 8..10 |
| common window f32-only | EMPTY | L 8..12 |
| common window JOINT (both legs, all six, +-25%) | EMPTY | **NON-EMPTY: L in {8, 9}** |
| single cells jointly matching all six | none | none either (5/6 at most per cell); coverage is per-seed |
| err_f32 chained-leg range (64 cells) | 1.7e-6..2.4e-6 at L~8 | 6.9e-6..6.4e-5 over L 2..14 |
| Q3 rel noise | n/a | 3.37e-6..4.66e-6 over L 2..14 (flat, ratio 1.38) |
| Q4 M4 f64 control | <= 1e-9 bound | max 7.0e-12 (holds) |

Concrete joint cells (transcribed, tests assert the seed-0/L-8 membership): at L=8, seed 0 jointly matches five cases (all but NN512-case1, f32-only there); NN512-case1 is jointly matched at L=9 (seeds 2, 6). The iteration-24 recorded verdict ("both cast-based and accumulation-based probes match one side's scale at a magnitude where the other side's is far off; no tested mechanism produces both floors at one common magnitude") is now SUPERSEDED by measurement: the two-leg two-scale construction produces both floors jointly at +-25% at L ~ 8-9, with the 1/8..1/12 e64/e32 ratio emerging from DIFFERENT effective aggregation scales of the two legs (exactly the option the ledger open question named).

Two new recorded structural results:

1. Construction-level lattice rule (from the runner source, `_measure` + `_next_logits` in `replacement_loss_v3_runner.py`): an f32-pass statistic t_log - a_log is an f64 difference of two f32-valued logits, so EVERY such element is an exact dyadic; when all involved |logit| >= 4 it is an EXACT integer multiple of 2^-21 (any binade >= [4,8) suffices — ulp >= 2^-21 and every f32 value with |value| >= 4 is a multiple of 2^-21; f64 subtraction of the involved dyadics stays exact for realistic magnitudes). This GENERALIZES the iteration-24 binade reading ("smallest-magnitude term sits in [4,8)"): no [4,8) requirement, only |logit| >= 4. Measured in-toy: chained e32 at L >= 32 is exactly on the 2^-21 lattice (counts 304-2476 integer, zero deviation; the one off cell at L=32 measured a half-integer count = finest involved spacing 2^-22, consistent with its argmax element involving a [2,4) logit). Consistency with the frozen data: the six real e32s are all exact integer multiples of 2^-21, i.e. each case's f32-leg argmax element involved |logit| >= 4 in the f32 pass.

2. e64-leg discriminator sharpened: under C1 (any f32 quantization of the f64-pass logits before the f64 comparison), the f64-leg argmax element would obey the SAME rule — e64 an exact multiple of 2^-21 when its involved |logit| >= 4, off-lattice only via sub-4 logits. The six frozen e64 are all OFF the 2^-21 lattice (7-40%), which under C1 forces sub-4 logits at each of the six argmax elements, while the same argmax elements in the f32 pass (per the e32 lattice fact) had involved |logit| >= 4. Under the M4/M5 family (f32-accumulated sites, no logit cast) the f64-leg lattice is dense and no such constraint applies. The owner-env record of per-case max |logit| and logit dtype now discriminates not just the L-band (C1: ~10^2; M5: ~8-9) but also, jointly with the lattice facts, whether the two legs' argmax elements can differ in binade membership as C1 would require.

Open questions / next iterations:
- Owner env (unchanged, single decisive record): per-case max |logit| and logit dtype. C1 predicts ~10^2 jointly for both legs; M5 predicts ~8-9 with ratio 1/8..1/12 from leg-scale separation. Either value decides a family; recording max |logit| also enables applying the lattice rule to the real argmax elements (needs the actual involved logit values, i.e. a diagnostic re-run only).
- Pure-Python follow-up: the M5 construction's ratio check at finer magnitude resolution (is 1/8..1/12 attained per-case, not just band-level?), and width/depth sensitivity (d=1024/2048 runtime permitting) to bound how far the conclusion extends beyond d=512 — both untested.

## Platform note (added when landing)
`tests/test_r7_chained_f32_toy.py` passed on the x86-64 Linux VM where this analysis was produced, but 2 of its tests (`test_Q5_lattice_membership_measured`, `test_lattice_reading_fields`) FAIL on arm64 macOS (e.g. a measured value of 0.5 vs bound 1e-06). Float32 summation order is platform dependent, so the chained-float32 construction's measured lattice behavior is not platform-independent. The test was not loosened. Treat this analysis as hypothesis-generating until it reproduces across platforms.
