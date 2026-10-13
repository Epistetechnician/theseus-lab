# V39 first-MLP fit-factorial pilot

2026-10-07. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-fit-factorial-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Purpose and claim

V39 still fails the full behavior and causal-retention gates. The adaptation-
only follow-ups found large derivative error both on reconstructed training
directions and on fresh directions. They did not isolate capacity from
optimization budget. This new, narrowly scoped pilot crosses those two factors
on only the first GPT-2 MLP.

The fixed target is the reused `mrm8488/spanish-gpt2` checkpoint at revision
`fc396ce08c65ea334dcf541de7dc05cac89a29b2`, site 0. This is one reused guide,
not a fresh checkpoint or an architecture-held-out test. The candidate MLP
keeps the V39 JVP objective and V36 fitter. Hidden width is either two or four
times the 768-feature input. AdamW update count is either 1,024 or 4,096; all
other optimizer, batch, probe, normalization, and gradient-cap settings stay
fixed. Three candidate seeds share paired adaptation data, initialization RNG,
minibatch order, and JVP direction stream across the four cells.

## Frozen question and decision rule

The primary outcome is pooled nRMSE on the four JVP directions used by the
fitter at the adaptation and donor token inputs. Capacity and update-budget
effects are calculated as paired log-error contrasts by seed. A factor is
supported only when its geometric-mean error ratio is at most 0.80 and every
seed-level paired contrast favors it. The interaction is reported. Three
candidate seeds from one checkpoint are a descriptive pilot; they do not
support population-level significance or generalization to other guides.

All four cells for all three seeds are evaluated on fresh, disjoint challenge
fixtures. The V39 thresholds are copied without change: output nRMSE at most
0.10, top-1 agreement at least 0.95, mean NLL increase at most 0.03, activation
nRMSE at most 0.15, actual rescue and rescue-prediction nRMSE at most 0.10,
and causal-effect resolved fraction at least 0.90, median relative error at
most 0.20, p90 at most 0.50, and median cosine at least 0.90. Zero, wrong-seed,
and capacity-matched random modules must each have joint-effect median error at
least 0.10 above the treatment. The guide no-op must reproduce the oracle
exactly. A factorial cell qualifies only if all gates and controls pass in all
four strata for all three seeds.

The protocol uses 128 adaptation and 128 donor prompts per seed, plus 32 fresh
recipient/donor pairs in each of four strata per seed. Fixture text and token
IDs are generated only after source freeze, remain in external custody, and
must be disjoint from inherited exclusion hashes, all V39 training and
challenge fixtures, and every new partition. Held-out entropy is drawn only
after candidate weights and traces are frozen. Candidates and controls are
predicted and digest-sealed before the guide oracle is evaluated.

## Execution and governance

The primary and replica rebuild all twelve candidates in separate isolated
V36-compatible CPU runtimes. The replica must match candidate weights and
traces before reading the primary challenge entropy, then match sealed
predictions before guide-oracle evaluation. A separate NumPy closeout
recomputes the training-probe formulas, factorial contrasts, held-out scores,
and control margins from persisted arrays.

Per-role caps: eight hours, 6 GiB peak RSS, 24 GiB study artifacts, and 25 GiB
free-space reserve. Paid execution cap is $0. Energy and full economic cost
remain unmeasured. Any custody mismatch, overlap, nonfinite value, resource
cap, execution error, or reproduction mismatch is `INCONCLUSIVE`; do not retry
the same artifact root or retune this identity.

Claim ceiling `LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_ONLY`.
No conclusion extends to full-depth replacement, architecture-general
validity, novel mechanism identity, consciousness, human-alignment equivalence,
independent science, or production. Reviewer status is `NOT_SUBMITTED`;
production remains a separate authorization gate.
