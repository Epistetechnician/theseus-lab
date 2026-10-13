# V34 adaptation-only derivative and endpoint diagnosis

State slices: `theseus-lab-sandbox-v1`,
`theseus-v34-adaptation-jacobian-audit-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus frozen source.
Monetary cap $0. Preserve V34's reproduced failure and every predecessor.
This diagnostic does not reopen a failed training identity.

## Scope and hypotheses

Inspect V34's three existing ReLU guides, three arms, three saved stages, and
all active sites. Use only the 2048 adaptation/donor pairs and their original
single-coordinate masks per guide, plus exact frozen model states. No new
training, fitting, optimizer update, or challenge outcome evaluation. A frozen
33-file input manifest binds all model/preparation files before diagnostic
arrays are loaded. Parent receipt/manifest integrity may hash challenge bytes;
no challenge arrays, predictions, or heldout target fields are queried for the
diagnostic or continuation decision.

H1: weak intervention errors are associated with coordinate secant or local
Jacobian mismatch despite low pointwise activation error. H2: shifted bases
can change ReLU regions in the shared suffix, so preserving pointwise states
alone does not establish finite intervention behavior. Association and exact
error decomposition are observations, not proof of a training failure's cause.
Three guides are the units; 2048 correlated adaptation rows and repeated
snapshots are not independent replicates.

## Exact decomposition and Jacobians

Let guide/candidate base activations be g/c, donor activations gd/cd, coordinate
mask P, guide suffix D, candidate suffix C. Let dg=P(gd−g), dc=P(cd−c).
Reference effect R=D(patch(g,gd))−D(g), candidate E=C(patch(c,cd))−C(c).
Define hybrid H=D(g+dc)−D(g) and shared candidate S=D(patch(c,cd))−D(c).
Then E−R=(H−R)+(S−H)+(E−S): delta, base, and suffix terms respectively.
Use exact `where` patches for reference/shared endpoints, and record floating
roundoff from the hybrid. Require maximum absolute decomposition residual
≤1e-5 over every row/cell. At block one, stem, suffix block, and head must
equal the guide exactly; suffix error must be ≤1e-6. Identity-model tests
and deliberately changed suffixes are engineering controls.

Measure exact local prefix Jacobians d(state_i)/d(input), including the stem
and every upstream block. ReLU uses zero subgradient at zero; GELU uses the
exact erf derivative; SiLU-gated uses the product rule. Verify all three
adapter formulas against float64 autograd on constructed engineering inputs
before scientific readout. These tests do not establish pretrained coverage.

For the ReLU suffix at site zero, count sign disagreements between guide
suffix preactivations at guide and candidate bases, and sign crossings
between guide factual/intervened endpoints. Compare local linear prediction
J_D(g)dg with the actual finite guide intervention. At site one, the suffix
is linear and boundary measures are zero. Save per-row vectors, Jacobians,
component errors, norms, and boundary measures only in external custody.

## Predeclared continuation rule

Reference norms ≥1e-4 are resolved. Sort resolved rows by norm, stable ties;
the bottom ceil(20%) are weak, remaining rows are the comparison group.
Compute median relative effect errors, their difference, full resolved p90,
Spearman correlation between relative effect error and masked coordinate
secant error, and prefix-Jacobian normalized RMSE. Spearman uses average tie
ranks, and constant inputs return unavailable rather than a fabricated zero.
Also report boundary correlation, activation RMSE at both endpoints, component
RMS, and finite-effect linearization discrepancy. No tuning after readout.

Primary diagnostic cells are global-normalized block-one site-zero for the
three guides. A guide supports a new endpoint/derivative hypothesis only if:

- at least .90 of effects resolve;
- weak median relative effect error exceeds other rows by at least .10; and
- secant/error Spearman ≥.30 OR prefix-Jacobian normalized RMSE ≥.15.

All decomposition controls must pass, and at least two distinct guides must
support the hypothesis. Outcome then is
`DIAGNOSTIC_SUPPORTS_ENDPOINT_OR_DERIVATIVE_HYPOTHESIS`; otherwise
`DIAGNOSTIC_DOES_NOT_SUPPORT_CONTINUATION`. This is a development decision,
not scientific acceptance. The thresholds require a visible association and
material derivative discrepancy; they are descriptive, not powered inference.
Only after this decision reproduces may a fresh protocol test matched donor
endpoint, secant, and Jacobian supervision. That protocol must freeze fresh
data, loss rules, updates, selectors, native gates, and stops before training.
A supportive diagnostic does not guarantee the new training method succeeds.

Derivative supervision is established prior art in
[Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html) and
[Sobolev training](https://arxiv.org/abs/1706.04859). No novelty/priority claim.

## Budgets, custody, reproduction, and completion

Budgets per diagnostic runtime: 300 seconds, 4 GiB RSS, 1 GiB artifacts,
deterministic one-thread CPU, float32 model calculations, exact V34 package
lock and Python 3.12.13. Monetary cost $0; energy and amortization unmeasured.
Use V34's two existing isolated runtime prefixes with unchanged dependencies.
No synthetic diagnostic data are redrawn; use the identical frozen inputs.
Retain failures and partial receipts as `INCONCLUSIVE`, never retry a root.

Primary and replica reconstruct the diagnosis separately. Require identical
input, per-row array hashes, scalar metrics, decisions, and archived source
manifests, equal toolchains, and distinct runtime prefixes/site-packages.
Closeout verifies manifest bytes/digests and parent source unchanged. This can
support same-owner `REPRODUCED_LOCAL`, not external blind replication.

Raw artifacts stay under
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-v34-derivative-audit`.
Repo receives code, tests, protocol, input/freeze digests, and aggregate reports.
Reviewer `NOT_SUBMITTED`; claim ceiling
`ADAPTATION_ONLY_ENDPOINT_AND_DERIVATIVE_DIAGNOSIS`. Nonclaims: qualifying
learned replacement, heldout predictive success, all architectures, pretrained
transfer, causal attribution of training failure, metaphysical identity,
consciousness, alignment, novelty, independent science, production.
Completing and reproducing either diagnostic decision completes this scope;
an unsupported decision stops continuation. External authority, human
calibration, and production authorization remain separate future gates.
