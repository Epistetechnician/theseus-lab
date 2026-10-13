# V36: whole-MLP pretrained transfer qualification

State slices: `theseus-lab-sandbox-v1`,
`theseus-pretrained-whole-block-transfer-v36`,
`global-experiment-assurance-lifecycle-v1`. Revision
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus source packet.
Prepared 2026-10-05. Parent: [V35](replacement-loss-v35-results.md),
packet `2b5a23c1a7c1fd5979b384f559ae152cd6f4febd8c5f2d455db05a4abd569656`.

## End state, hypothesis, and claim ceiling

Implement, qualify, and freeze a complete offline capture/fit/seal/evaluate/
reproduce runner before fresh checkpoint selection. Actual execution requires
verified external custody and the frozen local runtime. Missing custody is
`CUSTODY_UNAVAILABLE`, never a scientific failure or completed replication.
Source freeze may precede custody verification; it does not authorize intake
against missing drives or substitute for verifying the parent artifacts.

H1: independently initialized whole MLP replacements fitted with endpoint,
finite-input-effect, and directional derivative supervision retain measured
decoder behavior, interface intervention effects, and activation restoration
through two successive replacements in nine architecture/seed cases.
The endpoint arm tests the added derivative information. A 20% paired reduction
in joint-effect p90 versus endpoints is a descriptive effect of interest;
selection/PASS requires absolute all-case retention, not this comparison.

The derivatives are local module Jacobian-vector products at four random
directions per adaptation row, not V35's full prefix Jacobians. This is a new
predeclared transfer hypothesis, not an identical transplant of V35's loss.
Matching probes is not a guarantee of finite intervention generalization.

Claim ceiling: `LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY`.
No claims of hidden-neuron identity, mapping-aligned equivalence, all
architectures, novelty priority, subjective experience, human-equivalent
alignment, independent science, or production authorization. Attention,
embeddings, norms, and the language-model head remain original and frozen.

## Models, selection, custody, and freshness

Exactly one fresh public base checkpoint per family: OPT/ReLU, GPT-2/GELU,
Llama/SiLU-gated. Exactly three replacement/corpus seeds: 36101/36102/36103.
These are replacement seeds on one checkpoint per family, not nine independently
pretrained checkpoints. Sources of variance: checkpoint, corpus, replacement
initialization/minibatches, and sealed challenge entropy. Rows and stages
are dependent observations; no powered superiority or population claim.

Checkpoints must be selected after source freeze and bound to a later verified
historical intake. The intake snapshots predecessor JSON manifests/records,
model identities, text/token hashes, and prior artifact digests once. Its exact
file list is verified thereafter; the source checkpoint registry is also bound
into the source freeze. The history never expands to include V36 or its
replica. Parent V35 primary/replica results, every manifested artifact, and
reconciliation/closeout digests must match the reported identities.

No previously recorded model identity or weight-file digest may be reused.
The external checkpoint manifest attests base-model status, lawful local use,
selection after freeze, and absence from protocol design; these are custodian
attestations, not independent authority. It binds every local checkpoint file,
the source packet, and intake digests. Paths must remain inside V36 custody;
symlink escapes, unmanifested files, custom Python, and pickle weights fail.
Only safetensors weights and local Transformers loaders are supported.
Limits: 250M parameters, hidden width 8–1024, ≥4 layers, vocabulary≤65,536,
1.1 GiB files/model.
The runtime's supported family and activation are checked before fitting.

Both actual external volumes must be mounted; their existing custody must be
present. Never create replacement directories under absent `/Volumes` mounts.
Artifacts: `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36`.
Runtimes: `/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v36`.
Instantiated prompts, token IDs, captures, model assets, candidate weights,
entropy, predictions, oracle outcomes, and detailed receipts remain external.
Repo contents: code, generator rules, tests, lock, protocol, source freeze,
and scalar readiness/results. No downloads, provider calls, cloud workers,
review submission, publication, commit, or merge are part of this packet.

## Visible target qualification and corpus

Use 64 new visible prompts per family, separate from adaptation/assessment.
Capture final-token outputs of all MLPs; select the largest mean squared
output in each depth half, breaking ties by lower layer index. Remove those
two modules individually. Each must change next-token argmax on ≥20% of
visible prompts and produce ≥.10 logit nRMSE. This establishes visible next-token
sensitivity, not ground-truth task competence. If any family fails, stop
`NO_QUALIFIED_TARGET` before replacement fitting or challenge generation.
Do not substitute another checkpoint/site or relax qualification.

Each family/seed uses 128 adaptation and 128 donor prompts. Capture final-token
MLP inputs in the candidate context with earlier replacements frozen. Fit the
frozen original target MLP as a function of these inputs; only adaptation
module outputs and derivatives are teacher targets. No challenge targets.

Procedural prompt rules are frozen in `replacement_loss_v36_fixtures.py`.
Fresh record tags namespace protocol/family/seed/partition. All instantiated
texts and full token sequences must be globally disjoint from intake and
other partitions. Tokenization length 2–96; no truncation, resampling, or retry
to repair overlap or length failures. Shared causal prefixes are not claimed
to be globally independent activation rows. Semantic near-duplicates remain
a limitation of the procedural corpus.

After every candidate freezes, OS entropy generates 32 base/donor prompt pairs
in each of four unseen strata: ordinary, input composition, interchange
composition, and adversarial. The latter three use distinct procedural
language compositions. All arms use identical paired fixtures. Donor draws
use the declared seed+1000 namespace; masks use the sealed entropy.

## Replacement and fixed training

Replace both projections and activation in OPT; replace the complete GPT-2
MLP; replace all three projections and gate in Llama. New module: hidden→2hidden
→hidden; ReLU for OPT, exact GELU otherwise. All trainable parameters are
independently initialized. Output mean/coordinate scale are fitted from
adaptation teacher outputs, not copied from original weights. Verify no
original selected-MLP parameter remains in the installed module graph.
Modules are frozen after fitting; prior replacements stay frozen while the
next module is fitted. Retained decoder weights never receive optimizer updates.

Each module/arm: 1024 AdamW updates, batch min(64, adaptation rows), learning
rate .001, decay .0001, combined norm clip 1. Endpoints share initialization,
corpus, single-coordinate interchange masks, probe bank, and minibatches.
No equal-FLOP claim: the derivative arm adds candidate derivative arithmetic.

Base loss: mean normalized MSE at x and donor plus normalized MSE of the
finite single-coordinate module-input effect. Global target mean-square
normalization floor 1e-8. Four independent Rademacher directions per row,
each divided by sqrt(hidden width), are fixed from the preparation seed.
Derivative loss: mean normalized MSE of teacher/student JVPs at both endpoints.
Endpoint arm adds zero auxiliary loss. Norms are detached float64 sums;
coefficient=min(1, .5*base_norm/max(aux_norm,1e-12)). Require weighted auxiliary
norm≤.5*base_norm+1e-8 on every update; retain both norms/coefficient traces.

[PyTorch JVP](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html)
defines a directional Jacobian product; compatibility is separately tested
against autograd in pinned torch 2.6.0. Full-MLP shape adapters follow the
locally inspected and pinned [Transformers OPT source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/opt/modeling_opt.py).
Jacobian matching has prior art; no novelty claim arises from derivative use.

## Prediction seal, interventions, rescue, and gates

Save both stage snapshots. Freeze all candidate/corpus/trace/module-lineage
digests before drawing challenge entropy. Predict decoder logits, language
NLL, native MLP activations, and input-intervention effects before guide oracle
assessment. Interventions replace 1/2/4/8 coordinates of the final-token MLP
input using each model's own unperturbed base/donor inputs. Adversarial
coordinates extrapolate by 1.4*(donor−base). Joint tests patch every active
site simultaneously from the unperturbed factual states; they are unseen
finite interventions, not infinitesimal probe checks. Native interface
coordinates have equal dimensions; replaced hidden-neuron coordinates do not.

For each active site, actual activation restoration runs the original complete
guide with candidate MLP output injected at that site's final token. Candidate
factual logits are the prebound restoration prediction. This is a declared
proxy and can fail when other replacements/prefixes differ; it is not an oracle
call disguised as prediction. First-stage joint reference contains only its
one active site; second-stage reference patches both sites.

Every method/stage/stratum requires logit nRMSE≤.10; next-token top1
agreement≥.95; mean language NLL increase≤.03 nats/token; each individual and
joint effect resolved fraction≥.90 at reference norm≥1e-4, median relative
error≤.20, p90≤.50, median cosine≥.90; activation nRMSE≤.15, actual restoration
nRMSE≤.10, prebound restoration-prediction nRMSE≤.10. These language guards
replace synthetic truth/sign guards explicitly; they are not truth-task scores.

Guide/no-op positive control must pass using an actual decoder rerun with
the original MLPs reinstalled and actual output restoration. This control is
assessed after candidate prediction sealing. Random, zero-module, and swapped-module
controls are predicted and sealed before guide outcomes. In every second-stage
stratum each negative joint-effect median error must exceed treatment by≥.10.
All-stage failures remain disqualifying. Overall `PASS` requires all nine
derivative-arm cases. Report endpoint results separately; no post hoc arm
selection, CKA comparison, symmetry sweep, or threshold tuning.

## Reproduction, resources, retention, and stop conditions

Fresh isolated Python 3.12.13 environments, exact V36 lock, one-thread CPU,
float32, eager attention, deterministic algorithms. Replica reconstructs
candidate bindings before primary entropy reveal, and matches sealed prediction
digests before oracle access. Require equal prepared arrays/weights/traces,
fixtures, predictions, outcomes, and metrics; toolchain/platform equality and
distinct prefixes/site-packages. Runtime receipts bind the interpreter and
installed dependency file digests, excluding bytecode caches and external
entrypoint shims. Shared host/system libraries are not independently reproduced.
Same-owner reproduction can yield only
`REPRODUCED_LOCAL`, never independent review or external blind replication.

Per runtime: 24-hour wall cap, 6 GiB peak RSS, 24 GiB artifacts; resource checks
at decoder-query boundaries and every 16 optimizer updates. These are
cooperative boundaries, not hard process preemption. Monetary cap $0; energy
and machine amortization unmeasured. Retain manifests and final byte/RSS/wall
receipts; no missing outcome implies zero cost. Full-run resource performance
is unknown until real execution; small engineering fixtures are not a speed
or memory estimate for pretrained models. The fixed prediction/oracle/control/
rescue payload (float32 logits/activations and float64 NLL) is
1024*(27*vocabulary+11*hidden_width+12) bytes per case,
before NPZ metadata, model weights, captures, and training traces. At the
declared vocabulary/width bounds this alone is about 15.28 GiB across nine
cases. The 24 GiB artifact cap accommodates those full-vocabulary arrays;
an 8 GiB cap would be insufficient for typical 50k-vocabulary checkpoints.
Before fitting, require measured free space≥the artifact cap+1 GiB. This is a
conservative check, not a hard disk reservation; other processes can consume
space later. Resource receipts retain available free bytes. Full-run resource
performance remains unmeasured until actual execution.

Any source, model, custody, runtime, disjointness, numerical, resource, or
replication fault is `INCONCLUSIVE`; preserve the terminal root and error
receipt. Never retry that root or repair frozen bytes. A visible qualification
miss stops before learning. An actual treatment miss is literal `FAIL`.
Source/readiness work can continue while custody is unavailable; real execution
cannot. Reviewer `NOT_SUBMITTED`; independent science, human calibration,
consciousness research, and production remain separate authority/program gates.

The separate closeout utility verifies persisted artifact/capture/weight/
oracle/rescue and prediction-seal digests, gradient caps, and independently
recomputes every native numeric cell, control sensitivity, and aggregate verdict.
It requires all declared reconciliation keys and distinct locked runtimes.
This validates the saved numbers; it does not independently
reimplement the decoder or prove correctness of shared capture semantics.

Commands after source freeze and verified drive reconnection:

```sh
PYTHONDONTWRITEBYTECODE=1 <locked-python> replacement_loss_v36_runner.py verify
PYTHONDONTWRITEBYTECODE=1 <locked-python> replacement_loss_v36_runner.py intake
PYTHONDONTWRITEBYTECODE=1 <primary-python> replacement_loss_v36_runner.py run --checkpoint-manifest <external-checkpoints.json> --role primary
PYTHONDONTWRITEBYTECODE=1 <replica-python> replacement_loss_v36_runner.py run --checkpoint-manifest <external-checkpoints.json> --role replica
PYTHONDONTWRITEBYTECODE=1 <locked-python> replacement_loss_v36_closeout.py
```

The external checkpoint JSON has `packet_sha256`, `intake_sha256`, and three
`checkpoints` entries. Each supplies `family`, `model_id`, full `revision`,
relative `local_path`, the four selection/rights boolean attestations, and
`files` rows containing relative `path`, `bytes`, `sha256`. Gate failures never
trigger an automatic acquisition or a second target selection.
