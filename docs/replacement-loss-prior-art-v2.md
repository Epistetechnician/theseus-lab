# Replacement-loss prior-art follow-up v2

As of 2026-10-02.

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-prior-art-v2`, and
`global-experiment-assurance-lifecycle-v1`.

Repository revision reviewed: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
This is a targeted primary-source follow-up to
[prior-art v1](replacement-loss-prior-art-v1.md) and
[comparison-readiness v1](replacement-loss-not-comparison-readiness-v1.md).
It changes no frozen experiment packet, result, checkpoint, or external-custody
artifact. The prior-art search remains non-systematic and cannot establish
novelty priority.

## Material overlap found

The earlier comparison note treated held-out causal-intervention prediction as
a possible differentiator from representation alignment and task retention.
That framing is incomplete: interchange-intervention training (IIT) already
trains neural models to match counterfactual behavior under interventions in an
aligned high-level causal model. Its stated zero-loss result is a causal
abstraction guarantee for that model relationship. This is substantial
methodological overlap with causal-effect supervision and evaluation, though
it is not itself a component-replacement procedure.
[IIT paper, arXiv:2112.00826](https://arxiv.org/abs/2112.00826).

A closer language-model example is *Type-level Interchange Intervention
Training* (Findings of ACL 2023). It adapts IIT to learn character-level
causal structure inside subword language models and reports robustness on
unseen token sequences. This means neither intervention-supervised learning
nor generalization to unseen input sequences can be presented as the novel
part by themselves. The paper does not study replacing a module inside an
already-trained guide model.
[ACL Anthology paper](https://aclanthology.org/2023.findings-acl.770/).

*Causal Distillation for Language Models* (NAACL 2022) is a still closer
training comparator: it augments task training and hidden-state imitation
with IIT so a student imitates the teacher's causal computation. The remaining
distinction is the study target—causal student distillation versus replacement
inside an already-trained guide—not the general idea of transferring causal
behavior to an independently trained model.
[Causal Distillation paper](https://aclanthology.org/2022.naacl-main.318/).

An especially direct overlap surfaced in a wider targeted query:
*Causal Mechanism Reduction: Mechanism Replacement for Neural Network Pruning
and Abstraction* (arXiv v2, 2026) explicitly treats a trained feedforward
network as a deterministic SCM, replaces internal variables with constants or
affine functions of retained variables, compiles the reduced network, and
checks interchange-intervention fidelity. Its replacement class and reduction
goal are narrower than a newly learned nonlinear component inserted into a
trained guide, but “causal mechanism replacement with interventional
verification” is already an explicit research direction in a public preprint. It is a
required comparator and narrows any possible novelty claim further.
[CMR arXiv record](https://arxiv.org/abs/2602.24266),
[CMR paper text](https://arxiv.org/html/2602.24266).

*Structured Pruning for Efficient Generative Pre-trained Language Models*
(Findings of ACL 2023) also combines structural removal of transformer FFN
neurons, heads, and hidden dimensions with causal distillation during
fine-tuning, including GPT-2 and encoder-decoder models. This is not an
independently learned nonlinear replacement module, but it overlaps with
structural change plus causal-fidelity training in language models.
[ACL Anthology paper](https://aclanthology.org/2023.findings-acl.692/).

The authors' public [char-IIT repository](https://github.com/explanare/char-iit)
states that it contains data and models for the ACL 2023 paper and links a
Colab reproduction. The inspected `main` ref resolved read-only to commit
`e602f040e2f9bbb2e01acd5199bbc88f68d03969`. Its notebook implements character
annotations and interchange-intervention evaluation, and loads `t5-small` or
`google/byt5-small`; either checkpoint family is therefore known during this
protocol design and cannot serve as an architecture-held-out checkpoint for
this study. The notebook also uses unpinned dependency installation and model
IDs, so it is not itself a frozen reproduction packet.

The repository's root `LICENSE` is MIT, but its Git tree lists separate
`data.tgz` (24,372,040 bytes) and `tokenizers.tgz` (833,999 bytes) without
artifact-specific license metadata visible in the inspected tree. The archive
contents, source-dataset terms, exact split hashes, and custody requirements
remain unaudited. No archive or model was downloaded or imported. The source
is a benchmark lead, not an approved dataset.

*Network of Theseus* is listed by its official arXiv record as a preprint
submitted on 2025-12-03. The preprint progressively replaces guide components,
aligns the partially converted network by representational similarity, and
then fine-tunes the completed target on its downstream task; its experiments
include GPT-2-to-RNN language-model conversion. The official arXiv record does
not by itself verify the prior note's description of the work as a published
2026 paper, so use “arXiv preprint” unless publication metadata is independently
confirmed.
[NoT arXiv record](https://arxiv.org/abs/2512.04198),
[NoT paper text](https://arxiv.org/html/2512.04198).

## Relevant local result already in the record

V17 already trained whole-MLP OPT replacements with adaptation-only
finite-intervention supervision and compared them with equal-budget
pointwise-only candidates. Its frozen scientific outcome is `FAIL`, while a
same-owner second runtime reproduced the artifacts and verdict as
`REPRODUCED_LOCAL`. Next-logit effect cosine and relative-error gates failed
in all nine seed-by-stratum cases; the causal-versus-pointwise improvement
gates passed together in only three of nine. V17 remains valid negative
evidence and must not be rerun or retuned as if it were a new confirmation.
[V17 protocol](replacement-loss-v17-causal-supervision.md),
[V17 results](replacement-loss-v17-results.md).

V17 is an intervention-supervised replacement baseline, but its packet does
not specify an aligned high-level structural causal model and IIT-style
interchange interventions. It therefore does not count as an IIT reproduction.
The practical next question is whether a faithful causal-abstraction baseline
can be defined for the replacement task at all; if not, report that limitation
and compare explicitly named intervention-supervised objectives without
calling them IIT.

V29/V30 are also not IIT qualifications. Their source functions are synthetic
nonlinear modules, and their targets are local finite differences under
perturbed input vectors; the packets do not define an SCM with aligned
variables or base/source interchange interventions. V29 stopped
`INCONCLUSIVE` before training. V30 completed a same-owner clean-runtime replay
but scientifically `FAIL`ed eight of nine cases. These remain valid negative
methods results and must not be tuned or relabeled as causal-abstraction
evidence. [V29 results](replacement-loss-v29-results.md),
[V30 protocol](replacement-loss-v30-float32-recovery.md),
[V30 results](replacement-loss-v30-results.md).

## Narrowed claim boundary

The following are already covered by the identified prior work and are not
defensible novelty claims on their own:

- progressively replacing components while retaining task behavior;
- replacing internal mechanisms with constants or affine functions and
  evaluating the resulting network as an approximate causal abstraction;
- training neural representations to match counterfactual intervention
  outcomes under an explicit causal model; and
- distilling a teacher's causal computation into a student or fine-tuning a
  structurally pruned model with causal distillation; and
- evaluating generalization on unseen input sequences or compositions.

A narrower candidate remains worth testing, but is not established as novel:

> Inside a fixed trained guide model, independently construct and fit a
> replacement module using a declared adaptation split, freeze it, bind its
> predictions, and measure whether it predicts causal effects for held-out
> interventions while preserving the guide's behavior.

The remaining possible distinction is a **newly fitted nonlinear replacement
module inserted into a trained guide, evaluated by predictions bound before
held-out interventions**. This targeted scan has not established that
combination as novel. The local V17 experiment is relevant but failed its
absolute causal-retention gates, so it supplies no positive evidence for a
novel successful method.

## Required comparison before a novelty claim

If a study proceeds, use one frozen guide checkpoint, replacement schedule,
adaptation corpus, held-out intervention split, and resource budget across
methods. Include at least:

1. **Naive and structural controls:** unchanged guide, naive replacement,
   wrong-module replacement, random replacement, and no-op intervention.
2. **NoT-style arm:** staged representational alignment during full-target
   conversion, followed by the predeclared downstream task fine-tuning needed
   to match the published procedure.
3. **Causal-distillation arm:** task loss plus teacher-state imitation and
   IIT-style causal objective, with its aligned causal variables and
   base/source interchange examples specified explicitly. Otherwise label it
   an `intervention-supervised baseline`, not a Causal Distillation or IIT
   reproduction.
4. **CMR arm:** constant and affine internal-mechanism replacements scored by
   the predeclared interventional-risk/CMR-Logit procedure, with its fixed
   alignment map and compiled network verified under the same interchange
   challenges. Do not describe this as a learned nonlinear module replacement.
5. **Local finite-effect arm:** the V17-style adaptation-only effect objective
   as a separately named comparator, under a new matched packet; preserve its
   existing `FAIL` unchanged.
6. **Candidate arm:** the replacement-specific method under study, with its
   objective and permitted adaptation data fixed before held-out challenge
   generation. Bind the candidate weights and predictions before scoring.

Score ordinary task behavior, representation similarity, activation rescue,
module-boundary intervention effects, and downstream intervention effects as
separate outcomes. Split by intervention types or compositions, not just by
random examples, and do not tune against held-out outcomes. Match optimization
and compute budgets as closely as the methods permit, and report any mismatch.

The first feasible study may make only a claim about its one checkpoint and
conversion. Do not call the result architecture-general, independently
validated, or novel from a local head-to-head alone. A clean second runtime can
support `REPRODUCED_LOCAL`; it does not supply external acceptance.

## Current disposition

- Targeted prior-art status: `EXPANDED_TARGETED_SCAN_COMPLETE`;
  novelty: `NOT_ESTABLISHED`.
- Local NoT comparator: `MODULE_LEVEL_ONLY`; it is not a full NoT reproduction.
- IIT/Causal Distillation comparator: `NOT_IMPLEMENTED`; V17 is finite-effect
  supervision, while V29/V30 are synthetic local finite-difference tests. None
  establishes an explicit high-level causal model and IIT-aligned
  interchange-intervention protocol.
- CMR comparator: `NOT_IMPLEMENTED`; no local constant/affine SCM-reduction
  baseline is bound to the same replacement and held-out challenge packet.
- New head-to-head: `NOT_STARTED`; no new checkpoint, corpus, or challenge
  entropy was selected or generated in this follow-up.
- Assurance for this literature follow-up: `INTERNAL_PASS`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `NOT_APPLICABLE` (no experiment performed).
- Custody: only this note was added to the repository; model assets and prior
  prompt-bearing artifacts remain in their existing external owner custody.
- Next gate: freeze a matched study with explicit SCM variables and
  base/source interchange semantics, plus NoT-style, DIITO, CMR, V17-style
  finite-effect, and candidate arms. If using char-IIT archives, first resolve
  asset-level rights and custody; otherwise freeze an independently generated
  synthetic task. Select a fresh checkpoint only after that freeze, then
  generate held-out interventions. Keep V17 and V29/V30 as unchanged evidence.
