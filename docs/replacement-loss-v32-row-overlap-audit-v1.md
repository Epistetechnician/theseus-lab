# V32 preparation-row audit addendum

State slices: `theseus-lab-sandbox-v1`,
`theseus-v32-row-disjointness-audit-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus digest-bound uncommitted packets.
Assurance `INTERNAL_PASS`; reviewer `NOT_SUBMITTED`. This read-only audit
preserves V32's source, raw artifacts, metrics, frozen FAIL, and reproduction.

The [exact source-file and overlap inventory](replacement-loss-v32-row-overlap-audit-v1.json)
found zero within-case preparation overlaps and zero challenge/preparation
overlap rows. It also found **24 cross-case partition overlap pairs**, totaling
33,792 pairwise shared rows. That sum counts pairwise reuse, not unique rows.
The source packet is
`a50766c1611c72a444fa8213378761adaef84811f5f3afdc0181aafce7f11c67`.

V32 used adjacent case seeds and small partition offsets. For example,
case 3201's visible seed 3212 is also case 3202's train seed 3212. The
individual case's train/visible/adaptation/donor sets were disjoint, but
different cases partly reused preparation vectors. Its code checked each
case locally and checked heldout endpoints against all preparation sets;
it did not enforce global preparation disjointness. The protocol's language
was broader than that enforcement. Independent corpus draws across the
three case seeds are therefore not established. Numeric reproduction does
not repair that gap. No heldout leak into fitting was found by this audit.

V33's stronger global check exposed the same seed-scheduling bug before
replacement training and stopped INCONCLUSIVE. V33R1 uses namespaced
partition streams, tests every declared stream identity, and verifies exact
row disjointness across cases, partitions, prior inputs, and challenges.
All raw vectors remain in their original owner-controlled custody. The
audit claim is exact row reuse for these ten bound input files only;
other experiments and model/data relationships were not audited here.
External science and production authority remain unopened.
