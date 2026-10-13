# Replacement-loss v9 implementation qualification

State slice: `theseus-lab-sandbox-v1`.

Before packet freeze, focused runner tests found two draft defects: the CKA
draft called `_module_digest` on the v3 runner adapter, which does not export
that helper, and the replicator read an unused primary-freeze value in a
fixture-order test. The runner was corrected to call the pinned v3 core helper
and the replicator stopped reading that unused value. Both errors occurred
before any checkpoint, challenge, or model result was selected or generated.
After the primary-freeze verification was tightened, its unit fixture was
updated to stub packet validation while it checks the reveal ordering.

The repository's pre-existing `.venv` contains Transformers 5.17.0, so it was
not used to qualify the v9 packet. A clean Python 3.12.13 environment was
installed from the v9 direct pins, and the exact installed distribution set
was captured in `requirements-v9.lock.txt`. The focused v9 suite passed 16/16
in that pinned environment after the fixes. Tiny randomly initialized OPT,
GPT-2, and Llama configs exercised projection capture, layer selection, ridge
fitting, replacement, and held-out forward prediction. These are adapter smoke
checks only, not eligible checkpoints or a model experiment. This qualifies
the local contracts; it is not independent review or a scientific result.
