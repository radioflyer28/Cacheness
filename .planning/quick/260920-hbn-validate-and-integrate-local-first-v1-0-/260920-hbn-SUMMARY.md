---
quick_id: 260920-hbn
status: incomplete
date: 2026-09-20
candidate_commit: 7975622de7e2e62014a5c81e461889124501cb38
---

# Integration halted at the regression gate

The live HTTPS remote check found `main` still at
`f1f87a2909ad781683a3912fb2b4c25ba15c4c80` and `v1.0` still at tag
object `6c03f9da9c4fe1084cee72066addf233b2646e50`. The integration
candidate's parent remains the remote tip and its tree remains exactly the
tagged v1.0 tree. `uv lock --check` passed in an isolated checkout.

The documented complete local regression command did not pass on that exact
checkout. Initially collection failed because `tools/verify_phase4_cutover.py`
reads the now-archived Phase 4 `04-VALIDATION.md` from its old active-phase
path. A temporary one-line path correction made its 16 targeted tests pass,
but the full suite then exposed many more historical test/verifier references
to archived phase artifacts, superseded roadmap sections, and a Phase 3 tag
assertion contradicted by the already-existing `v1.0` tag. The temporary
correction was reverted; no product or test source was committed.

The independent fresh-wheel packaging gate **passed** on exact revision
`7975622de7e2e62014a5c81e461889124501cb38`, including base public
round trips and all five declared extras as non-live probes where applicable.
Its evidence was written outside the repository at
`/private/tmp/cacheness-v1-integration-Ol5PxT-packaging.json`.

Because the complete local suite is a documented regression boundary, neither
local nor remote `main` was moved and nothing was pushed. The next task is a
bounded post-archive test/verifier reconciliation, with an explicit decision
about the inherently historical tag assertion. Do not mark the suite green by
adding compatibility layout shims or by counting excluded tests as a pass.
After that, rerun the full suite and fresh-wheel gate on the final commit and
recheck the live remote tip before the fast-forward.
