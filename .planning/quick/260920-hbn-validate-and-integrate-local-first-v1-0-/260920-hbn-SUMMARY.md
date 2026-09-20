---
quick_id: 260920-hbn
status: complete
date: 2026-09-20
candidate_commit: 7975622de7e2e62014a5c81e461889124501cb38
integrated_snapshot_commit: ac25040a0d9daded23bd27582a701b6754dfa2f3
---

# Integration completed after regression repair

## Initial gate failure

The first live HTTPS remote check found `main` at
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

Because the complete local suite is a documented regression boundary, no main
ref moved during that first attempt. The separate quick task `260920-hmi`
repaired the post-archive checks without compatibility layout shims or blanket
test exclusions.

## Completed integration

The final snapshot `ac25040a0d9daded23bd27582a701b6754dfa2f3` has the
fetched remote tip `f1f87a2909ad781683a3912fb2b4c25ba15c4c80` as its
sole parent. Its tree exactly equals repaired local main at
`7247f7afa55e335ef128ecdd5ea5e0aa2de29a10`. The latter commit and its
full atomic history remain at `codex/archive-local-main-pre-integration`;
the original exact-v1.0 candidate and both earlier archive refs also remain.

In a fresh checkout of exact `ac25040`, `uv lock --check`, the documented
complete non-live suite, and `tools/run_phase8_packaging.py` all passed.
Packaging evidence is at `/private/tmp/cacheness-final-ac25040-packaging.json`.
The live HTTPS remote main tip was rechecked before a normal, non-force
fast-forward push and verified afterward at `ac25040`; the `v1.0` tag object
remained unchanged. Local `main` and `origin/main` were aligned to that same
commit. No GitHub release was created or claimed, and live PostgreSQL/S3,
controlled-Linux performance, and native Windows remain unqualified.
