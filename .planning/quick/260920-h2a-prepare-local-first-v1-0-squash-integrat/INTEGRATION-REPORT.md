# Local-first v1.0 integration candidate

Prepared on 2026-09-20. This is a local-only candidate, not a merge, push,
release, or assertion of remote-service qualification.

## Refs and exact-tree proof

| Role | Ref | Commit |
|---|---|---|
| Audited local snapshot and full history | `v1.0`, `codex/archive-v1-full-history` | `422ab93c2590585393087f47097fc30d21b03783` |
| Fetched GitHub main tip and preserved history | `codex/archive-remote-main-v0.6` | `f1f87a2909ad781683a3912fb2b4c25ba15c4c80` |
| One-commit integration candidate | `codex/v1-local-first-squash` | `7975622de7e2e62014a5c81e461889124501cb38` |

The candidate's sole parent is the fetched GitHub main tip. Its tree is
`53652c802dc1ad70a66a8e659409de94d2eafea3`, exactly `v1.0^{tree}`.
Thus a future fast-forward of remote main to this branch would retain the
remote history as ancestry while making the audited local files authoritative
in one commit. The local `v1.0` tag and archive branch retain the 1,292 local
commits as a separate, inspectable history; this candidate does **not** carry
those commits as ancestry. Local and fetched remote tips share merge base
`a22f4b4575cb8213d9783ed388d2a70727563db1`; the remote has 56 commits
not in local main. The local `origin/main` tracking ref is stale at that merge
base; all comparisons here use the exact fetched remote commit, not that ref.

No `main` ref or tag was moved and no branch was pushed. The candidate differs
from the fetched remote in 982 paths (+195,453/−66,258 lines); that scale is
why it must be reviewed as a deliberate replacement snapshot, not an ordinary
small merge.

## Remote-only material replaced by the candidate

The fetched remote has 119 paths absent from the v1.0 tree: 37 tests, 21 docs,
19 examples, 7 source files, 7 scripts, 7 `.beads` files, 6 `config` files,
5 benchmarks, and 10 other root/editor/CI files. This is a tree comparison,
not a claim that all these paths were created in the 56 remote-only commits.

| Material | Disposition before any integration |
|---|---|
| `.github/workflows/release.yml` | **Do not silently restore.** It builds on `v*` tag pushes and creates a GitHub release. The local v1.0 closeout explicitly did not qualify a published release or live PostgreSQL/S3 and controlled-Linux performance claims. Decide publication workflow only with the future release gates. |
| `src/cacheness/custom_metadata.py`, `storage/backends/{base,blob_backends,s3_backend}.py`, `storage/paths.py` | Older authority/backend/path surfaces. Do not transplant into the current BlobStore/obstore architecture; reassess any specific capability against ADR 0001 and current topology contracts. The old path helper accepted absolute/relative path conversion and cannot bypass current containment. |
| `src/cacheness/entry_list.py` | Convenience result-list API from the old cache surface. Omit during pre-production API cutover unless a new product requirement justifies it. |
| `src/cacheness/size_utils.py` | General parsing/formatting helper. Potentially portable, but it is not a reason to restore an old module without a current call site and test contract. |
| 37 absent remote tests | Do not copy wholesale; many bind to retired namespaces, storage modes, inline blobs, TensorFlow, or old backend abstractions. Salvage individual invariants only after mapping them to the current public API. |
| Remote docs/examples/benchmarks/dev tooling | Treat as historical candidates, not live documentation. Several describe retired APIs or qualifications. Port an individual useful item only after verifying it against the v1.0 implementation. |

The remote package declares version `0.6.0`; the v1.0 snapshot declares
`0.3.14`. The milestone/tag version is not the package version. Resolve the
next distribution version and release notes deliberately before publication;
do not alter this exact-tree candidate just to reconcile numbers.

## Gates before a future main update or release

1. Confirm the remote main still equals the preserved `f1f87a2` tip, or redo
   the remote-only review and rebase/recreate the candidate on the new tip.
2. Review this large replacement diff and explicitly accept the dispositions
   above, especially the removal of the tag-triggered release workflow.
3. Re-run exact-commit qualification for `7975622` (or its eventual successor).
   Tree equality with audited `v1.0` does not make the new commit SHA itself an
   already qualified release. Preserve the v1.0 deferred live-service and
   controlled-Linux nonclaims.
4. Decide a package version and publication policy separately. An unauthenticated
   release lookup did not establish whether a GitHub release currently exists;
   do not infer one way or the other from it.
5. Only then, if approved, perform a fast-forward main update and remote push.
   Neither action is part of this task.

Useful reproducibility checks:

```sh
git rev-parse codex/v1-local-first-squash^ v1.0^{tree} codex/v1-local-first-squash^{tree}
git rev-list --count codex/archive-remote-main-v0.6..codex/v1-local-first-squash
git diff --name-status --diff-filter=A v1.0 codex/archive-remote-main-v0.6
git diff --stat codex/archive-remote-main-v0.6 codex/v1-local-first-squash
```
