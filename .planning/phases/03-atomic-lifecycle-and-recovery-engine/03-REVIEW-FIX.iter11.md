---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T18:23:18Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 9
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T18:23:18Z
**Source review:** `03-REVIEW.md`
**Iteration:** 9

## Summary

- Findings in scope: 6
- Fixed: 6
- Skipped: 0

The former CR-01 and CR-02 adversarial premises are resolved by the canonical
D-21/D-22 product contract. This is not a claim that a local filesystem provides
immutable authority against its owning OS principal or cross-principal Windows
coordination. Observable substitutions remain fail closed, and unavailable local
authority capabilities now have stable typed outcomes.

## Fixed Issues

### CR-01: Root-xattr authority errors did not always have a typed failure

**Files modified:** `.planning/PROJECT.md`, `03-CONTEXT.md`, `docs/SECURITY.md`, `src/cacheness/error_handling.py`, `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_public_api_contract.py`
**Commit:** `9497a69`
**Applied fix:** D-21 explicitly places availability of the local lifecycle-control
namespace within the trusted store-owner boundary. The root-xattr path now maps
unavailable APIs, unsupported filesystems, permission/policy denial, read-only
roots, and comparable capability failures to `CacheBlobBackendError` with the
stable `blob_backend_capability_unsupported` reason; unexpected xattr I/O maps
to the stable backend-failure reason and preserves its cause. A missing binding
is created only through the initial create path. After an existing binding has
been observed, disappearance or a different value is an unsafe-path failure and
is never repaired or rebound.

### CR-02: Windows authority scope was not an explicit supported contract

**Files modified:** `.planning/PROJECT.md`, `03-CONTEXT.md`, `docs/SECURITY.md`, `docs/WINDOWS_COMPATIBILITY.md`, `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`
**Commit:** `9497a69`
**Applied fix:** D-22 now defines the supported topology as one Windows OS user
and one interactive or service session. The HKCU plus `Local\\` mutex adapter,
its capability error, and its adapter tests state that exact scope and do not
claim cross-user, cross-service, or cross-session authority. Public deployment
guidance requires ACLs to exclude other principals and an external transactional
authority for a broader topology.

### CR-03: Windows control durability used an unsupported directory flush

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`
**Commits:** `714983d`, `57f43bb`
**Applied fix:** Native control moves request `MOVEFILE_WRITE_THROUGH`; surviving
regular files are flushed through a reparse-safe regular-file handle; and the
code does not call `FlushFileBuffers` on a directory handle. Adapter tests cover
the source-level call and close contract.

### CR-04: Windows pending-control recovery could delete a substituted pathname

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`
**Commit:** `714983d`
**Applied fix:** The Windows fallback leaves an already-superseded pending
candidate blocked instead of unlinking a pathname after its verified handle has
been released. A deterministic substitution regression proves the unrelated
replacement remains untouched.

### WR-01: Uncertain admission unlock reopened the barrier

**Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_concurrency.py`
**Commits:** `800d26f`, `3caba6a`
**Applied fix:** Every release failure poisons the shared barrier, including a
body-error-plus-unlock-error outcome. New normal or aggregate work fails with
the typed lock-release error until owning stores close and reconstruct the
barrier.

### WR-02: Reopen could consume a live clear and change `clear()`'s count

**Files modified:** `src/cacheness/storage/lifecycle.py`, `tests/test_blob_store_concurrency.py`
**Commit:** `800d26f`
**Applied fix:** `clear()` retains its continuation lease from before snapshot
publication through reclamation and retirement. Constructor recovery cannot
consume a still-live clear or change its public count.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness`.

- CPython 3.13 full suite: passed, with only documented optional/platform skips
  and the existing collection warning.
- CPython 3.13 focused authority, close, and public-API suite: 68 passed.
- CPython 3.11.16: `python -m compileall -q src/cacheness` and `import cacheness`
  passed; the focused authority, close, and public-API suite: 68 passed.
- Changed-path Ruff passed; `uv lock --check` and `git diff --check` passed.
- Documentation consistency search confirms all four canonical/public documents
  specify the same trusted-store-owner and one-user/session Windows scope.

## Remaining Platform Verification Gap

Native Windows filesystem, ACL, session, and crash/reopen execution has not run
in this macOS checkout. The adapter tests intentionally do not claim that proof;
the later CI/platform phase owns native Windows execution for the documented
one-user/session topology.

---

_Fixed: 2026-09-01T18:23:18Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 9_

