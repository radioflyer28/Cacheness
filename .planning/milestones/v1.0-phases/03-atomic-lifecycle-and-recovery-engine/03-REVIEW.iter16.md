---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T04:52:07Z
depth: deep
head: 0bb710d
files_reviewed: 35
files_reviewed_list:
  - docs/SECURITY.md
  - docs/WINDOWS_COMPATIBILITY.md
  - pyproject.toml
  - uv.lock
  - src/cacheness/config.py
  - src/cacheness/error_handling.py
  - src/cacheness/metadata.py
  - src/cacheness/serialization.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/integrity.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/manifest.py
  - src/cacheness/storage/manifest_repository.py
  - src/cacheness/storage/operation_record.py
  - src/cacheness/storage/operation_repository.py
  - src/cacheness/storage/path_security.py
  - src/cacheness/storage/reconciliation.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_blob_store_close_contract.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_phase1_quality_gates.py
  - tests/test_public_api_contract.py
findings:
  critical: 3
  warning: 1
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T04:52:07Z  
**Depth:** deep  
**HEAD:** `0bb710d`  
**Status:** issues_found

## Summary

This is a fresh review of the complete Phase 3 implementation after the v2
head/event inventory rewrite. The current fix report, commit messages, and green
suites were treated as claims. D-21 and D-22 were respected; none of the findings
requires a hostile store owner or cross-user Windows topology.

The rewrite does bound individual head/event reads, preserve exact post-entry
cursors, keep reverse publication order clearable when there is no empty prefix,
use one deadline across local and kernel key admission, reject blocking Win32
adapters, and revalidate exact bytes before destructive work. Four actionable
issues remain. Clear loses the durable chain when its first inventory window is
empty; a conflict on the first reconciliation source still produces a terminal
token; repository paging converts real authority/evidence read failures into an
empty clean-looking inventory; and pending scheduling events are never compacted.

The six focused Phase 3 modules completed with terminal exit 0. Python 3.11.16
package import, changed-path Ruff, `uv lock --check`, and `git diff --check` also
passed. Those checks do not cover the deterministic failures below.

## Critical Issues

### CR-01: Empty manifest windows leave clear pages unreachable from the durable chain

**Files:** `src/cacheness/storage/lifecycle.py:524-546`,
`src/cacheness/storage/lifecycle.py:748-798`

**Issue:** `_snapshot_clear_targets()` now accepts an empty nonterminal manifest
page, but advances `source_cursor` only in memory. It deliberately writes no
zero-target page or other bridge at the page ID derived from the old cursor. The
first later nonempty page is therefore persisted under a page ID derived from the
advanced cursor. `_continue_clear_locked()` always starts at `source_cursor=None`;
when that first page ID is absent, it treats the clear as having no targets,
marks the operation terminal, and then artifact retirement finds the unreachable
later page's zero-progress checkpoint and fails.

A bounded deterministic probe used `max_inventory_items=2`, overwrote key `k`
three times, added key `x`, and called `clear()`. The first two manifest events
were a valid stale-only window. Clear raised
`CacheManifestIntegrityError: Clear target checkpoint is incomplete during retirement`
instead of deleting the two current entries. Reverse publication without the
stale prefix succeeded on memory, JSON, and SQLite, which isolates this to the
empty-window bridge rather than target sorting.

This still violates D-10/D-11 and STOR-05. A process interruption is not required;
ordinary overwrite history triggers it.

**Fix:** durably represent every source-cursor transition that later continuation
must traverse. Either permit an authenticated zero-target bridge page with a
separate checkpoint contract, or persist the first effective source cursor in
the signed clear operation and make continuation start there. Exercise leading,
middle, and trailing stale windows across memory/JSON/SQLite, page size one and
default page sizes, replacements/deletions, encoded splits, and interruption at
every bridge/page boundary.

### CR-02: A conflict on the first apply source still returns a terminal token

**Files:** `src/cacheness/storage/reconciliation.py:421-494`,
`src/cacheness/storage/reconciliation.py:1063-1115`

**Issue:** Post-conflict reclassification correctly says that a still-present
primary or sidecar requires retry. The retain helpers cannot encode that result
when the affected member is first in the page: they return the incoming cursor,
which is `None` at the start of a chain. `None` also means terminal. Because the
scan already incremented `consumed_*` to the page length, the subsequent pending
calculation is false and no priority/token is emitted.

A deterministic probe created one authenticated completed orphan sidecar,
injected an exact retirement conflict, and ran one-action apply. The exact sidecar
remained; findings included the original orphan, `reconciliation_action_conflict`,
and a refreshed orphan; nevertheless `resume_token` was `None`. This reproduces
the iteration-16 CR-04 failure at current HEAD.

This violates D-14 and STOR-06: a caller following the authenticated continuation
is told that unresolved safe debt is terminal.

**Fix:** represent a generation-bound position *before* the first member (for
example `next_sequence=1` plus the page high-water), rather than overloading
`None`. Derive pending state from the retained outcome, not only consumed counts
and nullable cursors. Cover primary and sidecar update/retire conflicts as the
sole item and first of several, budgets one/two, later independent progress,
repeat/reopen, and exact terminal convergence.

### CR-03: Inventory paging suppresses authority/evidence read failures and reports false terminal cleanliness

**Files:** `src/cacheness/storage/manifest_repository.py:887-897`,
`src/cacheness/storage/operation_repository.py:871-883`,
`src/cacheness/storage/reconciliation.py:343-360`

**Issue:** Both paging implementations conflate a stale scheduling event with a
failure to read or decode its current authority object. Manifest paging catches
malformed/migration-required projections and silently assigns `current_raw=None`.
Operation paging catches typed backend and bounds/integrity failures and likewise
drops the member. Once the high-water position is exhausted, reconciliation can
return no findings and no resume token while the current manifest or control
evidence remains present and unreadable.

Two bounded probes demonstrate the false-clean result:

- After a valid in-memory put, replacing only the stored canonical-manifest
  projection with invalid base64 made dry-run return zero findings and a terminal
  token instead of failing closed or reporting a blocked manifest.
- After creating one valid indexed completed sidecar, injecting a typed backend
  failure in its exact read made dry-run return zero findings and terminal while
  the sidecar remained. An exactly indexed oversized sidecar behaves the same.

The inventory digest is correctly non-authoritative, but that does not authorize
turning a failed current read into proven absence. This violates D-13/D-15 and
STOR-06 and weakens the fail-closed integrity boundary.

**Fix:** distinguish `absent`, `stale digest`, and `read/parse failure`. Propagate
typed manifest/primary authority failures or yield an explicit blocked member;
sidecar and pending page types already allow `raw=None` and should actually emit
that blocked candidate with its exact source cursor. Add malformed projection,
oversized record, permission/backend read failure, digest mismatch, later valid
event, and resume tests proving no error becomes a clean terminal report.

## Warnings

### WR-01: Pending inventory events are never compacted and recovery cost grows with store lifetime

**Files:** `src/cacheness/storage/operation_repository.py:746-841`,
`src/cacheness/storage/operation_repository.py:1783-1885`

**Issue:** The v2 repository implements a safe sparse compactor for all three
families, but calls it only after primary and sidecar retirement. Successful
exclusive control creation always publishes a pending-family event before its
temporary candidate; after the no-replace rename consumes that temporary name,
the stale pending event is never retired. Five ordinary puts deterministically
left five pending event files and `next_sequence=6`. With
`max_inventory_items=2`, every pending page from a fresh cursor was empty but
nonterminal at sequence 3; after the persisted recovery cursor eventually reaches
terminal it is deleted, so the next recovery cycle starts from the lifetime-old
prefix again.

There is no artificial byte ceiling now, but disk use and the number of reopen or
resume cycles before a newly interrupted candidate is reached grow without bound.
That contradicts the claimed safe-compaction/lifetime behavior and undermines the
production-reliability purpose of the bounded startup path.

**Fix:** checkpoint enough exact pending-event identity to compact the event after
successful promotion/consumption, and make bounded recovery compact stale slots
without invalidating existing high-water cursors. Test thousands of successful
controls followed by one interrupted candidate, terminal-reset behavior, crash
during compaction, concurrent append, and a hard per-call read/write bound.

## Verification Performed

- Focused Phase 3 corpus: 6 modules, terminal exit 0.
- Reverse publication clear: memory, JSON, and SQLite succeeded without a stale
  prefix.
- Leading stale-window clear: deterministic failure documented in CR-01.
- First-source sidecar conflict: deterministic terminal-token failure documented
  in CR-02.
- Malformed manifest and injected sidecar read failure: deterministic false-clean
  terminal reports documented in CR-03.
- Pending inventory lifetime probe: deterministic uncollected events documented
  in WR-01.
- Full-suite sandbox attempt reached 100%; its only two failures were environment
  denials (Unix-socket bind returned `EPERM`, and the nested quality-gate `uv`
  process could not access `~/.cache/uv`). They are not cited as product failures
  or as a passing full-suite run.
- Python 3.11.16 isolated package import: passed (`cacheness 0.3.14`).
- Changed-path Ruff: passed.
- `uv lock --check`: passed.
- `git diff --check`: passed.

---

_Reviewer: independent deep rereview, iteration 17_
