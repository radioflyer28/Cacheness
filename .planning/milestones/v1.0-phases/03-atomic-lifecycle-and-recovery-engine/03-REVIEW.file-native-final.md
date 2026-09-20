---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-04T16:42:31Z
depth: deep
iteration: 28
archived_as: 03-REVIEW.file-native-final.md
head: f19cc94
base: 8035847
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
  critical: 1
  warning: 0
  info: 0
  total: 1
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-04T16:42:31Z
**Depth:** deep
**Iteration:** 28
**HEAD:** `f19cc94`
**Base:** `8035847`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The iteration-28 changes repair the previously reported committed-successor,
forward-anchor, delayed-writer, cleanup-convergence, and family-wide durable-I/O
lease cases exercised by the focused suite. The remaining durability protocol
still has an earlier interruption window: both exact conditional replacement
and immutable exclusive creation first fsync a randomly named control
candidate. If the process stops at that boundary, ordinary language-level
cleanup does not run, while the bounded inventory has no authenticated locator
for the candidate. Reopen and reconciliation can consequently report a clean
store while leaving managed control residue behind.

This affects scheduler control used by the primary, sidecar, and pending
inventory families, as well as fixed exact-CAS heads and related fixed
authorities. Repetition is not charged to any cursor, inspection, byte, action,
or work budget, so the residue namespace can grow without a protocol bound.

The focused three-module suite passed, the two-delayed-public-head regression
passed ten consecutive repetitions, and Ruff passed on every file changed from
`8035847..f19cc94`. The targeted durability probes below nevertheless
reproduced the false-clean state on a fresh open.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Interrupted control staging leaves unindexed, unbounded managed residue

**Classification:** BLOCKER

**Files:** `/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/path_security.py:1230-1244`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/path_security.py:1279-1286`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/path_security.py:1441-1467`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/path_security.py:1471-1479`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/path_security.py:1826-1858`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:945-983`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2514-2557`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:5004-5026`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/tests/test_manifest_repository_cas.py:2674-2727`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/tests/test_blob_store_concurrency.py:588-632`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/tests/test_blob_store_concurrency.py:635-691`

**Issue:** Exact conditional replacement creates a random
`.cas.tmp` candidate, fsyncs it, and only then enters the exact compare-and-rename
transition. Its only retirement is the in-process `finally` block. Immutable
exclusive creation similarly fsyncs a digest-and-random-token `.pending.*.tmp`
candidate before the no-replace rename, with cleanup confined to its exception
handler. A process interruption after either file fsync therefore leaves a
durable named candidate without executing either cleanup path.

The pending-control observer does not close this window for the inventory's own
event, tail, head, publication, or anchor records: `_record_pending_control()`
explicitly returns for every locator under `.cacheness-inventory-v2` to avoid
recursive self-publication. Exact-CAS candidates have no observer at all. On
reopen, `_uncommitted_inventory_successor()` checks only the one final event,
tail, head, and publication locator named by the signed high-water. It cannot
derive either random temporary name. `list_pending_control_page()` enumerates
only pre-indexed candidates and deliberately performs no directory scan.

Two deterministic fresh-process durability probes demonstrate both paths:

- Interruption immediately after staging and fsyncing a primary inventory
  fixed-head CAS left `.head.json.<random>.cas.tmp`. A new `BlobStore` instance
  opened normally; reconciliation returned no findings; the candidate remained.
- Interruption immediately after staging and fsyncing the next primary inventory
  event left `.event-00000000000000000001.json.pending.<digest>.<random>.tmp`.
  A new instance still read the primary high-water as `next_sequence == 1`;
  reconciliation returned no findings; the candidate remained.

The immutable-create mechanism is shared by scheduler records for primary,
sidecar, and pending inventories, so the same pre-rename window applies to all
three families. Fixed exact-CAS replacement is also used for scheduler heads
and the fixed authorities that advance anchors/cursors. The existing receipt
interruption matrix raises only after `create_bytes_durable_exclusive()` has
already returned, so it starts after the missing candidate-fsync window.
Existing delayed-candidate tests pause at that staging hook but resume the
Python frames, which guarantees their `finally` cleanup and therefore does not
exercise fresh-process recovery.

Because these names are neither authoritative nor indexed, they are absent
from ordinary read/recovery work accounting. Repeated interruptions can add
arbitrarily many files without advancing any signed inventory cursor or
consuming an inspection, byte, action, or cleanup budget. This violates the
phase requirements that every created control object be durably discoverable
before it can survive interruption, that reopen converge without loose scans,
and that reconciliation never return clean while managed lifecycle residue is
unaccounted for.

**Fix:** Do not leave a persistent named staging file unless an authenticated,
bounded recovery locator is durable first. For exact-CAS heads, anchors, and
cursors, use a platform primitive with no visible pre-install name, or publish a
deterministic candidate identity bound to family, predecessor, successor, and
content digest before staging; recovery must exact-compare the predecessor and
then promote or retire only that candidate. For immutable inventory controls,
replace the self-recursive observer gap with a non-recursive bounded scheduling
protocol, or use a durable atomic no-replace strategy that cannot strand a
named pre-install file. Do not repair this with a glob or directory sweep.

Add true fresh-process interruption coverage at candidate creation, candidate
fsync, rename, and directory acknowledgement for exact replace and exclusive
create on every primary/sidecar/pending family. Each case must assert that a
fresh open and bounded reconciliation either account for and resolve the exact
candidate or fail closed; neither may report clean while the file remains.

## Verification Performed

- Focused Phase 3 suite covering
  `tests/test_manifest_repository_cas.py`,
  `tests/test_blob_store_concurrency.py`, and
  `tests/test_blob_store_reconciliation.py`: passed at 100%.
- `test_two_delayed_public_head_candidates_reopen_to_a_clean_authority`:
  passed ten consecutive repetitions.
- Delayed exact-head candidates across many later generations and two delayed
  candidates are covered by the passing focused concurrency suite.
- Fresh-process interruption after exact-head candidate fsync: reproduced an
  unindexed `.cas.tmp` residue that survived reopen and clean reconciliation.
- Fresh-process interruption after immutable inventory-event candidate fsync:
  reproduced an unindexed digest-bound `.pending.*.tmp` residue while the signed
  primary high-water remained at sequence 1 and reconciliation reported clean.
- Ruff on all five files changed by `8035847..f19cc94`: passed.
- Target identity verified as `f19cc949d923dcf66050e90474b884bba29ad348`.

---

_Reviewed: 2026-09-04T16:42:31Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
