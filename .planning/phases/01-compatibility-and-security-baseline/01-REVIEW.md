---
phase: 01-compatibility-and-security-baseline
reviewed: 2026-08-30T04:07:10Z
depth: deep
files_reviewed: 5
files_reviewed_list:
  - src/cacheness/core.py
  - src/cacheness/metadata.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - tests/test_clear_recovery.py
findings:
  critical: 1
  warning: 0
  info: 0
  total: 1
status: issues_found
---

# Phase 1: Renewed Fix Re-review Report

**Reviewed:** 2026-08-30T04:07:10Z
**Depth:** deep
**Files Reviewed:** 5
**Fix Commit:** `08b3f99`
**Review Head:** `9158c24`
**Status:** issues_found

## Summary

The fix closes CR-R1's JSON publication-authority failure and CR-R2's ordinary
committed-journal failure boundary. It also closes CR-R3 for `put()`, the primary
BlobStore reads, and the primary UnifiedCache `get`/list/stats paths: live same- and
second-instance JSON/SQLite writers now serialize behind clear, stale JSON instances
refresh after admission, and unresolved publication outcomes poison the live owner.

CR-R3 is not fully closed because public UnifiedCache query reads bypass the new
prepared/committed state gate. A direct SQLite reproduction left a prepared journal
after metadata clear: `get()` correctly raised `CacheStorageError`, while
`query_meta(score=1)` returned `[]`. The focused implementation matrix otherwise
passed. Validation must remain draft/pending.

The explicitly deferred `STOR-03..STOR-06`, `CACH-03`, `BACK-03`, and `BACK-06`
generalizations remain outside this review. The finding is limited to consistency
among current Phase 1 public read paths during the new narrow clear transaction.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-R4: Public metadata queries bypass prepared-clear admission and expose an intermediate empty state

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:491-614,616-795,797-813`

**Issue:** The fix decorates `get()`, `get_stats()`, and `list_entries()` with
`_clear_read_coordinated`, but leaves `query_meta()`, `query_custom()`,
`query_custom_session()`, and `get_custom_metadata_for_entry()` outside the clear
coordinator. After a `BaseException` occurs immediately after SQLite metadata clear,
the journal remains `prepared` and the payloads are recoverable tombstones. In a
direct reproduction, `get(score=1)` correctly rejected the read with “A prepared
clear journal requires recovery before reads,” but `query_meta(score=1)` opened a
SQLite session directly and returned `[]`. That result is indistinguishable from a
real no-match result and exposes a state which the next mutation/restart will roll
back. Custom-metadata query paths can likewise observe link/table state without the
same admission decision. The public API therefore gives contradictory answers for
one transaction and violates the fix's own fail-closed prepared-read contract.

**Fix:** Apply clear read admission to `query_meta()`, `query_custom()`, and
`get_custom_metadata_for_entry()`. For `query_custom_session()`, hold admission for
the entire returned context-manager lifetime rather than only while constructing
the query. Ensure admission errors occur outside the methods' broad exception
handlers so they cannot be converted to `None`, `[]`, or `{}`. Add prepared,
committed, poisoned, live-clear blocking, and second-instance SQLite tests for each
public query/read surface.

## Renewed Blocker Closure Audit

- **CR-R1 — CLOSED:** backup retirement after acknowledged JSON replacement logs
  cleanup debt without revoking metadata authority or deleting the referenced
  candidate. First-write and cross-format cases cover BlobStore and UnifiedCache.
- **CR-R2 — CLOSED for the reviewed boundary:** a proven prepared journal rolls
  back; committed/uncertain publication and `BaseException` outcomes poison the
  live coordinator and reject ordinary decorated work until restart recovery.
- **CR-R3 — PARTIAL:** puts, deletes, updates, invalidation, clear, and primary
  reads participate in root admission. Public query reads remain outside it as
  described by CR-R4.

## Verification

- Focused matrix passed:
  `uv run pytest -q -o log_cli=false tests/test_clear_recovery.py tests/test_metadata.py tests/test_cache_integrity.py tests/test_filesystem_containment.py tests/test_core.py::TestCacheness::test_concurrent_access -x`
  (one expected Windows-junction skip).
- Direct prepared-state reproduction: `UnifiedCache.get()` raised the expected
  `CacheStorageError`; `UnifiedCache.query_meta()` incorrectly returned `[]`.
- `01-VALIDATION.md` remains draft/pending.

## Prior Review History (preserved)

The standard review at `2026-08-30T00:32:17Z` reported CR-01 through CR-06:
staged-inode substitution, BlobStore overwrite corruption, UnifiedCache candidate
leakage, partial global clear, anonymous clear tombstones, and untyped oversized
integer query failures.

The renewed deep review at `2026-08-30T03:35:34Z` reported:

1. **CR-R1:** acknowledged JSON metadata could be misclassified as uncommitted.
2. **CR-R2:** failed committed-journal publication left unsafe prepared work live.
3. **CR-R3:** normal writes bypassed clear admission and could be silently discarded.

Commit `08b3f99` was reviewed as the attempted closure of CR-R1 through CR-R3.
The closure audit above preserves their disposition and records the remaining
public-read defect separately as CR-R4.

---

_Reviewed: 2026-08-30T04:07:10Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
