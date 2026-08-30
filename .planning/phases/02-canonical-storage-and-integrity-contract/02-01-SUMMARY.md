---
phase: 02-canonical-storage-and-integrity-contract
plan: "01"
subsystem: storage
tags: [blobstore, canonical-manifest, hmac-sha256, sha256, integrity]
requires:
  - phase: 01-compatibility-and-security-baseline
    provides: guarded handler snapshots, path containment, and candidate publication
provides:
  - Signed schema-1 BlobStore manifests with deterministic canonical JSON
  - Bounded manifest decoding and typed direct-read integrity outcomes
  - A committed-record direct BlobStore put/get tracer
affects: [phase-02-plans, BlobStore, UnifiedCache-future-seam]
actuals:
  tokens: 12043
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Canonical UTF-8 JSON signed with fixed HMAC-SHA256 projection
    - One guarded snapshot verified by SHA-256 and size before deserialization
key-files:
  created:
    - src/cacheness/storage/manifest.py
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/integrity.py
    - tests/test_blob_manifest.py
    - tests/test_blob_store_read_contract.py
  modified:
    - src/cacheness/storage/blob_store.py
    - src/cacheness/error_handling.py
    - src/cacheness/storage/__init__.py
key-decisions:
  - "Schema-1 manifests use restricted deterministic JSON and bind every canonical field except signature."
  - "Only a genuinely absent raw repository record maps to None; malformed, unauthenticated, unsupported, and lifecycle-conflict records are typed failures."
  - "Read integrity validates one private payload snapshot before handler deserialization."
patterns-established:
  - "Canonical records are bounded independently across raw bytes, depth, collection size, node count, string bytes, and signed-64 integers."
  - "BlobStore owns direct integrity and lifecycle error classification while UnifiedCache remains untouched."
requirements-completed: [STOR-01, STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08, MIGR-02, MIGR-07]
coverage:
  - id: D1
    description: Signed committed manifests drive one direct BlobStore object put/get path.
    requirement: STOR-01
    verification:
      - kind: integration
        ref: tests/test_blob_store_read_contract.py
        status: pass
    human_judgment: false
  - id: D2
    description: Canonical schema-1 bytes are deterministic, bounded, independently versioned, and completely signed.
    requirement: SECU-05
    verification:
      - kind: unit
        ref: tests/test_blob_manifest.py
        status: pass
    human_judgment: false
  - id: D3
    description: Direct reads preserve absence while surfacing typed integrity, version, signing, and lifecycle failures before payload access.
    requirement: STOR-08
    verification:
      - kind: integration
        ref: tests/test_blob_manifest.py tests/test_blob_store_read_contract.py
        status: pass
    human_judgment: false
duration: 7h 50m
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 01: Canonical Manifest Tracer Summary

**Signed schema-1 BlobStore manifests now provide deterministic bounded records, committed-read verification, and typed integrity failures without altering native handler payloads.**

## Performance

- **Duration:** 7h 50m
- **Started:** 2026-08-30T05:33:25Z
- **Completed:** 2026-08-30T13:23:40Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments

- Added an end-to-end BlobStore tracer that persists an authenticated committed manifest, verifies one guarded payload snapshot, and deserializes only after SHA-256 and size checks.
- Froze deterministic UTF-8 canonical JSON with exact parser and model bounds, independently versioned manifest and payload identifiers, and a complete HMAC-SHA256 projection.
- Introduced public reason-coded manifest integrity, version, lifecycle, backend, and migration error types while preserving direct absence as `None` only for a missing repository record.

## Task Commits

1. **Task 1: Store and retrieve one authenticated committed object end to end** - `74556ea` (RED), `afd1b44` (GREEN)
2. **Task 2: Freeze canonical bytes, bounds, versions, and typed early failures** - `50964cd` (RED), `cafb959` (GREEN)

## Files Created/Modified

- `src/cacheness/storage/manifest.py` - Frozen v1 model, canonical codec, bounds, and strict version/signer validation.
- `src/cacheness/storage/manifest_repository.py` - Raw canonical record repository seam over existing metadata backends.
- `src/cacheness/storage/integrity.py` - HMAC-SHA256 and streaming SHA-256/size helpers with persistent signing-key handling.
- `src/cacheness/storage/blob_store.py` - Canonical manifest put/get tracer with authenticate-before-snapshot ordering.
- `src/cacheness/error_handling.py` and `src/cacheness/storage/__init__.py` - Public typed manifest outcomes and supported exports.
- `tests/test_blob_manifest.py` and `tests/test_blob_store_read_contract.py` - Boundary, signed-projection, and end-to-end ordering contracts.

## Decisions Made

- Canonical schema-1 records sign every canonical field except the signature itself, using a fixed HMAC-SHA256 algorithm.
- The full canonical document, rather than only nested metadata, defines the 16-level nesting limit; its fixture accounts for the root contract node.
- Direct BlobStore failures are typed and non-mutating; only a truly absent repository record returns `None`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test fixture] Corrected the nesting-boundary fixture to count the complete manifest document.**
- **Found during:** Task 2: Freeze canonical bytes, bounds, versions, and typed early failures
- **Issue:** The original fixture counted nested metadata independently of the canonical document root, making the exact 16-level boundary fail despite correct production validation.
- **Fix:** Built the raw fixture at the document-level depth and retained `MAX_NESTING_DEPTH = 16` in production.
- **Files modified:** `tests/test_blob_manifest.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py tests/test_blob_store_read_contract.py -x`
- **Committed in:** `cafb959`

**2. [Rule 2 - Security] Enforced the fixed signer and complete outgoing record bounds.**
- **Found during:** Task 2: Freeze canonical bytes, bounds, versions, and typed early failures
- **Issue:** Direct model construction did not bound required top-level strings or final encoded bytes, and a manifest could declare an unsupported signature algorithm.
- **Fix:** Enforced HMAC-SHA256 as the v1 signature algorithm, bounded required strings and emitted canonical bytes, and added exact total-node boundary coverage plus signed-projection mutations for metadata and signer fields.
- **Files modified:** `src/cacheness/storage/manifest.py`, `tests/test_blob_manifest.py`
- **Verification:** Focused pytest and Ruff passed after the fix.
- **Committed in:** `cafb959`

---

**Total deviations:** 2 auto-fixed (1 test-fixture correction, 1 security-boundary completion).
**Impact on plan:** Both fixes preserve the frozen contract and are required for deterministic, fail-closed manifests; no scope expansion occurred.

## Issues Encountered

The sandbox initially could not open the external uv cache. The project checks passed once cache access was authorized; no test, lint, or authentication failure remained.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plans 02-02 through 02-07 can build on the signed raw-record seam, immutable manifest contract, and direct BlobStore tracer. General lifecycle CAS/recovery, backend parity, UnifiedCache delegation, and migration execution remain intentionally deferred to their planned phases.

## Self-Check: PASSED

- All eight planned source and test files exist.
- TDD RED/GREEN commits `74556ea`, `afd1b44`, `50964cd`, and `cafb959` exist in git history.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
