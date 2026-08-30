---
phase: 02-canonical-storage-and-integrity-contract
plan: "05"
subsystem: storage-security
tags: [blobstore, manifest, hmac-sha256, sha256, integrity, guarded-snapshot]
requires:
  - phase: 02-02
    provides: exact local canonical-manifest persistence
  - phase: 02-03
    provides: non-deserializing native payload-contract resolution
  - phase: 02-04
    provides: typed direct-read failure taxonomy
provides:
  - Strict canonical HMAC key provenance with no read-time key replacement
  - Authenticated, validated, one-snapshot BlobStore direct reads
  - Typed non-mutating missing-payload and tamper outcomes
affects: [BlobStore, phase-02-plan-06, phase-06-unified-cache]
actuals:
  tokens: 7191
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Canonical manifest keys are read-only during reopen and explicitly initialized only for a fresh store.
    - Authenticated manifests resolve declared payload contracts before one guarded snapshot is hashed and deserialized.
key-files:
  created:
    - tests/test_blob_store_integrity.py
  modified:
    - src/cacheness/storage/integrity.py
    - src/cacheness/security.py
    - src/cacheness/storage/blob_store.py
key-decisions:
  - "Canonical signing uses exact 32-byte key material or strict POSIX no-follow key-file attestation; legacy CacheEntrySigner behavior is excluded."
  - "Direct reads authenticate and validate signed metadata before opening one private snapshot, which is SHA-256 and size checked before handler access."
requirements-completed: [STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08, MIGR-07]
coverage:
  - id: D1
    description: Required canonical HMAC signing rejects absent, unsafe, malformed, and replacement key material without generating fallback keys.
    requirement: SECU-04
    verification:
      - kind: unit
        ref: tests/test_blob_store_integrity.py key/signature/provider matrix
        status: pass
    human_judgment: false
  - id: D2
    description: Every canonical direct read authenticates and validates metadata before one snapshot is digest/size verified and passed to its native handler.
    requirement: SECU-03
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_blob_store_integrity.py tests/test_blob_store_read_contract.py tests/test_filesystem_containment.py -x
        status: pass
    human_judgment: false
  - id: D3
    description: Manifest, lifecycle, handler-contract, missing-payload, and tamper failures are typed and leave authenticated evidence untouched.
    requirement: STOR-08
    verification:
      - kind: integration
        ref: tests/test_blob_store_integrity.py critical-field and payload-tamper matrix
        status: pass
    human_judgment: false
duration: 10 min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 05: Authenticated BlobStore Read Boundary Summary

**Strict canonical key provenance and a one-snapshot verified BlobStore read pipeline now prevent unauthenticated metadata or altered payload bytes from reaching a handler.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-08-30T14:30:12Z
- **Completed:** 2026-08-30T14:40:58Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Separated canonical manifest signing from the legacy configurable cache signer, requiring exact 32-byte supplied material or POSIX no-follow regular-file ownership and permission attestation.
- Prevented read and reopen paths from replacing missing keys or downgrading signed records, while retaining a deliberate fresh-store initialization path for writes.
- Ordered direct reads as manifest authentication, key/lifecycle/handler-contract/locator validation, one guarded snapshot, SHA-256 and size verification, then handler deserialization.
- Added adversarial contracts for absent/wrong signatures, altered critical fields, unsupported handler identities, same-length/truncated/extended payloads, missing payloads, and unchanged failure evidence.

## Task Commits

1. **Task 1: Enforce strict required canonical signing and key provenance** - `4b37f6d` (RED), `760eaee` (GREEN)
2. **Task 2: Enforce authenticate-validate-snapshot-digest-deserialize ordering** - `79f7b86` (RED), `33b0ada` (GREEN)

## Files Created/Modified

- `src/cacheness/storage/integrity.py` - strict key provider, POSIX provenance checks, fixed HMAC-SHA256, and snapshot digest helpers.
- `src/cacheness/security.py` - documents the legacy-only scope of the configurable cache signer.
- `src/cacheness/storage/blob_store.py` - canonical key initialization and authenticated pre-deserialization direct-read pipeline.
- `tests/test_blob_store_integrity.py` - key, signature, ordering, critical-field, and tamper invariance matrix.

## Decisions Made

- Canonical key reads are non-mutating and use only an exact supplied 32-byte value or an attested POSIX file; no ephemeral or replacement key is allowed during reads or reopens.
- Handler contract resolution is a pre-snapshot metadata validation step; only the verified private snapshot reaches the trusted-payload handler.

## TDD Gate Compliance

- RED commits: `4b37f6d`, `79f7b86`
- GREEN commits: `760eaee`, `33b0ada`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test fixture] Corrected the canonical test key to the required exact 32-byte length.**
- **Found during:** Task 1 GREEN
- **Issue:** The initial literal exceeded the fixed HMAC key length and therefore exercised validation rather than the intended signing path.
- **Fix:** Replaced it with a stable 32-byte test-only value.
- **Files modified:** `tests/test_blob_store_integrity.py`
- **Verification:** The exact key/signature/provider matrix and targeted Ruff gate passed.
- **Committed in:** `760eaee`

---

**Total deviations:** 1 auto-fixed (Rule 1 test-fixture correction).
**Impact on plan:** The correction tightened the planned exact-key contract without broadening storage, recovery, or cache-policy scope.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02-06 can rely on canonical direct reads having a strict authenticity and one-snapshot integrity boundary. Phase 3 recovery/CAS and Phase 6 cache-policy rewiring remain outside this plan.

## Self-Check: PASSED

- Confirmed the three planned source files, `tests/test_blob_store_integrity.py`, and this SUMMARY exist on disk.
- Confirmed TDD commits `4b37f6d`, `760eaee`, `79f7b86`, and `33b0ada` exist in git history.
- Re-ran all three blocking high-severity test subsets, the full planned test suite (one expected Windows-only skip), and targeted Ruff successfully.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
