---
phase: 30
slug: multi-process-backend-parity
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-14
---

# Phase 30 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest via `uv run --python 3.12 pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --python 3.12 pytest tests/test_blob_namespace.py tests/test_blob_store.py tests/test_backend_parity.py tests/test_fault_injection.py tests/test_storage_mode.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Full suite command** | `uv run --python 3.12 pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Estimated runtime** | ~30-90 seconds targeted, full suite per project baseline |

---

## Sampling Rate

- **After every task commit:** Run the task-specific command from the verification map.
- **After every plan wave:** Run the focused Phase 30 command.
- **Before `$gsd-verify-work`:** Full suite must be green or any unrelated baseline failure must be documented with evidence.
- **Max feedback latency:** 90 seconds for targeted checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 30-01-01 | 01 | 1 | PAR-01 | T-30-01 | Unique temp blob writes do not corrupt same-blob final content | unit/regression | `uv run --python 3.12 pytest tests/test_blob_namespace.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | ✅ | ⬜ pending |
| 30-02-01 | 02 | 1 | PAR-02 | T-30-02 | JSON and SQLite preserve/filter user metadata through BlobStore; PostgreSQL remains skip-gated when unavailable | parity/integration | `uv run --python 3.12 pytest tests/test_backend_parity.py tests/test_blob_store.py tests/test_metadata.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py` | ✅ | ⬜ pending |
| 30-03-01 | 03 | 2 | PAR-03 | T-30-03 | Failed same-key overwrite preserves previous committed value and blob in cache mode | fault-injection/regression | `uv run --python 3.12 pytest tests/test_core.py tests/test_fault_injection.py tests/test_cache_integrity.py -x -q --ignore=tests/test_tensorflow_handler.py` | ✅ | ⬜ pending |
| 30-03-02 | 03 | 2 | PAR-03 | T-30-04 | Failed same-key overwrite preserves previous committed value and blob in storage mode | storage-mode regression | `uv run --python 3.12 pytest tests/test_storage_mode.py tests/test_fault_injection.py -x -q --ignore=tests/test_tensorflow_handler.py` | ✅ | ⬜ pending |
| 30-04-01 | 04 | 2 | PAR-04 | T-30-05 | Integrity enumeration sees custom-extension orphan blobs while excluding reserved/temp files | integrity/regression | `uv run --python 3.12 pytest tests/test_cache_integrity_verification.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification. PostgreSQL service-backed tests may skip when `CACHENESS_TEST_POSTGRES_URL` is unavailable; the plan must keep those skip gates explicit rather than requiring local service setup.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s for targeted checks
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-14
