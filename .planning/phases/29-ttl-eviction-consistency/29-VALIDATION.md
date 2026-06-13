---
phase: 29
slug: ttl-eviction-consistency
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-13
---

# Phase 29 - Validation Strategy

> Per-phase validation contract for TTL, init cleanup, metadata preservation,
> and remote eviction consistency.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 with pytest-xdist and pytest-json-report |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --python 3.12 pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Full suite command** | `uv run --python 3.12 pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Estimated runtime** | ~15-90 seconds targeted, ~7 minutes full suite |

---

## Sampling Rate

- **After every task commit:** Run the task-specific command listed in the plan.
- **After every plan wave:** Run the Phase 29 focused cross-cutting command.
- **Before `$gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** 90 seconds for the focused Phase 29 slice.

Focused Phase 29 command:

`uv run --python 3.12 pytest tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_update_operations.py tests/test_storage_mode.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py`

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 29-01 | 01 | 1 | TTL-01 | R5 | Stored `expires_at` controls cache-mode reads and JSON/SQLite/PostgreSQL cleanup; fallback TTL applies only when no stored expiry exists. | unit + integration + parity | `uv run --python 3.12 pytest tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_storage_mode.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 29-02 | 02 | 1 | TTL-02 | R6 | Init-time expired cleanup uses public cleanup semantics, deleting metadata, blobs, and invoking eviction hooks. | integration | `uv run --python 3.12 pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 29-03 | 03 | 2 | TTL-03 | R9/R10 | Same-key overwrites preserve access counters; metadata-only updates preserve `created_at`, `ttl_seconds`, `expires_at`, and valid signatures. | backend + integration + signing | `uv run --python 3.12 pytest tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 29-04 | 04 | 2 | TTL-04 | R13 | Size eviction deletes URI blobs through the configured blob backend instead of leaking remote-style objects. | integration | `uv run --python 3.12 pytest tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |

*Status: pending = tests still need to be added or updated during execution.*

---

## Wave 0 Requirements

- [ ] `tests/test_core.py` - stored-expiry read behavior, init cleanup blob deletion, and memory-URI size eviction regressions.
- [ ] `tests/test_metadata.py` and/or `tests/test_backend_parity.py` - JSON/SQLite stored-expiry cleanup parity.
- [ ] `tests/test_storage_mode.py` - storage mode ignores expired `expires_at`.
- [ ] `tests/test_update_operations.py` - metadata-only update preserves `created_at`, TTL fields, and signatures.
- [ ] `tests/test_cache_signing.py` or `tests/test_cache_integrity_verification.py` - signed entry remains readable after metadata-only update.
- [ ] `tests/test_pg_schema_versioning.py` and/or `tests/test_postgresql_backend.py` - PostgreSQL semantics covered with existing Docker-skip behavior where available.

---

## Manual-Only Verifications

All Phase 29 behaviors should have automated verification. PostgreSQL and S3-specific integration checks may skip when Docker-backed services are unavailable, but JSON/SQLite and memory-URI coverage must always run.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify.
- [ ] Wave 0 covers all MISSING references.
- [ ] No watch-mode flags.
- [ ] Feedback latency < 90s for focused validation slice.
- [ ] `nyquist_compliant: true` set in frontmatter after validation passes.

**Approval:** pending
