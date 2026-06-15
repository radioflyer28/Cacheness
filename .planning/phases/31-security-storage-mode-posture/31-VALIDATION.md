---
phase: 31
slug: security-storage-mode-posture
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-14
---

# Phase 31 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest via `uv run --python 3.12 pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Full suite command** | `uv run --python 3.12 pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` |
| **Estimated runtime** | ~45-120 seconds targeted, full suite per project baseline |

---

## Sampling Rate

- **After every task commit:** Run the task-specific command from the verification map.
- **After every plan wave:** Run the focused Phase 31 command.
- **Before `$gsd-verify-work`:** Full suite must be green or any unrelated baseline failure must be documented with evidence.
- **Max feedback latency:** 120 seconds for targeted checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01-01 | 01 | 1 | SEC-01 | T-31-01 | Minimum signature version rejects downgraded signatures while defaults preserve old-cache compatibility | unit/security regression | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_key_rotation_api.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-01-02 | 01 | 1 | SEC-01 | T-31-02 | Unsigned-entry risk and recommended strict settings are documented | docs/regression | `uv run --python 3.12 pytest tests/test_cache_signing.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-02-01 | 02 | 1 | SEC-02 | T-31-03 | Encrypted reads route through the configured blob backend and round-trip with in-memory backends | integration/security regression | `uv run --python 3.12 pytest tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-02-02 | 02 | 1 | SEC-02 | T-31-04 | Encrypted reads prefer handler byte paths and leave no plaintext temp files after fallback reads | unit/security regression | `uv run --python 3.12 pytest tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-03-01 | 03 | 2 | SEC-03 | T-31-05 | Interrupted key rotation keeps the original key file active and old entries verifiable | fault-injection/security regression | `uv run --python 3.12 pytest tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-03-02 | 03 | 2 | SEC-03 | T-31-06 | Local blob re-encryption during rotation uses an atomic rotating temp path rather than in-place truncation | fault-injection/regression | `uv run --python 3.12 pytest tests/test_key_rotation.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-04-01 | 04 | 2 | SEC-04 | T-31-07 | UnifiedCache and BlobStore produce signatures over the same canonical field set for new writes | parity/security regression | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-04-02 | 04 | 2 | SEC-04 | T-31-08 | Existing old-shape signed entries remain readable through an explicit compatibility path | compatibility/security regression | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_cross_system_compatibility.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-05-01 | 05 | 1 | STRG-01 | T-31-09 | Storage-mode destructive cache APIs emit a loud warning under the selected warning-first policy | storage-mode regression | `uv run --python 3.12 pytest tests/test_storage_mode.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-05-02 | 05 | 1 | STRG-01 | T-31-10 | Storage mode still never reintroduces implicit TTL, eviction, or invalid-entry deletion behavior | storage-mode regression | `uv run --python 3.12 pytest tests/test_storage_mode.py tests/test_cache_integrity.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-06-01 | 06 | 2 | STRG-02 | T-31-11 | Storage-mode transaction guarantees document atomic rename vs power-loss durability limits | docs/regression | `uv run --python 3.12 pytest tests/test_config_options.py tests/test_config_validation.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-06-02 | 06 | 2 | STRG-02 | T-31-12 | Opt-in `fsync_on_write` paths invoke local fsync hooks for JSON saves, local blob writes, and intent files where feasible | unit/regression | `uv run --python 3.12 pytest tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-07-01 | 07 | 2 | STRG-03 | T-31-13 | Write intents are recorded before blob writes in cache mode and cleanup tolerates missing never-created blobs | fault-injection/regression | `uv run --python 3.12 pytest tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |
| 31-07-02 | 07 | 2 | STRG-03 | T-31-14 | Write intents are recorded before blob writes in storage mode without deleting committed durable entries | storage-mode/fault-injection regression | `uv run --python 3.12 pytest tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors should have automated verification. Power-loss durability itself is not simulated; STRG-02 should verify documentation and opt-in fsync wiring rather than claiming real power-loss proof.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s for targeted checks
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-14
