---
phase: 28
slug: silent-data-loss-remediation
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-13
validated: 2026-06-13
---

# Phase 28 - Validation Strategy

> Retroactive Nyquist validation reconstructed from Phase 28 plans, summaries,
> source changes, and current automated test evidence.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest with Hypothesis |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --group recommended --with pytest --with hypothesis --python 3.12 pytest tests/test_core.py tests/test_write_intent.py tests/test_metadata.py tests/test_json_schema_versioning.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py tests/test_configurable_serialization.py -x -q --override-ini="addopts=" -p no:cacheprovider --ignore=tests/test_tensorflow_handler.py` |
| **Full suite command** | `uv run --group recommended --with pytest --with hypothesis --python 3.12 pytest tests/ -x -q --override-ini="addopts=" -p no:cacheprovider --ignore=tests/test_tensorflow_handler.py` |
| **Estimated runtime** | ~78 seconds focused, ~7 minutes full suite |

---

## Sampling Rate

- **After every task commit:** Run the task-specific pytest command from each plan.
- **After every plan wave:** Run the Phase 28 focused validation command above.
- **Before `$gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** ~78 seconds for the focused Phase 28 slice.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 28-01 | 01 | 1 | REL-01 | R1 | `clear_all()` removes active namespace blobs and preserves reserved files; `clear_all_namespaces()` clears namespace blob dirs. | regression | `uv run ... pytest tests/test_core.py ...` | yes: `tests/test_core.py` | green |
| 28-02 | 02 | 1 | REL-02 | R2/R17 | Stale intent relative blob paths resolve under `cache_dir`, not process CWD. | regression | `uv run ... pytest tests/test_write_intent.py ...` | yes: `tests/test_write_intent.py` | green |
| 28-02 | 02 | 1 | REL-03 | R2/R17 | Stale-intent cleanup preserves blobs with committed metadata entries. | regression | `uv run ... pytest tests/test_write_intent.py ...` | yes: `tests/test_write_intent.py` | green |
| 28-02 | 02 | 1 | REL-04 | R2/R17 | Storage-mode init cleanup removes only uncommitted orphan blobs and preserves durable entries. | regression | `uv run ... pytest tests/test_write_intent.py ...` | yes: `tests/test_write_intent.py` | green |
| 28-03 | 03 | 1 | REL-02/REL-03/REL-04 | R8/SEED-006 | Non-inline cache-mode and storage-mode writes record planned intents before blob I/O begins. | regression | `uv run ... pytest tests/test_write_intent.py ...` | yes: `tests/test_write_intent.py` | green |
| 28-04 | 04 | 1 | REL-05 | R3 | JSON `put_entry()` and `remove_entry()` surface data-critical save failures while telemetry remains best-effort. | regression | `uv run ... pytest tests/test_metadata.py ...` | yes: `tests/test_metadata.py` | green |
| 28-04 | 04 | 1 | REL-06 | R4 | Corrupt JSON metadata files are preserved as timestamped `*.corrupt-*` backups before starting empty. | regression | `uv run ... pytest tests/test_json_schema_versioning.py ...` | yes: `tests/test_json_schema_versioning.py` | green |
| 28-05 | 05 | 1 | KEY-01 | U1 | Large tuples, arbitrary hashable objects, and default repr fallback paths avoid process-randomized key material. | regression | `uv run ... pytest tests/test_serialization.py tests/test_cache_key_consistency.py ...` | yes: `tests/test_serialization.py`, `tests/test_cache_key_consistency.py` | green |
| 28-05 | 05 | 1 | KEY-02 | U1/property todo | Cache-key determinism has subprocess and bounded Hypothesis coverage for large tuples and repeated calls. | property/regression | `uv run ... pytest tests/test_property_based.py tests/test_cache_key_consistency.py ...` | yes: `tests/test_property_based.py`, `tests/test_cache_key_consistency.py` | green |

*Status: green = covered by automated tests that passed during validation.*

---

## Current Validation Evidence

- Phase 28 focused validation run on 2026-06-13:
  `203 passed, 1 warning in 78.32s`.
- Full suite evidence from Phase 28 execution summary:
  `1780 passed, 130 skipped, 14 warnings`.
- The focused validation command initially failed under the workspace sandbox
  because Blosc2 attempted to read its Windows AppData CPU cache. The same
  command passed when rerun with filesystem access outside the sandbox.
- `ty check` failures reported in plan summaries are existing baseline typing
  diagnostics, not Nyquist test gaps for Phase 28 behavior.
- Plan 05 documents one process deviation: the verify-first subprocess repro
  was missed before implementation. The final cross-subprocess regression now
  exists and passed during this validation.

---

## Wave 0 Requirements

Existing pytest/Hypothesis infrastructure covers all Phase 28 requirements.

---

## Manual-Only Verifications

All Phase 28 behaviors have automated verification.

---

## Validation Audit 2026-06-13

| Metric | Count |
|--------|-------|
| Requirements audited | 8 |
| Automated coverage found | 8 |
| Gaps found | 0 |
| Resolved by new tests | 0 |
| Escalated manual-only | 0 |

---

## Validation Sign-Off

- [x] All tasks have automated verification or existing test infrastructure coverage.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Focused feedback latency is under 120 seconds.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-06-13
