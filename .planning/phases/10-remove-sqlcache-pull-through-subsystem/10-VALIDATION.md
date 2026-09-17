---
phase: 10
slug: remove-sqlcache-pull-through-subsystem
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-17
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for removing the SQL pull-through subsystem while proving that the supported BlobStore and UnifiedCache surfaces remain installable and usable.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase10_sqlcache_removal.py tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py tests/test_phase4_cutover_verifier.py tests/test_phase6_contract_verifier.py tests/test_phase071_contract_verifier.py tests/test_phase9_documentation.py tests/packaging/test_wheel_matrix.py` |
| **Full suite command** | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` |
| **Estimated runtime** | Quick: ~90 seconds; full: ~8 minutes |

---

## Sampling Rate

- **After every task commit:** Run the narrowest affected test node plus `uv lock --check` after dependency or lockfile changes.
- **After every plan wave:** Run the quick command above.
- **Before `$gsd-verify-work`:** Run the full non-live suite, scoped Ruff, reference scan, and fresh-wheel acceptance proof.
- **Max feedback latency:** 10 minutes.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-W0-01 | TBD | 0 | CACH-07 | T-10-01 | Removed names cannot be resurrected through aliases, hooks, or package metadata | source/API contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py` | ❌ W0 | ⬜ pending |
| 10-W0-02 | TBD | 0 | CACH-07 | T-10-02 | Built artifacts omit the removed module and dependency surface | packaging integration | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py` | ✅ update | ⬜ pending |
| 10-DEL-01 | TBD | TBD | CACH-07 | T-10-01 | Public imports fail naturally and no compatibility tombstone remains | unit/source contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py` | ❌ W0 / ✅ update | ⬜ pending |
| 10-DEP-01 | TBD | TBD | CACH-07 | T-10-02 | DuckDB and the `sql` extra are absent while retained SQLAlchemy/PostgreSQL features remain | manifest/lock contract | `uv lock --check` | ✅ update | ⬜ pending |
| 10-DOC-01 | TBD | TBD | CACH-07 | T-10-03 | Current docs state the supported boundary and do not direct users to removed APIs | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py tests/test_phase9_documentation.py` | ❌ W0 / ✅ update | ⬜ pending |
| 10-REG-01 | TBD | TBD | CACH-07 | — | BlobStore, UnifiedCache, handlers, and non-live PostgreSQL behavior remain intact | regression | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` | ✅ | ⬜ pending |
| 10-PKG-01 | TBD | TBD | CACH-07 | T-10-02 | A source-free wheel install provides working BlobStore/UnifiedCache round trips and no removed module or metadata | artifact acceptance | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py` | ✅ update | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase10_sqlcache_removal.py` — negative source, API, reference, manifest, and untouched-caller-data boundary for CACH-07.
- [ ] Extend `tools/run_phase8_packaging.py` and `tests/packaging/test_wheel_matrix.py` — wheel-member and installed-metadata absence proof.
- [ ] Refactor fixed verifier manifests before deleting the three dedicated SqlCache test modules.
- [ ] Update Phase 9 documentation assertions to name exact cutover-note owners instead of using blanket bans.

No new test framework, package, service, or fixture is required.

---

## Manual-Only Verifications

All Phase 10 behaviors have automated verification. Live PostgreSQL and remote S3 qualification remain outside this deletion phase because Phase 10 does not change those participants.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verification or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verification.
- [ ] Wave 0 covers all missing references.
- [ ] No watch-mode flags.
- [ ] Feedback latency < 10 minutes.
- [ ] `nyquist_compliant: true` set in frontmatter after plan-task mapping is finalized.

**Approval:** pending plan checker
