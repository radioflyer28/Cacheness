---
phase: 10
slug: remove-sqlcache-pull-through-subsystem
status: ready
nyquist_compliant: true
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
| 10-01-01 | 10-01 | 1 | CACH-07 | T-10-01/T-10-04 | Negative source/API/reference/manifest boundary is literal and collectable | source/API contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase10_sqlcache_removal.py` | ❌ create | ⬜ pending |
| 10-01-02 | 10-01 | 1 | CACH-07 | T-10-01 | Reusable public and isolation contracts express natural absence | public contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py` | ✅ update | ⬜ pending |
| 10-02-01 | 10-02 | 2 | CACH-07 | T-10-05 | Phase 4 verifier no longer loads deleted diagnostics | verifier contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase4_cutover_verifier.py` | ✅ update | ⬜ pending |
| 10-02-02 | 10-02 | 2 | CACH-07 | T-10-06 | Phase 6 verifier owns only CACH-01 through CACH-06 | verifier contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase6_contract_verifier.py` | ✅ update | ⬜ pending |
| 10-02-03 | 10-02 | 2 | CACH-07 | T-10-06 | Phase 07.1 points to the Phase 10 negative owner | verifier contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase071_contract_verifier.py` | ✅ update | ⬜ pending |
| 10-03-01 | 10-03 | 1 | CACH-07 | T-10-02/T-10-SC | One WheelArtifact binds member and metadata inspection | packaging unit | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/packaging/test_wheel_matrix.py` | ✅ update | ⬜ pending |
| 10-03-02 | 10-03 | 1 | CACH-07 | T-10-02 | Integration expectations cover stale member/metadata failure | packaging contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/packaging/test_wheel_matrix.py` | ✅ update | ⬜ pending |
| 10-04-01 | 10-04 | 1 | CACH-07 | T-10-07 | Static gates retain useful AST coverage without deleted paths | static contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase1_quality_gates.py` | ✅ update | ⬜ pending |
| 10-04-02 | 10-04 | 1 | CACH-07 | T-10-07 | Full-suite invocation remains frozen and current | environment contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_full_suite_environment.py` | ✅ update | ⬜ pending |
| 10-04-03 | 10-04 | 1 | CACH-07 | T-10-08 | Only three canonical docs own the bounded cutover note | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase9_documentation.py` | ✅ update | ⬜ pending |
| 10-05-01 | 10-05 | 3 | CACH-07 | T-10-01 | Runtime/module/exports disappear and version stays fixed | source/import contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py::test_public_names_and_module_are_naturally_absent tests/test_phase10_sqlcache_removal.py::test_package_version_remains_unchanged` | ✅ after 10-01 | ⬜ pending |
| 10-05-02 | 10-05 | 3 | CACH-07 | T-10-09 | Orphan reasons disappear while retained public owners pass | public regression | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py` | ✅ update | ⬜ pending |
| 10-05-03 | 10-05 | 3 | CACH-07 | T-10-09 | Three dedicated tests are absent and suite collection succeeds | collection contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only` | ✅ delete | ⬜ pending |
| 10-06-01 | 10-06 | 4 | CACH-07 | T-10-10/T-10-SC | DuckDB/sql group is absent while retained dependencies remain exact | manifest/lock contract | `uv lock --check` | ✅ update | ⬜ pending |
| 10-06-02 | 10-06 | 4 | CACH-07 | T-10-02 | Fresh wheel proves member/metadata absence and local round trips | artifact acceptance | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py` | ✅ update | ⬜ pending |
| 10-07-01 | 10-07 | 2 | CACH-07 | T-10-11 | Three dedicated guides are absent with no dead links | documentation deletion | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py` | ✅ delete | ⬜ pending |
| 10-07-02 | 10-07 | 2 | CACH-07 | T-10-11 | Dedicated examples are absent and canonical journeys pass | example regression | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_quality_workflow.py` | ✅ delete | ⬜ pending |
| 10-07-03 | 10-07 | 2 | CACH-07 | T-10-03 | Mixed obsolete demos are absent without historical erasure | source contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py::test_dedicated_docs_and_examples_are_absent` | ✅ delete | ⬜ pending |
| 10-08-01 | 10-08 | 2 | CACH-07 | T-10-08/T-10-04 | Canonical notes state exact use cases, non-replacement, and untouched tables | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py tests/test_public_api_contract.py` | ✅ update | ⬜ pending |
| 10-08-02 | 10-08 | 2 | CACH-07 | T-10-03 | Mixed docs retain dataframe/platform value without product claims | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py -k 'reference or documentation' tests/test_phase9_documentation.py` | ✅ update | ⬜ pending |
| 10-09-01 | 10-09 | 5 | CACH-07 | T-10-12 | Primary current architecture guidance matches final tree | current-map contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py` | ✅ update | ⬜ pending |
| 10-09-02 | 10-09 | 5 | CACH-07 | T-10-10/T-10-12 | Stack/integration/structure/testing maps match final ownership | current-map/lock contract | `uv lock --check` | ✅ update | ⬜ pending |
| 10-09-03 | 10-09 | 5 | CACH-07 | T-10-02/T-10-10 | Focused, Ruff, lock, full non-live, reference, and wheel gates pass | phase acceptance | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` | ✅ | ⬜ pending |

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

- [x] All tasks have `<automated>` verification or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verification.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 10 minutes.
- [x] `nyquist_compliant: true` set in frontmatter after plan-task mapping is finalized.

**Approval:** pending plan checker; task mapping finalized for Plans 10-01 through 10-09
