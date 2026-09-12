---
phase: 7
slug: explicit-migration-and-rebuild-cutover
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-09
updated: 2026-09-11
---

# Phase 7 — Validation Evidence

Phase 7 has fixed deterministic local evidence for explicit offline migration and rebuild. It does not qualify live PostgreSQL/AWS S3, Windows, a Python-version matrix, or performance distributions.

## Fixed Acceptance Boundary

tools/verify_phase7_contracts.py owns literal reviewed tuples for production/public/documentation paths, Plans 01–11, focused test nodes, requirements, decisions, 46 declared security threats, four flagged assumptions, and the eight Plan 01 prohibitions. It rejects missing, duplicate, or root-escaping entries before execution and never derives scope from Git, test discovery, comments, strings, a latest run, a candidate, or a live-service observation.

Self-tests mutate source inventory and exercise adversarial executable fixtures for ordinary maintenance reachability, a second authority, coordination primitives, candidate/listing adoption, handler reads before verification, unbounded inventory, secret output, false qualification, altered coverage evidence, and false PASS rendering.

## Execution Evidence

| Command | Environment | Exit | Result |
|---|---|---:|---|
| uv run --frozen pytest -q tests/test_phase7_contract_verifier.py -x -o log_cli=false | frozen base Python 3.13.15 | 0 | 14 verifier self-tests passed |
| uv run --frozen python tools/verify_phase7_contracts.py --quick | frozen base Python 3.13.15 | 0 | fixed local quick inventory passed; deterministic PostgreSQL contract explicitly NOT RUN because base lacks optional psycopg |
| uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all | isolated locked all-extras/dev Python 3.13.15 | 0 | fixed Phase 7 inventory, active detector comparison, deterministic non-live suite, and scoped Ruff passed |
| uv run --isolated --all-extras --group dev --frozen pytest -q [15 fixed Phase 7 modules] --junitxml=… -o log_cli=false | isolated locked all-extras/dev Python 3.13.15 | 0 | 140 tests; 0 failures, 0 errors, 0 skips |
| uv run --isolated --all-extras --group dev --frozen pytest -q --ignore=tests/integration/test_postgresql_authority.py --ignore=tests/integration/test_remote_topology.py --ignore=tests/integration/test_s3_generation.py --junitxml=… -o log_cli=false | isolated locked all-extras/dev Python 3.13.15 | 0 | 1311 tests; 0 failures, 0 errors, 9 documented platform/optional skips |
| uv run --isolated --all-extras --group dev --frozen ruff check [fixed Phase 7 Python inventory] | isolated locked all-extras/dev Python 3.13.15 | 0 | all scoped paths clean |

The all gate invokes the active installed gsd-core/bin/lib/api-coverage.cjs --json over the fixed Roadmap/Plan scope and compares its typed result to 07-COVERAGE.md.

## Gap-closure fixed-verifier evidence (Plan 07-19)

The following observations supersede neither the original evidence above nor
the Phase 8 boundary. They record the post-gap exact-selector run. A row is
only marked complete after the fixed verifier returns zero; quick mode validates
the full manifest but intentionally does not execute the optional PostgreSQL
adapter selector.

| Command | Environment | Exit | Observed result |
|---|---|---:|---|
| `uv run --frozen python tools/verify_phase7_contracts.py --quick` | frozen base CPython 3.13.15 | 0 | Exact-manifest, architecture, documentation, baseline, coverage-record, 84-threat, and quick-selector checks passed. The verifier reported `MIGR-03` through `MIGR-06` as `PASS`; deterministic PostgreSQL adapter evidence is explicitly `NOT RUN` in quick mode. |
| `uv run --frozen python -c '… fixed_pytest_nodes(True) … pytest -q …'` | frozen base CPython 3.13.15 | 0 | The verifier-owned quick selector union contained 66 unique exact selectors; execution completed with no failure or skip indication. |
| `uv run --frozen python -c '… fixed_pytest_nodes(True) … pytest --collect-only -q -o addopts= …'` | frozen base CPython 3.13.15 | 0 | The exact quick union collected 80 test cases (including parametrized cases), with no collection error. This is a count observation only; the fixed verifier remains the completion authority. |
| `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all` | isolated locked all-extras/dev CPython 3.13.15 | 0 | Fixed artifacts, exact selectors, 84-threat inventory, active coverage-detector comparison, deterministic non-live suite, and scoped Ruff all passed. Live PostgreSQL/AWS S3 modules remained explicitly NOT RUN / NOT QUALIFIED. |
| `uv run --isolated --all-extras --group dev --frozen python -c '… fixed_pytest_nodes(False) … pytest -q -o addopts= …'` | isolated locked all-extras/dev CPython 3.13.15 | 0 | The verifier-owned all selector union contained 70 exact selectors; 84 passed, 0 skipped, 0 failed. |
| `uv run --isolated --all-extras --group dev --frozen pytest -q --ignore=tests/integration/test_postgresql_authority.py --ignore=tests/integration/test_remote_topology.py --ignore=tests/integration/test_s3_generation.py -o log_cli=false -o addopts=` | isolated locked all-extras/dev CPython 3.13.15 | 0 | 1331 passed, 9 documented platform/optional skips, 0 failed. The three ignored live-service modules are Phase 8 work, not skips or passes. |

### Authoritative blocker rows

| Blocker | Exact regression evidence bound by the verifier | Quick status | Final status |
|---|---|---|---|
| CR-01 — bounded migration attribution and STAGING recovery | `test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan`; `test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup`; `test_resume_and_abort_staging_use_only_authority_attributed_batches`; `test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible` | PASS | PASS (`--all`) |
| CR-02 — exact rebuild response-loss recovery and cleanup | `test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss`; `test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt`; `test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work`; `test_rebuild_checkpoints_exact_destination_receipts_and_resumes_each_rebuild_state`; `test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt` | PASS | PASS (`--all`) |
| CR-03 — destination compatibility and guarded handler transformation | `test_compatibility_edge_requires_exact_destination_dimensions`; `test_registered_custom_handler_resolves_one_exact_directed_transformation`; `test_registered_handler_rejects_declared_edge_without_concrete_transform`; `test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity` | PASS | PASS (`--all`) |
| CR-04 — digest-bound confidential machine plans | `test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them`; `test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift` | PASS | PASS (`--all`) |
| CR-05 — fresh PostgreSQL post-load ordinary-worker fence | `test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load`; `test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open` | Manifest validated; not in quick environment | PASS (`--all`, deterministic adapter only) |
| CR-06 — fail-closed exact-selector verifier | `test_fixed_manifest_requires_exact_path_and_test_name_selectors`; `test_fixed_manifest_includes_every_gap_plan_threat_exactly_once`; `test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains`; `test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains`; `test_fixed_manifest_rejects_gap_threat_without_exact_selector` | PASS | PASS (`--all`) |

### Accepted recovery/progress limit and non-implementation boundary

The quick verifier's passing CR-01 selectors cover the accepted checker
override without increasing its guarantee: every run has explicit entry, byte,
and evidence limits; larger stores split into independently recoverable runs;
and resume/abort operate only on authority-attributed effects. A crash after
immutable publication but before the authority checkpoint can leave an
invisible, unattributed, unadopted orphan outside guaranteed exact cleanup.
Integrity and authority visibility remain guaranteed. This is ADR 0001's
topology-specific recovery/progress limit, not a failed atomicity claim.

For CR-02, the evidence is intentionally narrower than a second maintenance
lifecycle: rebuild staging calls the existing lifecycle engine through the
canonical maintenance wrapper, re-derives operation IDs, adds no maintenance
pre-publication intent, journal, or state, replays the exact receipt with empty
projection outcomes, and performs derived work only after explicit acceptance.

The obstore spike remains a non-implementation boundary in Phase 7. Cacheness
retains payload-participant/lifecycle authority ownership, the path-based
handler contract, and guarded staging/snapshots. The spike's streaming and S3
conditional-publication findings are PARTIAL; no obstore package, participant
cutover, conditional-publication policy, or multipart-copy policy is claimed
by this ledger.

### Non-live suite inventory

The non-live suite excludes exactly these Phase 8-owned real-service modules. They are NOT RUN / NOT QUALIFIED, never skips or passes:

- tests/integration/test_postgresql_authority.py
- tests/integration/test_remote_topology.py
- tests/integration/test_s3_generation.py

## Plan Task Ledger

| Task ID | Named evidence | Status |
|---|---|---|
| 07-01-T1 | tests/test_migration_cutover.py | ✅ |
| 07-01-T2 | tests/test_stored_compatibility.py | ✅ |
| 07-02-T1 | accepted D-02 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-02-T2 | accepted D-08 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-02-T3 | accepted D-15 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-03-T1 | tests/test_migration_plan_contract.py | ✅ |
| 07-03-T2 | plan-contract, inspection, and stored-compatibility tests | ✅ |
| 07-04-T1 | inspection and lifecycle-authority tests | ✅ |
| 07-04-T2 | PostgreSQL lifecycle-authority contract | ✅ deterministic adapter only |
| 07-05-T1 | tests/test_migration_run_evidence.py | ✅ |
| 07-05-T2 | evidence and cutover tests | ✅ |
| 07-06-T1 | accepted RQ-01; schema 8/4 verifier pin | ✅ |
| 07-06-T2 | cutover and lifecycle-authority tests | ✅ |
| 07-06-T3 | cutover, evidence, and projection SQL tests | ✅ |
| 07-07-T1 | PostgreSQL lifecycle-authority contract | ✅ deterministic adapter only |
| 07-07-T2 | remote-contract and S3 backend tests | ✅ deterministic adapter only |
| 07-08-T1 | cutover and lifecycle-authority tests | ✅ |
| 07-08-T2 | cutover and evidence tests | ✅ |
| 07-09-T1 | tests/test_rebuild_workflow.py | ✅ |
| 07-09-T2 | tests/test_handler_registration.py | ✅ |
| 07-09-T3 | rebuild, handler, and BlobStore read-contract tests | ✅ |
| 07-10-T1 | public-contract and stored-compatibility tests | ✅ |
| 07-10-T2 | test_external_api_coverage_declaration_is_detector_backed | ✅ |
| 07-11-T1 | verifier self-tests and fixed quick gate | ✅ |
| 07-11-T2 | fixed all gate, focused inventory, non-live suite, scoped Ruff | ✅ |

## Requirement, Decision, and Research Coverage

| Item | Named executable evidence | Status |
|---|---|---|
| MIGR-03 | inspection, plan-contract, and public-contract tests | ✅ |
| MIGR-04 | cutover, stored-compatibility, and lifecycle-authority tests | ✅ |
| MIGR-05 | evidence, cutover, projection SQL, and remote-contract tests | ✅ |
| MIGR-06 | rebuild, handler-registration, and BlobStore read-contract tests | ✅ |
| D-01…D-22 | literal DECISION_NODES map checked for exact equality and non-empty fixed-module evidence | ✅ |
| RQ-01 | SQLite 8 and PostgreSQL schema/capability 4 AST pins plus authority contracts | ✅ deterministic only |
| RQ-02 | public no-CLI test plus literal public/docs inventory | ✅ |
| RQ-03 | cutover/evidence tests and BlobStore ordinary-worker-fence audit | ✅ |
| RQ-04 | handler registration, rebuild, and BlobStore read-contract tests | ✅ |

The four former flagged assumptions remain the exact FLAGGED_ASSUMPTION_NODES map keyed by MIGR-03 through MIGR-06; the verifier fails if any key disappears.

## Security and Prohibition Coverage

The original 46 declared threat IDs have literal non-empty SECURITY_THREAT_NODES entries; the verifier rejects any missing or unexpected ID. That initial reviewed set is exactly T-07-01…T-07-24; T-07-26…T-07-29; T-07-31…T-07-34; T-07-36…T-07-40; T-07-42…T-07-50. The final 84-ID inventory, including all 38 gap-plan rows, is recorded below.

Named maps use inventory evidence for T-07-04/T-07-12…T-07-15; evidence security for T-07-05/T-07-11/T-07-16…T-07-20; authority/cutover for T-07-01/T-07-21…T-07-23/T-07-31…T-07-34; remote adapters for T-07-26…T-07-29; handler/rebuild for T-07-36…T-07-40; public boundary for T-07-42…T-07-45; and final verifier tamper/support evidence for T-07-46…T-07-50.

The ordered eight-row PLAN01_PROHIBITIONS inventory is mutation-tested. Its executable backstops are:

1. ordinary paths cannot reach offline maintenance;
2. listings, candidates, and projections cannot authorize activation;
3. migration code cannot add a lock, queue, or second authority;
4. behavior tests reject historical readers, force paths, and universal conversions;
5. cutover tests reject partial activation and retain the prior store through finalize;
6. evidence/cutover tests reject stale, unauthenticated, unexplained, and incomplete inputs;
7. evidence/public/verifier tests reject raw signing keys, credentials, and provider paths in output; and
8. bounded inventory/evidence and Phase 8 non-claim checks reject unbounded calls and inflated qualification language.

## Exact 84-threat selector ledger

The all-mode verifier statically validated SECURITY_THREAT_NODES, parsed the
eight gap-plan threat tables against its independent 38-ID ownership oracle,
and then executed the deterministic exact-selector union: 84 tests passed with
zero skips and zero failures. The complete inventory has 84 unique IDs: the
original 46 plus the 38 rows introduced by Plans 07-12 through 07-19. There
were zero missing, duplicate, or unexpected threat rows.

| Threat ID | Exact selector(s) | Status |\n|---|---|---|
| T-07-01 | tests/test_migration_cutover.py::test_candidate_and_evidence_never_authorize_activation | PASS (all) |
| T-07-02 | tests/test_stored_compatibility.py::test_ordinary_entry_points_never_adopt_or_modify_unsupported_roots | PASS (all) |
| T-07-03 | tests/test_blob_store_read_contract.py::test_composed_store_reopens_one_authenticated_canonical_generation | PASS (all) |
| T-07-04 | tests/test_migration_inspection.py::test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering | PASS (all) |
| T-07-05 | tests/test_migration_run_evidence.py::test_evidence_store_round_trips_canonically_and_rejects_forgery | PASS (all) |
| T-07-06 | tests/test_migration_plan_contract.py::test_release_window_accepts_only_current_and_immediately_previous_release | PASS (all) |
| T-07-07 | tests/test_migration_plan_contract.py::test_matrix_requires_one_exact_edge_per_changed_dimension | PASS (all) |
| T-07-08 | tests/test_migration_plan_contract.py::test_matrix_rejects_ambiguous_or_out_of_window_edges | PASS (all) |
| T-07-09 | tests/test_migration_plan_contract.py::test_plan_decoder_rejects_duplicate_json_keys | PASS (all) |
| T-07-10 | tests/test_migration_plan_contract.py::test_canonical_plan_round_trips_and_human_report_uses_the_same_model | PASS (all) |
| T-07-11 | tests/test_migration_run_evidence.py::test_checkpoint_requires_exact_previous_bytes_and_legal_state_transition | PASS (all) |
| T-07-12 | tests/test_migration_inspection.py::test_inventory_continuation_rejects_revision_drift_without_refreshing | PASS (all) |
| T-07-13 | tests/test_migration_inspection.py::test_initialized_sqlite_store_is_current_without_a_format_marker | PASS (all) |
| T-07-14 | tests/test_migration_inspection.py::test_historical_path_is_rebuild_only_without_writing | PASS (all) |
| T-07-15 | tests/test_migration_inspection.py::test_corrupt_authority_path_is_refused_without_writing | PASS (all) |
| T-07-16 | tests/test_migration_run_evidence.py::test_evidence_never_renders_or_logs_key_material | PASS (all) |
| T-07-17 | tests/test_migration_run_evidence.py::test_resume_requires_exact_run_and_evidence_then_revalidates_next_step | PASS (all) |
| T-07-18 | tests/test_migration_run_evidence.py::test_resume_refuses_mismatched_output_or_stale_source_without_adoption | PASS (all) |
| T-07-19 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-20 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-21 | tests/test_migration_cutover.py::test_memory_tracer_requires_explicit_whole_store_activation | PASS (all) |
| T-07-22 | tests/test_migration_cutover.py::test_sqlite_activation_seals_workers_until_explicit_finalize | PASS (all) |
| T-07-23 | tests/test_migration_cutover.py::test_offline_service_abort_removes_only_its_unactivated_candidate | PASS (all) |
| T-07-24 | tests/test_projection_sql_atomicity.py::test_failed_sql_projection_rebuild_discards_the_candidate_without_publishing | PASS (all) |
| T-07-26 | tests/contracts/test_postgresql_lifecycle_authority.py::test_constructor_is_non_materializing_and_initialize_is_explicit | PASS (all) |
| T-07-27 | tests/test_migration_remote_contract.py::test_s3_candidate_response_loss_revalidates_exact_receipt_without_listing | PASS (all) |
| T-07-28 | tests/test_migration_remote_contract.py::test_s3_candidate_rejects_unowned_locator_before_mutating | PASS (all) |
| T-07-29 | tests/test_migration_remote_contract.py::test_s3_candidate_rejects_unowned_locator_before_mutating | PASS (all) |
| T-07-31 | tests/test_migration_cutover.py::test_sqlite_activation_rollback_keeps_candidate_invisible | PASS (all) |
| T-07-32 | tests/test_migration_cutover.py::test_offline_service_rolls_back_only_the_activated_receipt | PASS (all) |
| T-07-33 | tests/test_migration_cutover.py::test_offline_service_finalize_requires_exact_confirmation_and_seals_rollback | PASS (all) |
| T-07-34 | tests/test_migration_cutover.py::test_offline_service_purge_is_separate_idempotent_retryable_cleanup | PASS (all) |
| T-07-36 | tests/test_rebuild_workflow.py::test_rebuild_plan_defaults_to_every_entry_and_cannot_use_migration_stage | PASS (all) |
| T-07-37 | tests/test_rebuild_workflow.py::test_rebuild_exclusion_regenerates_exact_plan_and_requires_its_confirmation | PASS (all) |
| T-07-38 | tests/test_handler_registration.py::test_invalid_transformation_edges_are_rejected_without_global_fallback | PASS (all) |
| T-07-39 | tests/test_rebuild_workflow.py::test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle | PASS (all) |
| T-07-40 | tests/test_rebuild_workflow.py::test_rebuild_integrity_failure_runs_before_custom_handler_and_discards_candidates | PASS (all) |
| T-07-42 | tests/test_migration_public_contract.py::test_public_storage_maintenance_exports_and_signatures_are_explicit | PASS (all) |
| T-07-43 | tests/test_migration_public_contract.py::test_documented_public_workflow_uses_one_model_and_offline_fencing | PASS (all) |
| T-07-44 | tests/test_migration_public_contract.py::test_runbook_marks_stop_conditions_and_non_claims | PASS (all) |
| T-07-45 | tests/test_migration_public_contract.py::test_ordinary_construction_exposes_no_migration_switch_or_cli | PASS (all) |
| T-07-46 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_is_complete_and_not_discovery_derived | PASS (all) |
| T-07-47 | tests/test_phase7_contract_verifier.py::test_fixed_mapping_validator_rejects_each_omission | PASS (all) |
| T-07-48 | tests/test_phase7_contract_verifier.py::test_source_mutation_cannot_drop_a_fixed_test_node | PASS (all) |
| T-07-49 | tests/test_phase7_contract_verifier.py::test_architecture_audit_rejects_each_executable_prohibition | PASS (all) |
| T-07-50 | tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets | PASS (all) |
| T-07-G12-01 | tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan<br>tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup<br>tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible | PASS (all) |
| T-07-G12-02 | tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches | PASS (all) |
| T-07-G12-03 | tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches | PASS (all) |
| T-07-G12-04 | tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan<br>tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup<br>tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible | PASS (all) |
| T-07-G12-05 | tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan<br>tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup<br>tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible | PASS (all) |
| T-07-13-01 | tests/contracts/test_postgresql_lifecycle_authority.py::test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load | PASS (all) |
| T-07-13-02 | tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open | PASS (all) |
| T-07-13-03 | tests/contracts/test_postgresql_lifecycle_authority.py::test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load | PASS (all) |
| T-07-13-04 | tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open | PASS (all) |
| T-07-13-SC | tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts | PASS (all) |
| T-07-14-01 | tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss<br>tests/contracts/test_lifecycle_authority.py::test_local_authorities_expose_exact_operation_replay_without_new_authority_state<br>tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_read_mutation_returns_exact_prepared_and_promoted_replay | PASS (all) |
| T-07-15-01 | tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt<br>tests/test_rebuild_workflow.py::test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work | PASS (all) |
| T-07-15-02 | tests/test_rebuild_workflow.py::test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt | PASS (all) |
| T-07-15-03 | tests/test_rebuild_workflow.py::test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt | PASS (all) |
| T-07-15-04 | tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt | PASS (all) |
| T-07-16-01 | tests/test_migration_plan_contract.py::test_compatibility_edge_requires_exact_destination_dimensions | PASS (all) |
| T-07-16-02 | tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation<br>tests/test_handler_registration.py::test_registered_handler_rejects_declared_edge_without_concrete_transform | PASS (all) |
| T-07-16-03 | tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity<br>tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible | PASS (all) |
| T-07-16-04 | tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity | PASS (all) |
| T-07-16-05 | tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity<br>tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible | PASS (all) |
| T-07-17-01 | tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them | PASS (all) |
| T-07-17-02 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-17-03 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-17-04 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-17-05 | tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift | PASS (all) |
| T-07-17-SC | tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts | PASS (all) |
| T-07-18-01 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors | PASS (all) |
| T-07-18-02 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains<br>tests/test_phase7_contract_verifier.py::test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains | PASS (all) |
| T-07-18-03 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors | PASS (all) |
| T-07-18-04 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors | PASS (all) |
| T-07-18-05 | tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors | PASS (all) |
| T-07-18-SC | tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts | PASS (all) |
| T-07-19-01 | tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass | PASS (all) |
| T-07-19-02 | tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass | PASS (all) |
| T-07-19-03 | tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass | PASS (all) |
| T-07-19-04 | tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets | PASS (all) |
| T-07-19-05 | tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets | PASS (all) |
| T-07-19-SC | tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts | PASS (all) |


## Known Pre-existing Suite Observation

Before the non-live inventory correction, the initial all-extras gate returned exit 1 because it incorrectly included the three Phase 8 live modules (nine missing-fixture errors) and also observed one tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot CacheBlobLifecycleConflictError. The correction changed only the fixed Phase 8 live-module inventory; it did not alter lifecycle/storage code or turn that failure into a pass. The repeated fixed non-live suite passed 1311 tests with zero failures/errors. The one-off concurrency observation remains in deferred-items.md as pre-existing Phase 3 evidence; ADR 0001 and the approved Wave 1 boundary prohibit reopening a race-patch loop here.

## Phase 8 Non-claims

- Live PostgreSQL and AWS S3: NOT QUALIFIED. Deterministic adapter coverage is not real-service evidence.
- Windows/native platform evidence and the supported Python-version matrix: NOT QUALIFIED. The deterministic run retained nine documented platform/optional skips.
- Performance distributions, budgets, and tail-latency claims: NOT QUALIFIED. No Phase 7 timing is a runtime correctness or progress guarantee.

## Plan 07-19 final gap-closure status

| Requirement or review item | Exact fixed evidence | Status |
|---|---|---|
| MIGR-03 | digest-bound plan, authenticated execution reread, exact fixed verifier evidence | PASS (`--all`) |
| MIGR-04 | bounded split runs, attributed-only migration recovery, exact destination contracts, PostgreSQL ordinary-worker fence | PASS (`--all`; PostgreSQL is deterministic adapter evidence only) |
| MIGR-05 | authority-attributed recovery, replayable rebuild receipts, receipt-bound cleanup debt, stale-plan rejection | PASS (`--all`) |
| MIGR-06 | explicit rebuild, directed concrete handler transformation, destination manifest identity, projection-free canonical replay | PASS (`--all`) |
| WR-01 | directory-entry durability is an accepted non-blocking warning; evidence loss fails closed and no stronger directory-durability claim is made | ACCEPTED — no claim expansion |
| WR-02 | declared handler transformation edges require a concrete implementation; exact reject/resolve selectors passed | COVERED (`--all`) |

`07-COVERAGE.md` remains the authoritative no-external-API declaration. The
all-mode detector comparison passed without inventing a capability matrix.
This ledger does not claim live PostgreSQL/S3, Windows, a supported-Python
matrix, performance evidence, obstore adoption, a conditional-publication
policy, or perfect reclamation of invisible pre-checkpoint orphans.

## Sign-off

- [x] Every plan task has named deterministic evidence.
- [x] Fixed requirement, decision, threat, assumption, and prohibition maps have no silent row drop.
- [x] The all-mode verifier passed: 84 exact selector cases passed with zero selector skips/failures; its full deterministic non-live suite passed 1331 tests with nine documented non-qualification skips.
- [x] All six former blockers have exact all-mode evidence, and every declared 84-threat selector row is listed above.
- [x] Integrity, recovery/progress, and performance claims remain separate.
- [x] wave_0_complete: true
- [x] nyquist_compliant: true

Approval: deterministic local Phase 7 gate complete; Phase 8 owns every listed non-qualification boundary.

## Current verified gap-cycle evidence — Plans 07-20 through 07-22 (2026-09-11)

This dated section supersedes only the stale Plan 07-19 completion disposition:
the prior 84-threat/70-selector report did not bind the three verifier-identified
behaviors. Historical execution observations above remain intact. The repaired
literal verifier now owns Plans 01–22, exactly 50 unique gap-plan threat rows,
and exactly 96 unique Phase 7 threat IDs. It validates each literal path and
AST-level `path::test_name` selector before pytest, so removing a current-gap
plan row or behavior cannot be hidden by an adjacent passing module.

| Command | Environment | Exit | Observed result |
|---|---|---:|---|
| `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick` | isolated locked all-extras/dev | 0 | `MIGR-03` through `MIGR-06` reported `PASS`; 75 exact quick selectors cover 92 passing cases. The deterministic PostgreSQL adapter selector remains outside quick mode. |
| `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all` | isolated locked all-extras/dev | 0 | Fixed manifest, 50-gap/96-total threat ownership, active coverage-detector comparison, 79 exact all selectors, its deterministic non-live suite, and scoped Ruff passed. Live PostgreSQL/AWS S3 modules remained NOT RUN / NOT QUALIFIED. |
| `uv run --isolated --all-extras --group dev --frozen python -c '… fixed_pytest_nodes(False) … pytest -q …'` | isolated locked all-extras/dev | 0 | 79 exact all selectors executed 96 passing cases with zero selector skips or failures. |
| `uv run --isolated --all-extras --group dev --frozen pytest -q --ignore=tests/integration/test_postgresql_authority.py --ignore=tests/integration/test_remote_topology.py --ignore=tests/integration/test_s3_generation.py -o log_cli=false -o addopts=` | isolated locked all-extras/dev | 1 | 1344 passed, 9 skipped, and the known unrelated `tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot` raised `CacheBlobLifecycleConflictError`. This direct rerun is recorded as a Phase 3 concurrency observation, not rendered as PASS, retried, or repaired in Phase 7. |

| Requirement or review item | Exact current evidence | Current disposition |
|---|---|---|
| MIGR-03 | Existing non-mutating inspection, canonical-plan, and public-workflow selectors remain in the fixed map. | PASS (`--all`) |
| MIGR-04 | `test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest`; `test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry`. | PASS (`--all`) |
| MIGR-05 | Typed S3 abort debt/retry, `test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates`, rebuild debt settlement, terminal-evidence rejection, forged-debt refusal, and changed-current-owner preservation. | PASS (`--all`) |
| MIGR-06 | Existing registered-handler/destination-BlobStore rebuild selectors plus exact receipt-bound settlement and changed-owner preservation. | PASS (`--all`) |
| WR-01 | Plan 07-20's exact partial-abort receipt-count selector proves `deleted_entries` reflects only deletion or proven absence in the current call. | CLOSED (`--all`) |
| WR-02 | The custom-handler `name=` alias behavior is unrelated to these three migration gaps and is recorded below in `deferred-items.md`. | DEFERRED — no Phase 7 completion claim |

The current requirements are marked PASS only because every selector named by
their fixed maps executed successfully in the all-mode evidence. The later
direct full-suite failure above has no mapped current-gap selector and is kept
as a visible non-PASS observation rather than concealed by the all-mode result.

### Preserved boundaries

- `07-COVERAGE.md` remains the authoritative no-external-API declaration.
- A crash after immutable publication and before authority checkpoint may leave
  an invisible, unattributed, unadopted orphan outside exact reclamation; this
  accepted ADR 0001 limit is not reopened.
- Metadata authority and filesystem/S3 payload effects are not one
  cross-resource ACID transaction. Typed operational interruption leaves exact
  attributed debt; integrity, receipt, ownership, and forged-evidence failures
  remain fail closed.
- Live PostgreSQL/Amazon-S3 services, Windows, supported-Python matrix,
  packaging, and performance qualification remain Phase 8 work. Deterministic
  S3/PostgreSQL adapters are not live-service qualification.
- No production obstore adoption, handler stream cutover, managed-locator
  exposure, or conditional-publication policy is claimed by Phase 7.
