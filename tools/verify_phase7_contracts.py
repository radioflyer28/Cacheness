#!/usr/bin/env python3
"""Verify Phase 7's fixed offline migration and rebuild contract.

This verifier intentionally uses a literal reviewed inventory.  It neither
consults a Git diff nor discovers tests, services, candidates, or maintenance
runs.  A green result is deterministic local evidence only: live PostgreSQL,
AWS S3, Windows, Python-version qualification, and performance distributions
remain explicit Phase 8 work.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTEST_TIMEOUT_SECONDS = 300

# Keep reviewed and executed inventories distinct.  The comparison below makes
# an accidental edit to the execution tuple observable rather than letting the
# mutable tuple become its own oracle.
_PHASE7_REVIEWED_PRODUCTION_PATHS = (
    "src/cacheness/core.py",
    "src/cacheness/error_handling.py",
    "src/cacheness/handlers.py",
    "src/cacheness/interfaces.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/lifecycle_authority.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/migration_authority.py",
    "src/cacheness/storage/migration_evidence.py",
    "src/cacheness/storage/projections.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/obstore_generation_io.py",
    "docs/STORAGE_MIGRATION.md",
    "docs/STORAGE_INITIALIZATION.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md",
)
PHASE7_PRODUCTION_PATHS = tuple(_PHASE7_REVIEWED_PRODUCTION_PATHS)

_PHASE7_REVIEWED_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/contracts/test_postgresql_lifecycle_authority.py",
    "tests/contracts/test_obstore_generation_io.py",
    "tests/contracts/test_topology_lifecycle.py",
    "tests/test_migration_run_evidence.py",
    "tests/test_projection_sql_atomicity.py",
    "tests/test_migration_remote_contract.py",
    "tests/test_s3_blob_backend.py",
    "tests/test_rebuild_workflow.py",
    "tests/test_handler_registration.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_blob_store_atomic_lifecycle.py",
    "tests/contracts/test_lifecycle_authority.py",
    "tests/test_migration_public_contract.py",
    "tests/test_phase7_contract_verifier.py",
)
PHASE7_TEST_NODES = tuple(_PHASE7_REVIEWED_TEST_NODES)
# The base runtime intentionally does not install optional PostgreSQL drivers.
# Quick feedback therefore exercises every deterministic local module it can
# import and reports this one fixed module as requiring ``--all`` in the locked
# all-extras environment.  It is neither skipped as a pass nor qualification.
PHASE7_QUICK_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/contracts/test_obstore_generation_io.py",
    "tests/contracts/test_topology_lifecycle.py",
    "tests/test_migration_run_evidence.py",
    "tests/test_projection_sql_atomicity.py",
    "tests/test_migration_remote_contract.py",
    "tests/test_s3_blob_backend.py",
    "tests/test_rebuild_workflow.py",
    "tests/test_handler_registration.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_blob_store_atomic_lifecycle.py",
    "tests/contracts/test_lifecycle_authority.py",
    "tests/test_migration_public_contract.py",
    "tests/test_phase7_contract_verifier.py",
)
PHASE8_LIVE_UNQUALIFIED_NODES = (
    "tests/integration/test_postgresql_authority.py",
    "tests/integration/test_remote_topology.py",
    "tests/integration/test_s3_generation.py",
)

_PHASE7_REVIEWED_PLAN_PATHS = (
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-01-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-02-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-03-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-04-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-05-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-06-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-07-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-08-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-09-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-10-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-11-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-12-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-13-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-14-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-15-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-16-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-17-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-18-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-19-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-20-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-21-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-22-PLAN.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-23-PLAN.md",
)
PHASE7_PLAN_PATHS = tuple(_PHASE7_REVIEWED_PLAN_PATHS)
PHASE7_CONTEXT_PATH = (
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md",
)

MIGRATION_REQUIREMENT_NODES = {
    "MIGR-03": (
        "tests/test_migration_inspection.py::test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering",
        "tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them",
        "tests/test_migration_public_contract.py::test_documented_public_workflow_uses_one_model_and_offline_fencing",
    ),
    "MIGR-04": (
        "tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan",
        "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup",
        "tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches",
        "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible",
        "tests/contracts/test_postgresql_lifecycle_authority.py::test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load",
        "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",
        "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",
    ),
    "MIGR-05": (
        "tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss",
        "tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt",
        "tests/test_rebuild_workflow.py::test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work",
        "tests/test_rebuild_workflow.py::test_rebuild_checkpoints_exact_destination_receipts_and_resumes_each_rebuild_state",
        "tests/test_rebuild_workflow.py::test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt",
        "tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",
        "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",
        "tests/test_migration_cutover.py::test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement",
        "tests/test_rebuild_workflow.py::test_rebuild_evidence_rejects_terminal_aborted_cleanup_debt",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_preserves_changed_current_ownership",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",
        "tests/test_rebuild_workflow.py::test_rebuild_evidence_rejects_accepted_cleanup_debt",
    ),
    "MIGR-06": (
        "tests/test_rebuild_workflow.py::test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle",
        "tests/test_migration_plan_contract.py::test_compatibility_edge_requires_exact_destination_dimensions",
        "tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation",
        "tests/test_handler_registration.py::test_registered_handler_rejects_declared_edge_without_concrete_transform",
    ),
}

DECISION_NODES = {
    "D-01": ("tests/test_stored_compatibility.py::test_current_format_two_store_reopens_with_catalog_values_and_payload",),
    "D-02": ("tests/test_migration_plan_contract.py::test_release_window_accepts_only_current_and_immediately_previous_release",),
    "D-03": ("tests/test_migration_plan_contract.py::test_matrix_requires_one_exact_edge_per_changed_dimension", "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",),
    "D-04": ("tests/test_migration_inspection.py::test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering",),
    "D-05": ("tests/test_stored_compatibility.py::test_development_and_foreign_markers_require_offline_migration_without_mutation",),
    "D-06": ("tests/test_migration_inspection.py::test_historical_path_is_rebuild_only_without_writing",),
    "D-07": ("tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan",),
    "D-08": ("tests/test_rebuild_workflow.py::test_rebuild_plan_defaults_to_every_entry_and_cannot_use_migration_stage",),
    "D-09": ("tests/test_migration_run_evidence.py::test_evidence_store_requires_inspection_first_and_rejects_hostile_raw_bytes",),
    "D-10": ("tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation", "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",),
    "D-11": ("tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss",),
    "D-12": ("tests/test_migration_cutover.py::test_candidate_and_evidence_never_authorize_activation",),
    "D-13": ("tests/test_migration_cutover.py::test_memory_tracer_requires_explicit_whole_store_activation",),
    "D-14": ("tests/test_migration_cutover.py::test_offline_service_finalize_requires_exact_confirmation_and_seals_rollback",),
    "D-15": ("tests/test_migration_cutover.py::test_offline_service_abort_removes_only_its_unactivated_candidate",),
    "D-16": ("tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches", "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry", "tests/test_migration_cutover.py::test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "D-17": ("tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup",),
    "D-18": ("tests/test_migration_plan_contract.py::test_canonical_plan_round_trips_and_human_report_uses_the_same_model",),
    "D-19": ("tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt", "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "D-20": ("tests/contracts/test_lifecycle_authority.py::test_local_authorities_expose_exact_operation_replay_without_new_authority_state", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "D-21": ("tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_preserves_changed_current_ownership", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "D-22": ("tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them",),
}

ORIGINAL_PHASE7_THREAT_IDS = (
    *(f"T-07-{number:02d}" for number in range(1, 25)),
    *(f"T-07-{number:02d}" for number in range(26, 30)),
    *(f"T-07-{number:02d}" for number in range(31, 35)),
    *(f"T-07-{number:02d}" for number in range(36, 41)),
    *(f"T-07-{number:02d}" for number in range(42, 51)),
)
GAP_PLAN_THREAT_IDS = (
    "T-07-G12-01", "T-07-G12-02", "T-07-G12-03", "T-07-G12-04", "T-07-G12-05",
    "T-07-13-01", "T-07-13-02", "T-07-13-03", "T-07-13-04", "T-07-13-SC",
    "T-07-14-01",
    "T-07-15-01", "T-07-15-02", "T-07-15-03", "T-07-15-04",
    "T-07-16-01", "T-07-16-02", "T-07-16-03", "T-07-16-04", "T-07-16-05",
    "T-07-17-01", "T-07-17-02", "T-07-17-03", "T-07-17-04", "T-07-17-05", "T-07-17-SC",
    "T-07-18-01", "T-07-18-02", "T-07-18-03", "T-07-18-04", "T-07-18-05", "T-07-18-SC",
    "T-07-19-01", "T-07-19-02", "T-07-19-03", "T-07-19-04", "T-07-19-05", "T-07-19-SC",
    "T-07-20-01", "T-07-20-02", "T-07-20-03", "T-07-20-04",
    "T-07-21-01", "T-07-21-02", "T-07-21-03", "T-07-21-04",
    "T-07-22-01", "T-07-22-02", "T-07-22-03", "T-07-22-04",
    "T-07-23-01", "T-07-23-02", "T-07-23-03", "T-07-23-04",
)
_DECLARED_PHASE7_THREAT_IDS = (*ORIGINAL_PHASE7_THREAT_IDS, *GAP_PLAN_THREAT_IDS)

GAP_PLAN_THREAT_OWNERS = {
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-12-PLAN.md": GAP_PLAN_THREAT_IDS[:5],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-13-PLAN.md": GAP_PLAN_THREAT_IDS[5:10],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-14-PLAN.md": GAP_PLAN_THREAT_IDS[10:11],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-15-PLAN.md": GAP_PLAN_THREAT_IDS[11:15],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-16-PLAN.md": GAP_PLAN_THREAT_IDS[15:20],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-17-PLAN.md": GAP_PLAN_THREAT_IDS[20:26],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-18-PLAN.md": GAP_PLAN_THREAT_IDS[26:32],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-19-PLAN.md": GAP_PLAN_THREAT_IDS[32:38],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-20-PLAN.md": GAP_PLAN_THREAT_IDS[38:42],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-21-PLAN.md": GAP_PLAN_THREAT_IDS[42:46],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-22-PLAN.md": GAP_PLAN_THREAT_IDS[46:50],
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-23-PLAN.md": GAP_PLAN_THREAT_IDS[50:54],
}

SECURITY_THREAT_NODES = {
    "T-07-01": ("tests/test_migration_cutover.py::test_candidate_and_evidence_never_authorize_activation",),
    "T-07-02": ("tests/test_stored_compatibility.py::test_ordinary_entry_points_never_adopt_or_modify_unsupported_roots",),
    "T-07-03": ("tests/test_blob_store_read_contract.py::test_composed_store_reopens_one_authenticated_canonical_generation",),
    "T-07-04": ("tests/test_migration_inspection.py::test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering",),
    "T-07-05": ("tests/test_migration_run_evidence.py::test_evidence_store_round_trips_canonically_and_rejects_forgery",),
    "T-07-06": ("tests/test_migration_plan_contract.py::test_release_window_accepts_only_current_and_immediately_previous_release",),
    "T-07-07": ("tests/test_migration_plan_contract.py::test_matrix_requires_one_exact_edge_per_changed_dimension",),
    "T-07-08": ("tests/test_migration_plan_contract.py::test_matrix_rejects_ambiguous_or_out_of_window_edges",),
    "T-07-09": ("tests/test_migration_plan_contract.py::test_plan_decoder_rejects_duplicate_json_keys",),
    "T-07-10": ("tests/test_migration_plan_contract.py::test_canonical_plan_round_trips_and_human_report_uses_the_same_model",),
    "T-07-11": ("tests/test_migration_run_evidence.py::test_checkpoint_requires_exact_previous_bytes_and_legal_state_transition",),
    "T-07-12": ("tests/test_migration_inspection.py::test_inventory_continuation_rejects_revision_drift_without_refreshing",),
    "T-07-13": ("tests/test_migration_inspection.py::test_initialized_sqlite_store_is_current_without_a_format_marker",),
    "T-07-14": ("tests/test_migration_inspection.py::test_historical_path_is_rebuild_only_without_writing",),
    "T-07-15": ("tests/test_migration_inspection.py::test_corrupt_authority_path_is_refused_without_writing",),
    "T-07-16": ("tests/test_migration_run_evidence.py::test_evidence_never_renders_or_logs_key_material",),
    "T-07-17": ("tests/test_migration_run_evidence.py::test_resume_requires_exact_run_and_evidence_then_revalidates_next_step",),
    "T-07-18": ("tests/test_migration_run_evidence.py::test_resume_refuses_mismatched_output_or_stale_source_without_adoption",),
    "T-07-19": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-20": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-21": ("tests/test_migration_cutover.py::test_memory_tracer_requires_explicit_whole_store_activation",),
    "T-07-22": ("tests/test_migration_cutover.py::test_sqlite_activation_seals_workers_until_explicit_finalize",),
    "T-07-23": ("tests/test_migration_cutover.py::test_offline_service_abort_removes_only_its_unactivated_candidate",),
    "T-07-24": ("tests/test_projection_sql_atomicity.py::test_failed_sql_projection_rebuild_discards_the_candidate_without_publishing",),
    "T-07-26": ("tests/contracts/test_postgresql_lifecycle_authority.py::test_constructor_is_non_materializing_and_initialize_is_explicit",),
    "T-07-27": ("tests/contracts/test_obstore_generation_io.py::test_mocked_s3_collision_and_lost_create_response_settle_only_by_exact_object",),
    "T-07-28": ("tests/contracts/test_topology_lifecycle.py::test_remote_inventory_without_snapshot_attribution_is_indeterminate",),
    "T-07-29": ("tests/contracts/test_obstore_generation_io.py::test_mocked_s3_enforces_direct_put_bounds_and_preserves_exact_maintenance_scope",),
    "T-07-31": ("tests/test_migration_cutover.py::test_sqlite_activation_rollback_keeps_candidate_invisible",),
    "T-07-32": ("tests/test_migration_cutover.py::test_offline_service_rolls_back_only_the_activated_receipt",),
    "T-07-33": ("tests/test_migration_cutover.py::test_offline_service_finalize_requires_exact_confirmation_and_seals_rollback",),
    "T-07-34": ("tests/test_migration_cutover.py::test_offline_service_purge_is_separate_idempotent_retryable_cleanup",),
    "T-07-36": ("tests/test_rebuild_workflow.py::test_rebuild_plan_defaults_to_every_entry_and_cannot_use_migration_stage",),
    "T-07-37": ("tests/test_rebuild_workflow.py::test_rebuild_exclusion_regenerates_exact_plan_and_requires_its_confirmation",),
    "T-07-38": ("tests/test_handler_registration.py::test_invalid_transformation_edges_are_rejected_without_global_fallback",),
    "T-07-39": ("tests/test_rebuild_workflow.py::test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle",),
    "T-07-40": ("tests/test_rebuild_workflow.py::test_rebuild_integrity_failure_runs_before_custom_handler_and_discards_candidates",),
    "T-07-42": ("tests/test_migration_public_contract.py::test_public_storage_maintenance_exports_and_signatures_are_explicit",),
    "T-07-43": ("tests/test_migration_public_contract.py::test_documented_public_workflow_uses_one_model_and_offline_fencing",),
    "T-07-44": ("tests/test_migration_public_contract.py::test_runbook_marks_stop_conditions_and_non_claims",),
    "T-07-45": ("tests/test_migration_public_contract.py::test_ordinary_construction_exposes_no_migration_switch_or_cli",),
    "T-07-46": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_is_complete_and_not_discovery_derived",),
    "T-07-47": ("tests/test_phase7_contract_verifier.py::test_fixed_mapping_validator_rejects_each_omission",),
    "T-07-48": ("tests/test_phase7_contract_verifier.py::test_source_mutation_cannot_drop_a_fixed_test_node",),
    "T-07-49": ("tests/test_phase7_contract_verifier.py::test_architecture_audit_rejects_each_executable_prohibition",),
    "T-07-50": ("tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets",),
    "T-07-G12-01": ("tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan", "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible"),
    "T-07-G12-02": ("tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches",),
    "T-07-G12-03": ("tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches",),
    "T-07-G12-04": ("tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan", "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible"),
    "T-07-G12-05": ("tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan", "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible"),
    "T-07-13-01": ("tests/contracts/test_postgresql_lifecycle_authority.py::test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load",),
    "T-07-13-02": ("tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open",),
    "T-07-13-03": ("tests/contracts/test_postgresql_lifecycle_authority.py::test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load",),
    "T-07-13-04": ("tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open",),
    "T-07-13-SC": ("tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts",),
    "T-07-14-01": ("tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss", "tests/contracts/test_lifecycle_authority.py::test_local_authorities_expose_exact_operation_replay_without_new_authority_state", "tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_read_mutation_returns_exact_prepared_and_promoted_replay"),
    "T-07-15-01": ("tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt", "tests/test_rebuild_workflow.py::test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work"),
    "T-07-15-02": ("tests/test_rebuild_workflow.py::test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt",),
    "T-07-15-03": ("tests/test_rebuild_workflow.py::test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt",),
    "T-07-15-04": ("tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt",),
    "T-07-16-01": ("tests/test_migration_plan_contract.py::test_compatibility_edge_requires_exact_destination_dimensions",),
    "T-07-16-02": ("tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation", "tests/test_handler_registration.py::test_registered_handler_rejects_declared_edge_without_concrete_transform"),
    "T-07-16-03": ("tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible"),
    "T-07-16-04": ("tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity",),
    "T-07-16-05": ("tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible"),
    "T-07-17-01": ("tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them",),
    "T-07-17-02": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-17-03": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-17-04": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-17-05": ("tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",),
    "T-07-17-SC": ("tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts",),
    "T-07-18-01": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors",),
    "T-07-18-02": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains", "tests/test_phase7_contract_verifier.py::test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains"),
    "T-07-18-03": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors",),
    "T-07-18-04": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors",),
    "T-07-18-05": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_requires_exact_path_and_test_name_selectors",),
    "T-07-18-SC": ("tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts",),
    "T-07-19-01": ("tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass",),
    "T-07-19-02": ("tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass",),
    "T-07-19-03": ("tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass",),
    "T-07-19-04": ("tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets",),
    "T-07-19-05": ("tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets",),
    "T-07-19-SC": ("tests/test_phase7_contract_verifier.py::test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts",),
    "T-07-20-01": ("tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",),
    "T-07-20-02": ("tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",),
    "T-07-20-03": ("tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",),
    "T-07-20-04": ("tests/test_migration_cutover.py::test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates",),
    "T-07-21-01": ("tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement",),
    "T-07-21-02": ("tests/test_rebuild_workflow.py::test_rebuild_evidence_rejects_terminal_aborted_cleanup_debt",),
    "T-07-21-03": (
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",
    ),
    "T-07-21-04": ("tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_preserves_changed_current_ownership",),
    "T-07-22-01": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_current_three_gap_repairs_exactly",),
    "T-07-22-02": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_includes_every_gap_plan_threat_exactly_once",),
    "T-07-22-03": ("tests/test_phase7_contract_verifier.py::test_main_never_renders_failed_requirement_as_pass",),
    "T-07-22-04": ("tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets",),
    "T-07-23-01": ("tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "T-07-23-02": ("tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",),
    "T-07-23-03": ("tests/test_rebuild_workflow.py::test_rebuild_evidence_rejects_accepted_cleanup_debt",),
    "T-07-23-04": ("tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly",),
}

FLAGGED_ASSUMPTION_NODES = {
    "A-MIGR03": ("tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them", "tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift"),
    "A-MIGR04": ("tests/test_migration_plan_contract.py::test_compatibility_edge_requires_exact_destination_dimensions", "tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation", "tests/test_migration_cutover.py::test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible", "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest"),
    "A-MIGR05": ("tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches", "tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss", "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_preserves_changed_current_ownership", "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts"),
    "A-MIGR06": ("tests/test_rebuild_workflow.py::test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work", "tests/test_handler_registration.py::test_registered_custom_handler_resolves_one_exact_directed_transformation"),
}

PLAN01_PROHIBITIONS = (
    "No ordinary constructor, open, initialize, read, cache policy, reconciliation, or cleanup path may migrate, rebuild, adopt, activate, or purge a store.",
    "No work-directory file, filesystem path, symlink, object listing, candidate presence, or derived projection may become lifecycle or cutover authority.",
    "No new lock, queue, lease, sidecar, daemon, scheduler, online writer protocol, or second lifecycle state machine may be added.",
    "No historical compatibility reader, manufactured persisted version, universal native payload converter, or force override may be introduced.",
    "No partial candidate may activate, and activation/finalize may not physically delete the prior valid store.",
    "No unauthenticated evidence, stale source fingerprint, unexplained candidate, or incomplete catalog may be adopted.",
    "No signing key bytes, raw credentials, or secret provider paths may appear in plans, reports, evidence, or logs.",
    "No unbounded inventory/evidence or Phase 8 live PostgreSQL/AWS S3/performance qualification claim may enter Phase 7.",
)

PLAN01_PROHIBITION_NODES = {
    PLAN01_PROHIBITIONS[0]: (
        "tests/test_migration_public_contract.py::test_ordinary_construction_exposes_no_migration_switch_or_cli",
    ),
    PLAN01_PROHIBITIONS[1]: (
        "tests/test_migration_cutover.py::test_candidate_and_evidence_never_authorize_activation",
        "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup",
        "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_preserves_changed_current_ownership",
    ),
    PLAN01_PROHIBITIONS[2]: (
        "tests/test_phase7_contract_verifier.py::test_architecture_audit_rejects_each_executable_prohibition",
    ),
    PLAN01_PROHIBITIONS[3]: (
        "tests/test_migration_inspection.py::test_historical_path_is_rebuild_only_without_writing",
        "tests/test_handler_registration.py::test_invalid_transformation_edges_are_rejected_without_global_fallback",
        "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",
    ),
    PLAN01_PROHIBITIONS[4]: (
        "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup",
        "tests/test_migration_cutover.py::test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest",
    ),
    PLAN01_PROHIBITIONS[5]: (
        "tests/test_migration_run_evidence.py::test_resume_refuses_mismatched_output_or_stale_source_without_adoption",
        "tests/test_rebuild_workflow.py::test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access",
    ),
    PLAN01_PROHIBITIONS[6]: (
        "tests/test_migration_plan_contract.py::test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them",
        "tests/test_migration_run_evidence.py::test_evidence_never_renders_or_logs_key_material",
    ),
    PLAN01_PROHIBITIONS[7]: (
        "tests/test_phase7_contract_verifier.py::test_document_and_coverage_audits_reject_false_qualification_and_secrets",
    ),
}

PHASE7_RUFF_PATHS = (
    "src/cacheness/error_handling.py",
    "src/cacheness/handlers.py",
    "src/cacheness/interfaces.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/lifecycle_authority.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/migration_authority.py",
    "src/cacheness/storage/migration_evidence.py",
    "src/cacheness/storage/projections.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/obstore_generation_io.py",
    *PHASE7_TEST_NODES,
    "tests/test_phase7_contract_verifier.py",
)

_ORDINARY_MODULES = frozenset(
    {
        "src/cacheness/core.py",
        "src/cacheness/storage/blob_store.py",
        "src/cacheness/storage/projections.py",
    }
)
_MAINTENANCE_NAMES = frozenset(
    {
        "OfflineMigrationService",
        "inspect_migration_store",
        "create_rebuild_plan",
        "activate_verified_candidate",
        "finalize_verified_candidate",
        "purge",
    }
)


def _dotted_name(node: ast.AST) -> str | None:
    """Return a static dotted name while rejecting dynamic expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _call_name(node: ast.Call) -> str | None:
    """Return the terminal component of a statically known call."""
    dotted = _dotted_name(node.func)
    return None if dotted is None else dotted.rsplit(".", maxsplit=1)[-1]


def _mapping_difference(
    actual: Iterable[str], expected: Iterable[str], label: str
) -> tuple[str, ...]:
    """Render a deterministic fixed-inventory mismatch."""
    actual_set = frozenset(actual)
    expected_set = frozenset(expected)
    if actual_set == expected_set:
        return ()
    return (
        f"{label} differs from the fixed Phase 7 set: "
        f"missing={sorted(expected_set - actual_set)!r}, "
        f"unexpected={sorted(actual_set - expected_set)!r}",
    )


def validate_paths(root: Path, paths: Iterable[str]) -> tuple[str, ...]:
    """Reject root escape and missing fixed artifacts before any execution."""
    errors: list[str] = []
    for path in paths:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"manifest path is not repository-relative: {path}")
        elif not (root / candidate).is_file():
            errors.append(f"manifest artifact is missing: {path}")
    return tuple(errors)


def _parse_selector(selector: str) -> tuple[str, str] | None:
    """Return a safe repository-relative pytest selector or ``None``."""
    if selector.count("::") < 1:
        return None
    path, symbol = selector.split("::", maxsplit=1)
    candidate = PurePosixPath(path)
    if (
        not path.startswith("tests/")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not symbol
        or any(not part.isidentifier() for part in symbol.split("::"))
    ):
        return None
    return candidate.as_posix(), symbol


def _module_symbols(path: Path) -> frozenset[str]:
    """Read pytest-style module and class test names without executing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}::{member.name}")
    return frozenset(symbols)


def _validate_selector_mapping(
    root: Path,
    label: str,
    mapping: Mapping[str, Sequence[str]],
    known_paths: frozenset[str],
) -> tuple[str, ...]:
    """Fail closed before pytest if a claim's exact evidence is malformed."""
    errors: list[str] = []
    symbol_cache: dict[str, frozenset[str]] = {}
    for claim, selectors in mapping.items():
        if not selectors:
            errors.append(f"{label} mapping has no executable evidence: {claim}")
            continue
        if len(selectors) != len(set(selectors)):
            errors.append(f"{label} mapping has duplicate exact selector: {claim}")
        for selector in selectors:
            parsed = _parse_selector(selector)
            if parsed is None:
                errors.append(f"{label} mapping has malformed selector: {claim}: {selector}")
                continue
            path, symbol = parsed
            if path not in known_paths:
                errors.append(
                    f"{label} mapping references non-manifest node: {claim}: {selector}"
                )
                continue
            try:
                symbols = symbol_cache.setdefault(path, _module_symbols(root / path))
            except (OSError, SyntaxError) as error:
                errors.append(
                    f"{label} mapping cannot inspect selector: {claim}: {selector}: {error}"
                )
                continue
            if symbol not in symbols:
                errors.append(
                    f"{label} mapping references missing test selector: {claim}: {selector}"
                )
    return tuple(errors)


def _plan_threat_ids(path: Path) -> tuple[str, ...]:
    """Parse one plan's threat table without interpreting prose outside it."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<threat_model>(.*?)</threat_model>", text, re.DOTALL)
    if match is None:
        raise ValueError("missing threat model")
    return tuple(
        re.findall(r"^\|\s*(T-07-[A-Za-z0-9-]+)\s*\|", match.group(1), re.MULTILINE)
    )


def validate_gap_plan_threat_inventory(root: Path) -> tuple[str, ...]:
    """Require every gap plan to retain its exact reviewed threat ownership."""
    errors: list[str] = []
    observed_all: list[str] = []
    for relative_path, expected in GAP_PLAN_THREAT_OWNERS.items():
        try:
            observed = _plan_threat_ids(root / relative_path)
        except (OSError, ValueError) as error:
            errors.append(f"gap threat inventory unreadable: {relative_path}: {error}")
            continue
        observed_all.extend(observed)
        duplicate_ids = sorted(threat_id for threat_id in set(observed) if observed.count(threat_id) > 1)
        for threat_id in duplicate_ids:
            errors.append(f"gap threat inventory duplicate: {relative_path}: {threat_id}")
        for threat_id in expected:
            if threat_id not in observed:
                errors.append(f"gap threat inventory missing: {relative_path}: {threat_id}")
        for threat_id in observed:
            if threat_id not in expected:
                errors.append(f"gap threat inventory unexpected: {relative_path}: {threat_id}")
    for threat_id in sorted(set(observed_all)):
        if observed_all.count(threat_id) > 1:
            errors.append(f"gap threat inventory has multiple owners: {threat_id}")
    if tuple(sorted(GAP_PLAN_THREAT_IDS)) != tuple(sorted(set(GAP_PLAN_THREAT_IDS))):
        errors.append("literal gap threat inventory has duplicate IDs")
    return tuple(errors)


def audit_gap_plan_command_contracts(root: Path) -> tuple[str, ...]:
    """Keep fixed plan execution on checked-in frozen dependencies only."""
    errors: list[str] = []
    install_action = re.compile(r"\b(?:pip|uv|npm|pnpm|yarn|cargo)\s+(?:install|add|sync|lock)\b")
    for relative_path in GAP_PLAN_THREAT_OWNERS:
        try:
            text = (root / relative_path).read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"gap command contract unreadable: {relative_path}: {error}")
            continue
        for command in re.findall(r"\buv run\b[^\n<`]*", text):
            if "--frozen" not in command.split():
                errors.append(f"gap command lacks --frozen: {relative_path}: {command.strip()}")
        if install_action.search(text):
            errors.append(f"gap plan contains package-install action: {relative_path}")
    return tuple(errors)


def validate_mapping_inventory(
    *,
    requirements: Mapping[str, Sequence[str]],
    decisions: Mapping[str, Sequence[str]],
    threats: Mapping[str, Sequence[str]],
    assumptions: Mapping[str, Sequence[str]],
    prohibitions: Sequence[str],
) -> tuple[str, ...]:
    """Reject a silent coverage-map deletion without inferring replacements."""
    errors: list[str] = []
    errors.extend(
        _mapping_difference(requirements, MIGRATION_REQUIREMENT_NODES, "MIGR mapping")
    )
    errors.extend(_mapping_difference(decisions, DECISION_NODES, "D mapping"))
    errors.extend(
        _mapping_difference(threats, _DECLARED_PHASE7_THREAT_IDS, "Threat mapping")
    )
    errors.extend(
        _mapping_difference(
            assumptions, FLAGGED_ASSUMPTION_NODES, "Flagged-assumption mapping"
        )
    )
    if tuple(prohibitions) != PLAN01_PROHIBITIONS:
        errors.append("Plan 01 prohibition inventory differs from the fixed eight-row set")
    return tuple(errors)


def validate_fixed_manifest(root: Path) -> tuple[str, ...]:
    """Validate the literal path and cross-reference inventories."""
    errors: list[str] = []
    errors.extend(
        _mapping_difference(
            PHASE7_PRODUCTION_PATHS,
            _PHASE7_REVIEWED_PRODUCTION_PATHS,
            "Phase 7 production inventory",
        )
    )
    actual_tests = frozenset(PHASE7_TEST_NODES)
    reviewed_tests = frozenset(_PHASE7_REVIEWED_TEST_NODES)
    if actual_tests != reviewed_tests:
        errors.append(
            "Phase 7 test inventory differs from the fixed reviewed set: "
            f"missing={sorted(reviewed_tests - actual_tests)!r}, "
            f"unexpected={sorted(actual_tests - reviewed_tests)!r}"
        )
    actual_plans = frozenset(PHASE7_PLAN_PATHS)
    reviewed_plans = frozenset(_PHASE7_REVIEWED_PLAN_PATHS)
    if actual_plans != reviewed_plans:
        errors.append(
            "Phase 7 plan inventory differs from the fixed reviewed set: "
            f"missing={sorted(reviewed_plans - actual_plans)!r}, "
            f"unexpected={sorted(actual_plans - reviewed_plans)!r}"
        )
    if errors:
        return tuple(errors)
    errors.extend(validate_paths(root, (*PHASE7_PRODUCTION_PATHS, *PHASE7_TEST_NODES)))
    errors.extend(validate_paths(root, (*PHASE7_PLAN_PATHS, *PHASE7_CONTEXT_PATH)))
    errors.extend(
        validate_mapping_inventory(
            requirements=MIGRATION_REQUIREMENT_NODES,
            decisions=DECISION_NODES,
            threats=SECURITY_THREAT_NODES,
            assumptions=FLAGGED_ASSUMPTION_NODES,
            prohibitions=PLAN01_PROHIBITIONS,
        )
    )
    errors.extend(validate_gap_plan_threat_inventory(root))
    errors.extend(audit_gap_plan_command_contracts(root))
    known_nodes = frozenset(PHASE7_TEST_NODES)
    for label, mapping in (
        ("MIGR", MIGRATION_REQUIREMENT_NODES),
        ("D", DECISION_NODES),
        ("threat", SECURITY_THREAT_NODES),
        ("flagged assumption", FLAGGED_ASSUMPTION_NODES),
        ("prohibition", PLAN01_PROHIBITION_NODES),
    ):
        errors.extend(_validate_selector_mapping(root, label, mapping, known_nodes))
    return tuple(errors)


class _ArchitectureVisitor(ast.NodeVisitor):
    """Audit executable migration shapes while ignoring comments and prose."""

    def __init__(self, filename: str) -> None:
        self.filename = PurePosixPath(filename).as_posix()
        self.findings: list[str] = []
        self._function_context: list[dict[str, bool]] = []

    def _add(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            self.filename == "src/cacheness/storage/migration.py"
            and node.name.endswith("MigrationAuthority")
            and node.name != "MigrationAuthority"
        ):
            self._add(f"second migration authority: {node.name}")
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_context.append({"listing": False, "integrity": False})
        self.generic_visit(node)
        self._function_context.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func) or ""
        name = _call_name(node)
        context = self._function_context[-1] if self._function_context else None
        if self.filename in _ORDINARY_MODULES and name in _MAINTENANCE_NAMES:
            self._add(f"ordinary lifecycle path invokes offline maintenance: {name}")
        if self.filename == "src/cacheness/storage/migration.py":
            if dotted in {"threading.Lock", "threading.RLock", "queue.Queue", "asyncio.Lock", "asyncio.Queue"}:
                self._add(f"migration coordination primitive: {dotted}")
            if name in {"listdir", "list_objects_v2", "list_objects"} and context is not None:
                context["listing"] = True
            if name == "activate_verified_candidate" and context and context["listing"]:
                self._add("candidate/listing adoption reaches activation")
            if name and name.startswith("verify_") and context is not None:
                context["integrity"] = True
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "handler"
                and node.func.attr in {"read", "load", "deserialize"}
                and context is not None
                and not context["integrity"]
            ):
                self._add("handler read precedes integrity verification")
            if name == "inventory_page":
                keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
                if not ({"limit", "page_size"} & keywords) or "work_cap" not in keywords:
                    self._add("unbounded inventory/evidence operation: inventory_page")
        if name in {"print", "warning", "error", "info", "debug"}:
            for argument in node.args:
                if isinstance(argument, ast.Name) and re.search(
                    r"(?:key|secret|credential|password|token)", argument.id, re.I
                ):
                    self._add(f"secret value flows to output: {argument.id}")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return deterministic executable-only architecture findings."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return (f"unreadable source: {error.msg}",)
    visitor = _ArchitectureVisitor(filename)
    visitor.visit(tree)
    return tuple(visitor.findings)


def audit_phase7_text(text: str) -> tuple[str, ...]:
    """Reject only positive support, performance, and secret-rendering claims."""
    normalized = " ".join(text.lower().split())
    findings: list[str] = []
    if (
        "phase 7" in normalized
        and "does not qualify" not in normalized
        and "do not qualify" not in normalized
        and "phase 8" not in normalized
        and "live postgresql" in normalized
        and "aws s3" in normalized
        and re.search(r"qualif|support", normalized)
    ):
        findings.append("false Phase 7 live-service qualification claim")
    if "performance distribution" in normalized and re.search(
        r"guarantee|qualif|pass", normalized
    ):
        findings.append("Phase 7 performance qualification claim")
    if re.search(r"(?:credential|secret|signing)[ _-]?path\s*=", normalized):
        findings.append("secret provider path appears in rendered text")
    if (
        "obstore" in normalized
        and "conditional" in normalized
        and "bounded-memory" in normalized
        and re.search(r"claim|guarantee|is bounded-memory", normalized)
    ):
        findings.append("false obstore bounded-memory conditional-publication claim")
    if (
        "multipart" in normalized
        and "copy" in normalized
        and re.search(r"silent|automatic|selected", normalized)
    ):
        findings.append("silent multipart-copy publication policy claim")
    if re.search(
        r"(?:handlers? receive.{0,80}|expose.{0,80}handlers?).{0,80}"
        r"(?:participant handles?|managed locators?)"
        r"|(?:participant handles?|managed locators?).{0,80}handlers?",
        normalized,
    ):
        findings.append("handler exposure of participant handle or managed locator")
    return tuple(findings)


def audit_coverage_document(coverage: str) -> tuple[str, ...]:
    """Validate the stored detector record and explicit Phase 8 boundary."""
    start_marker = "<!-- phase7-api-coverage:detector-result:start -->\n```json\n"
    end_marker = "\n```\n<!-- phase7-api-coverage:detector-result:end -->"
    try:
        start = coverage.index(start_marker) + len(start_marker)
        end = coverage.index(end_marker, start)
        result = json.loads(coverage[start:end])
    except (ValueError, json.JSONDecodeError) as error:
        return (f"coverage detector record is unreadable: {error}",)
    if not isinstance(result, dict) or result.get("detected") is not True:
        return ("coverage detector result no longer preserves the recorded public-API signal",)
    required = (
        "No external API integration:",
        "PostgresqlLifecycleAuthority",
        "ObstoreGenerationIO",
        "tests/contracts/test_postgresql_lifecycle_authority.py",
        "tests/test_migration_remote_contract.py",
        "tests/contracts/test_s3_generation_io.py",
        "do not qualify live PostgreSQL/AWS S3",
        "Phase 8 alone",
    )
    if any(item not in coverage for item in required):
        return ("coverage declaration omits a required deterministic-adapter or Phase 8 boundary",)
    if "| capability | decision | reason |" in coverage.lower():
        return ("coverage declaration fabricates an external capability matrix",)
    if "qualifies live PostgreSQL/AWS S3" in coverage:
        return ("coverage declaration makes a false live-service qualification claim",)
    return ()


def _assignment_value(source: str, variable: str) -> object | None:
    """Read a simple assigned literal without treating comments as evidence."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
                if isinstance(node.value, ast.Constant):
                    return node.value.value
    return None


def _audit_persisted_baselines(root: Path) -> tuple[str, ...]:
    """Pin the released SQLite/PostgreSQL publication baselines to 9 and 5."""
    sqlite = (root / "src/cacheness/storage/sqlite_lifecycle_authority.py").read_text(
        encoding="utf-8"
    )
    postgresql = (
        root / "src/cacheness/storage/backends/postgresql_lifecycle_authority.py"
    ).read_text(encoding="utf-8")
    errors: list[str] = []
    if _assignment_value(sqlite, "SQLITE_USER_VERSION") != 9:
        errors.append("SQLite authority publication schema is not pinned to 9")
    if _assignment_value(postgresql, "POSTGRESQL_AUTHORITY_SCHEMA_VERSION") != 5:
        errors.append("PostgreSQL authority publication schema is not pinned to 5")
    if _assignment_value(postgresql, "POSTGRESQL_AUTHORITY_CAPABILITY") != "postgresql-lifecycle-authority-v5":
        errors.append("PostgreSQL authority capability is not pinned to schema 5")
    return tuple(errors)


def _audit_worker_fence(root: Path) -> tuple[str, ...]:
    """Check that ordinary BlobStore entry reuses the authority offline fence."""
    source = (root / "src/cacheness/storage/blob_store.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="src/cacheness/storage/blob_store.py")
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    canonical = methods.get("_require_canonical_store")
    if canonical is None:
        return ("BlobStore lacks the ordinary-worker canonical-store fence",)
    has_fence_lookup = any(
        isinstance(node, ast.Call)
        and _call_name(node) == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "require_ordinary_worker_access"
        for node in ast.walk(canonical)
    )
    has_fence_call = any(
        isinstance(node, ast.Call) and _call_name(node) == "require_worker_access"
        for node in ast.walk(canonical)
    )
    if not (has_fence_lookup and has_fence_call):
        return ("BlobStore canonical-store fence does not reject activated_offline workers",)
    return ()


def _phase7_detector_scope(root: Path) -> str:
    """Assemble the exact reviewed scope passed to the active detector."""
    roadmap = (root / ".planning/milestones/v1.0-ROADMAP.md").read_text(encoding="utf-8")
    start = roadmap.index("### Phase 7: Explicit Migration and Rebuild Cutover")
    end = roadmap.index("### Phase 8: Production Gates and Performance Stabilization", start)
    bodies: list[str] = []
    for relative_path in PHASE7_PLAN_PATHS:
        plan = (root / relative_path).read_text(encoding="utf-8")
        parts = plan.split("\n---\n", 1)
        if len(parts) != 2:
            raise ValueError(f"{relative_path} lacks a frontmatter/body boundary")
        bodies.append(parts[1])
    return roadmap[start:end] + "".join(bodies)


def _detector_path() -> Path | None:
    """Locate only the installed GSD detector; never substitute a local clone."""
    roots: list[Path] = []
    configured_root = os.environ.get("CODEX_HOME")
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path.home() / ".codex")
    for root in roots:
        candidate = root / "gsd-core/bin/lib/api-coverage.cjs"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _validate_live_detector(root: Path, coverage: str) -> tuple[str, ...]:
    """Compare stored detector JSON against the active runtime's actual output."""
    detector = _detector_path()
    if detector is None:
        return ("active GSD API-coverage detector is unavailable",)
    start_marker = "<!-- phase7-api-coverage:detector-result:start -->\n```json\n"
    end_marker = "\n```\n<!-- phase7-api-coverage:detector-result:end -->"
    start = coverage.index(start_marker) + len(start_marker)
    end = coverage.index(end_marker, start)
    stored = json.loads(coverage[start:end])
    try:
        completed = subprocess.run(
            ["node", str(detector), "--json"],
            input=_phase7_detector_scope(root),
            text=True,
            capture_output=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return (f"active GSD API-coverage detector could not run: {error}",)
    if completed.returncode not in {0, 1}:
        return (f"active GSD API-coverage detector exited {completed.returncode}: {completed.stderr.strip()}",)
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return (f"active GSD API-coverage detector emitted invalid JSON: {error}",)
    if observed != stored:
        return ("stored API-coverage detector result differs from the active detector",)
    return ()


def _run_pytest(
    root: Path,
    nodes: Sequence[str],
    label: str,
    *,
    options: Sequence[str] = (),
) -> tuple[bool, str]:
    """Run one finite pytest invocation and preserve non-green output."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *nodes,
        *options,
        "-o",
        "log_cli=false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        return False, f"{label}: timed out after {PYTEST_TIMEOUT_SECONDS} seconds\n{error.output or ''}"
    except OSError as error:
        return False, f"{label}: could not start pytest: {error}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"{label}: pytest exited {completed.returncode}\n{output}"
    return True, f"{label}: {output}"


def fixed_pytest_nodes(quick: bool) -> tuple[str, ...]:
    """Return the ordered, de-duplicated exact evidence selectors for one mode."""
    allowed_paths = frozenset(
        PHASE7_QUICK_TEST_NODES if quick else PHASE7_TEST_NODES
    )
    nodes: list[str] = []
    for mapping in (
        MIGRATION_REQUIREMENT_NODES,
        DECISION_NODES,
        SECURITY_THREAT_NODES,
        FLAGGED_ASSUMPTION_NODES,
        PLAN01_PROHIBITION_NODES,
    ):
        for selectors in mapping.values():
            for selector in selectors:
                parsed = _parse_selector(selector)
                if parsed is not None and parsed[0] in allowed_paths and selector not in nodes:
                    nodes.append(selector)
    return tuple(nodes)


def _run_ruff(root: Path) -> tuple[bool, str]:
    """Lint only the reviewed Phase 7 Python surface; no baseline is hidden."""
    command = [sys.executable, "-m", "ruff", "check", *PHASE7_RUFF_PATHS]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"Phase 7 scoped Ruff could not run: {error}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"Phase 7 scoped Ruff exited {completed.returncode}\n{output}"
    return True, f"Phase 7 scoped Ruff: {output or 'PASS'}"


def verify_repository(root: Path, quick: bool) -> tuple[bool, tuple[str, ...]]:
    """Run fixed static checks and the reviewed behavioral evidence.

    ``quick`` omits only the full repository suite and scoped Ruff.  It never
    turns an unavailable external service into a skip/pass and never makes a
    performance deadline part of storage correctness.
    """
    root = root.resolve()
    errors: list[str] = []
    errors.extend(validate_fixed_manifest(root))
    for relative_path in (
        "src/cacheness/core.py",
        "src/cacheness/storage/blob_store.py",
        "src/cacheness/storage/migration.py",
        "src/cacheness/storage/migration_evidence.py",
        "src/cacheness/storage/projections.py",
    ):
        try:
            errors.extend(audit_source((root / relative_path).read_text(encoding="utf-8"), relative_path))
        except OSError as error:
            errors.append(f"architecture source unreadable: {relative_path}: {error}")
    try:
        errors.extend(_audit_persisted_baselines(root))
        errors.extend(_audit_worker_fence(root))
    except (OSError, SyntaxError) as error:
        errors.append(f"persisted publication audit unreadable: {error}")
    coverage_path = root / ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md"
    try:
        coverage = coverage_path.read_text(encoding="utf-8")
        errors.extend(audit_coverage_document(coverage))
        if not quick:
            errors.extend(_validate_live_detector(root, coverage))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"coverage audit unreadable: {error}")
    for relative_path in (
        "docs/STORAGE_MIGRATION.md",
        "docs/STORAGE_INITIALIZATION.md",
    ):
        try:
            errors.extend(audit_phase7_text((root / relative_path).read_text(encoding="utf-8")))
        except OSError as error:
            errors.append(f"documentation audit unreadable: {relative_path}: {error}")
    context_path = root / PHASE7_CONTEXT_PATH[0]
    try:
        context = context_path.read_text(encoding="utf-8")
        for decision in DECISION_NODES:
            if decision not in context:
                errors.append(f"Phase 7 context omits declared decision: {decision}")
    except OSError as error:
        errors.append(f"Phase 7 context audit unreadable: {error}")
    if not errors:
        nodes = fixed_pytest_nodes(quick)
        passed, evidence = _run_pytest(root, nodes, "fixed Phase 7 behavioral inventory")
        if not passed:
            errors.append(evidence)
    if not quick and not errors:
        live_options = tuple(f"--ignore={node}" for node in PHASE8_LIVE_UNQUALIFIED_NODES)
        passed, evidence = _run_pytest(
            root,
            (),
            "full deterministic non-live suite",
            options=live_options,
        )
        if not passed:
            errors.append(evidence)
        else:
            passed, evidence = _run_ruff(root)
            if not passed:
                errors.append(evidence)
    return not errors, tuple(errors)


def _diagnostic_requirements(errors: Iterable[str]) -> frozenset[str]:
    """Map failed behavior or static evidence to the public MIGR labels."""
    labels: set[str] = set()
    for error in errors:
        for requirement in MIGRATION_REQUIREMENT_NODES:
            if error.startswith(f"{requirement}:"):
                labels.add(requirement)
        if error.startswith(
            (
                "MIGR mapping",
                "D mapping",
                "Threat mapping",
                "Flagged-assumption mapping",
                "Plan 01 prohibition",
                "Phase 7 test inventory",
                "Phase 7 production inventory",
            )
        ):
            labels.update(MIGRATION_REQUIREMENT_NODES)
        if error.startswith(("ordinary lifecycle", "SQLite authority", "PostgreSQL authority", "BlobStore canonical")):
            labels.update({"MIGR-04", "MIGR-05"})
        if error.startswith(("candidate/listing", "handler read", "unbounded", "secret value")):
            labels.update({"MIGR-03", "MIGR-05", "MIGR-06"})
    return frozenset(labels)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 7 verifier from any current working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="run fixed phase evidence")
    mode.add_argument("--all", action="store_true", help="also run full suite and scoped Ruff")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)

    passed, errors = verify_repository(args.repo_root, args.quick)
    failed_requirements = _diagnostic_requirements(errors)
    print("Phase 7 fixed migration/rebuild contract verifier")
    for requirement in MIGRATION_REQUIREMENT_NODES:
        status = "see diagnostics" if requirement in failed_requirements else "PASS"
        print(f"{requirement}: {status}")
    print("SQLite/PostgreSQL publication baseline: 9/5 deterministic contract")
    if args.quick:
        print(
            "Deterministic PostgreSQL adapter contract: NOT RUN in --quick; "
            "--all requires the locked all-extras environment"
        )
    else:
        print(
            "Phase 8 live PostgreSQL/AWS S3 modules: NOT RUN and NOT QUALIFIED; "
            "they are excluded by the fixed non-live inventory"
        )
    print("Remote/platform/performance: Phase 8 only; deterministic adapters are not qualification")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Fixed Phase 7 inventory: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
