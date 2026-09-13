#!/usr/bin/env python3
"""Run the closed, deterministic acceptance contract for Phase 07.1.

This is intentionally an inventory, not a test-discovery helper.  The phase
cutover has a small, security-sensitive proof surface: changing a reviewed
selector, plan, decision, or threat must fail before pytest is allowed to make
an incomplete matrix look green.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_DIRECTORY = (
    ".planning/phases/07.1-obstore-payload-participant-unification"
)
PHASE071_CONTEXT_PATH = f"{PHASE_DIRECTORY}/07.1-CONTEXT.md"

# These reviewed tuples are deliberately literal.  Do not derive them from the
# planning directory: discovery would let a deleted plan or requirement vanish
# from the verifier alongside its evidence.
_REVIEWED_PLAN_PATHS = (
    f"{PHASE_DIRECTORY}/07.1-01-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-02-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-03-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-04-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-05-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-06-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-07-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-08-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-09-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-10-PLAN.md",
    f"{PHASE_DIRECTORY}/07.1-11-PLAN.md",
)
PHASE071_PLAN_PATHS = tuple(_REVIEWED_PLAN_PATHS)

_REVIEWED_REQUIREMENTS = (
    "BACK-01",
    "BACK-02",
    "BACK-03",
    "BACK-04",
    "BACK-06",
    "BACK-07",
    "CACH-01",
    "CACH-02",
    "CACH-03",
    "MIGR-01",
    "MIGR-02",
    "MIGR-03",
    "MIGR-04",
    "MIGR-05",
    "MIGR-06",
    "MIGR-07",
    "QUAL-01",
    "QUAL-04",
    "QUAL-07",
    "SECU-01",
    "SECU-03",
    "SECU-04",
    "SECU-05",
    "SECU-07",
    "SECU-08",
    "STOR-01",
    "STOR-02",
    "STOR-03",
    "STOR-04",
    "STOR-05",
    "STOR-06",
    "STOR-07",
    "STOR-08",
)
PHASE071_REQUIREMENTS = tuple(_REVIEWED_REQUIREMENTS)

_REVIEWED_PLAN_THREATS = {
    "01": ("T-07.1-01-SC",),
    "02": ("T-07.1-02-01", "T-07.1-02-02", "T-07.1-02-03", "T-07.1-02-SC"),
    "03": (
        "T-07.1-03-01",
        "T-07.1-03-02",
        "T-07.1-03-03",
        "T-07.1-03-04",
        "T-07.1-03-05",
    ),
    "04": (
        "T-07.1-04-01",
        "T-07.1-04-02",
        "T-07.1-04-03",
        "T-07.1-04-04",
        "T-07.1-04-05",
    ),
    "05": (
        "T-07.1-05-01",
        "T-07.1-05-02",
        "T-07.1-05-03",
        "T-07.1-05-04",
        "T-07.1-05-05",
        "T-07.1-05-06",
    ),
    "06": ("T-07.1-06-01", "T-07.1-06-02", "T-07.1-06-03", "T-07.1-06-04"),
    "07": ("T-07.1-07-01", "T-07.1-07-02", "T-07.1-07-03", "T-07.1-07-04"),
    "08": (
        "T-07.1-08-01",
        "T-07.1-08-02",
        "T-07.1-08-03",
        "T-07.1-08-04",
        "T-07.1-08-05",
    ),
    "09": ("T-07.1-09-01", "T-07.1-09-02", "T-07.1-09-03", "T-07.1-09-04"),
    "10": ("T-07.1-10-01", "T-07.1-10-02", "T-07.1-10-03", "T-07.1-10-04"),
    "11": (
        "T-07.1-11-01",
        "T-07.1-11-02",
        "T-07.1-11-03",
        "T-07.1-11-04",
        "T-07.1-11-05",
    ),
}

_REVIEWED_THREAT_NODES = {
    "T-07.1-01-SC": (
        "tests/contracts/test_obstore_sdk_parity.py::test_consumed_sync_object_primitives_have_exact_immutable_parity",
    ),
    "T-07.1-02-01": (
        "tests/contracts/test_obstore_sdk_parity.py::test_s3_sdk_accepts_only_supported_bounded_transport_options",
    ),
    "T-07.1-02-02": (
        "tests/contracts/test_obstore_sdk_parity.py::test_error_classification_keeps_known_absence_narrow",
    ),
    "T-07.1-02-03": (
        "tests/contracts/test_obstore_sdk_parity.py::test_http_endpoint_override_is_limited_to_local_moto",
    ),
    "T-07.1-02-SC": (
        "tests/contracts/test_obstore_sdk_parity.py::test_moto_s3_fixture_never_reintroduces_owner_pinning",
    ),
    "T-07.1-03-01": (
        "tests/contracts/test_obstore_generation_io.py::test_local_store_provider_round_trips_native_npz_through_blob_store",
    ),
    "T-07.1-03-02": (
        "tests/contracts/test_obstore_generation_io.py::test_local_and_memory_providers_round_trip_one_store_local_mcap_handler",
    ),
    "T-07.1-03-03": (
        "tests/contracts/test_obstore_generation_io.py::test_accepted_then_lost_create_response_is_settled_by_exact_identity",
    ),
    "T-07.1-03-04": (
        "tests/contracts/test_obstore_generation_io.py::test_hostile_locator_is_rejected_before_any_obstore_call",
    ),
    "T-07.1-03-05": (
        "tests/contracts/test_obstore_generation_io.py::test_transfer_bounds_fail_before_unbounded_upload_or_download",
    ),
    "T-07.1-04-01": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_provider_completes_one_authoritative_lifecycle_for_native_and_mcap",
    ),
    "T-07.1-04-02": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_collision_and_lost_create_response_settle_only_by_exact_object",
    ),
    "T-07.1-04-03": (
        "tests/contracts/test_obstore_generation_io.py::test_s3_configuration_rejects_unsafe_or_unsupported_account_boundaries",
    ),
    "T-07.1-04-04": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_enforces_direct_put_bounds_and_preserves_exact_maintenance_scope",
    ),
    "T-07.1-04-05": (
        "tests/test_guarded_handler_io.py::test_stage_rejects_non_regular_or_unsafe_native_handler_artifacts",
    ),
    "T-07.1-05-01": (
        "tests/test_payload_transport_evidence.py::test_signed_observation_verifies_only_for_its_exact_immutable_identity",
    ),
    "T-07.1-05-02": (
        "tests/test_payload_transport_evidence.py::test_tampered_evidence_fails_closed",
        "tests/test_payload_transport_evidence.py::test_malformed_or_unknown_evidence_is_rejected_before_verification",
        "tests/test_payload_transport_evidence.py::test_noncanonical_transport_evidence_bytes_are_rejected",
    ),
    "T-07.1-05-03": (
        "tests/test_payload_transport_evidence.py::test_transport_comparison_cannot_claim_canonical_payload_verification",
    ),
    "T-07.1-05-04": (
        "tests/test_payload_faults.py::test_verified_participant_observation_is_signed_in_the_one_authority_transition",
    ),
    "T-07.1-05-05": (
        "tests/test_payload_faults.py::test_recovery_delete_faults_are_intent_or_cleanup_debt_not_rollback",
    ),
    "T-07.1-05-06": (
        "tests/test_metadata_only_authority_updates.py::test_catalog_and_metadata_updates_use_one_authority_cas_without_payload_io",
    ),
    "T-07.1-06-01": (
        "tests/test_manifest_versions.py::test_current_manifest_declares_sqlite_authority_schema_nine",
    ),
    "T-07.1-06-02": (
        "tests/test_sqlite_lifecycle_authority.py::test_sqlite_authority_reads_back_required_pragmas_and_identity",
    ),
    "T-07.1-06-03": (
        "tests/contracts/test_lifecycle_authority.py::test_local_tiers_share_safety_without_claiming_equal_progress",
    ),
    "T-07.1-06-04": (
        "tests/contracts/test_lifecycle_authority.py::test_postgresql_sqlstate_progress_is_typed_bounded_and_causal",
    ),
    "T-07.1-07-01": (
        "tests/test_blob_store_read_contract.py::test_composed_store_reopens_one_authenticated_canonical_generation",
    ),
    "T-07.1-07-02": (
        "tests/test_blob_store_read_contract.py::test_transport_comparison_matches_one_committed_generation_without_reading",
        "tests/test_payload_transport_evidence.py::test_opaque_etag_is_preserved_without_digest_interpretation",
    ),
    "T-07.1-07-03": (
        "tests/test_blob_store_read_contract.py::test_transport_comparison_match_never_bypasses_canonical_read_verification",
    ),
    "T-07.1-07-04": (
        "tests/test_blob_store_read_contract.py::test_unsupported_legacy_layout_is_rejected_without_mutation",
    ),
    "T-07.1-08-01": (
        "tests/test_migration_cutover.py::test_memory_tracer_requires_explicit_whole_store_activation",
    ),
    "T-07.1-08-02": (
        "tests/test_migration_cutover.py::test_maintenance_request_requires_stopped_workers_and_separate_work_dir",
    ),
    "T-07.1-08-03": (
        "tests/test_migration_remote_contract.py::test_remote_participant_publishes_one_exact_generation_without_inventory",
    ),
    "T-07.1-08-04": (
        "tests/test_migration_run_evidence.py::test_evidence_store_round_trips_canonically_and_rejects_forgery",
    ),
    "T-07.1-08-05": (
        "tests/test_rebuild_workflow.py::test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle",
    ),
    "T-07.1-09-01": (
        "tests/test_blob_store_composition.py::test_legacy_selectors_factories_and_constructor_overload_are_absent",
    ),
    "T-07.1-09-02": (
        "tests/test_supported_topologies.py::test_builtin_catalog_has_only_the_three_declared_reference_pairs",
    ),
    "T-07.1-09-03": (
        "tests/contracts/test_topology_lifecycle.py::test_remote_profile_uses_one_engine_and_bounded_authority_pages",
    ),
    "T-07.1-09-04": (
        "tests/test_phase6_policy_contract.py::test_policy_imports_no_obstore_transport_and_keeps_one_blob_store",
    ),
    "T-07.1-10-01": (
        "tests/test_migration_public_contract.py::test_payload_cutover_removes_legacy_runtime_mechanics_and_exports",
    ),
    "T-07.1-10-02": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_provider_completes_one_authoritative_lifecycle_for_native_and_mcap",
    ),
    "T-07.1-10-03": (
        "tests/contracts/test_obstore_sdk_parity.py::test_moto_s3_fixture_never_reintroduces_owner_pinning",
    ),
    "T-07.1-10-04": (
        "tests/test_migration_public_contract.py::test_runbook_marks_stop_conditions_and_non_claims",
    ),
    "T-07.1-11-01": (
        "tests/test_full_suite_environment.py::test_phase071_runtime_extras_keep_boto3_in_test_tooling_only",
    ),
    "T-07.1-11-02": (
        "tests/test_full_suite_environment.py::test_phase071_clean_wheel_base_and_selected_extras_cut_over_to_obstore",
    ),
    "T-07.1-11-03": (
        "tests/qualification/test_live_evidence.py::test_live_qualification_requires_an_explicit_region_without_owner_pinning",
    ),
    "T-07.1-11-04": (
        "tests/test_phase071_contract_verifier.py::test_documentation_states_current_d16_integrity_and_phase8_boundaries",
    ),
    "T-07.1-11-05": (
        "tests/test_phase071_contract_verifier.py::test_source_mutation_cannot_drop_a_reviewed_plan_or_threat",
    ),
}
THREAT_NODES = dict(_REVIEWED_THREAT_NODES)

DECISION_NODES = {
    "D-01": (
        "tests/contracts/test_obstore_generation_io.py::test_create_precondition_is_the_other_typed_immutable_collision",
    ),
    "D-02": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_enforces_direct_put_bounds_and_preserves_exact_maintenance_scope",
    ),
    "D-03": (
        "tests/test_payload_transport_evidence.py::test_opaque_etag_is_preserved_without_digest_interpretation",
    ),
    "D-04": (
        "tests/contracts/test_obstore_generation_io.py::test_local_and_memory_share_exact_immutable_generation_contract",
    ),
    "D-05": (
        "tests/contracts/test_obstore_sdk_parity.py::test_consumed_sync_object_primitives_have_exact_immutable_parity",
    ),
    "D-06": (
        "tests/test_full_suite_environment.py::test_phase071_runtime_extras_keep_boto3_in_test_tooling_only",
    ),
    "D-07": (
        "tests/test_migration_public_contract.py::test_payload_cutover_removes_legacy_runtime_mechanics_and_exports",
    ),
    "D-08": (
        "tests/test_blob_store_read_contract.py::test_unsupported_legacy_layout_is_rejected_without_mutation",
    ),
    "D-09": (
        "tests/test_guarded_handler_io.py::test_stage_retains_a_private_suffix_preserving_regular_file",
    ),
    "D-10": (
        "tests/test_metadata_only_authority_updates.py::test_catalog_and_metadata_updates_use_one_authority_cas_without_payload_io",
    ),
    "D-11": (
        "tests/test_payload_transport_evidence.py::test_signed_observation_verifies_only_for_its_exact_immutable_identity",
    ),
    "D-12": (
        "tests/test_blob_store_read_contract.py::test_transport_comparison_matches_one_committed_generation_without_reading",
    ),
    "D-13": (
        "tests/contracts/test_obstore_generation_io.py::test_mocked_s3_provider_completes_one_authoritative_lifecycle_for_native_and_mcap",
    ),
    "D-14": (
        "tests/test_blob_store_composition.py::test_builtin_payload_composition_has_no_legacy_transport_factory_imports",
    ),
    "D-15": (
        "tests/test_migration_public_contract.py::test_runbook_marks_stop_conditions_and_non_claims",
    ),
    "D-16": (
        "tests/contracts/test_obstore_sdk_parity.py::test_moto_s3_fixture_never_reintroduces_owner_pinning",
    ),
}

COVERAGE_NODES = {
    "obstore participant primitives": (
        "tests/contracts/test_obstore_sdk_parity.py::test_consumed_sync_object_primitives_have_exact_immutable_parity",
    ),
    "canonical read and transport comparison": (
        "tests/test_blob_store_read_contract.py::test_transport_comparison_match_never_bypasses_canonical_read_verification",
    ),
    "cache boundary": (
        "tests/test_unified_cache_lifecycle_authority.py::test_topology_form_creates_one_owned_blob_store",
    ),
    "public boundary": (
        "tests/test_migration_public_contract.py::test_public_storage_maintenance_exports_and_signatures_are_explicit",
    ),
    "package boundary": (
    "tests/test_full_suite_environment.py::test_phase071_clean_wheel_base_and_selected_extras_cut_over_to_obstore",
    ),
}

# Every node here is a repository-relative path::test_name, so removing or
# renaming a test fails in validate_selector before it can be skipped by pytest.
PHASE071_TEST_NODES = tuple(
    dict.fromkeys(
        selector
        for mapping in (DECISION_NODES, THREAT_NODES, COVERAGE_NODES)
        for selectors in mapping.values()
        for selector in selectors
    )
)

# The local PostgreSQL authority contract is deterministic but depends on the
# optional driver.  It belongs to --all under that declared extra, never to the
# sub-30-second participant/evidence quick gate; live PostgreSQL remains a
# Phase 8 non-claim.
POSTGRESQL_NODES = (
    "tests/contracts/test_lifecycle_authority.py::test_postgresql_sqlstate_progress_is_typed_bounded_and_causal",
    "tests/contracts/test_postgresql_lifecycle_authority.py::test_postgresql_verification_replays_exact_transport_evidence",
)
PHASE071_QUICK_NODES = tuple(
    node for node in PHASE071_TEST_NODES if node not in POSTGRESQL_NODES
)

_ALL_ONLY_NODES = (
    "tests/test_guarded_handler_io.py::test_stage_rejects_non_regular_or_unsafe_native_handler_artifacts",
    "tests/test_payload_faults.py::test_integrity_and_recovery_put_boundaries_converge_to_one_complete_generation",
    "tests/contracts/test_lifecycle_authority.py::test_memory_authority_replaces_only_mutable_manifest_metadata_with_one_cas",
    "tests/test_sqlite_lifecycle_authority.py::test_sqlite_authority_reopens_exact_transport_evidence",
    "tests/test_migration_cutover.py::test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity",
    "tests/test_migration_remote_contract.py::test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry",
    "tests/test_migration_run_evidence.py::test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift",
    "tests/test_rebuild_workflow.py::test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts",
    "tests/test_payload_reconciliation.py::test_reconciliation_preserves_participant_inventory_cursor_as_report_only",
    "tests/test_blob_store_composition.py::test_selected_filesystem_participant_supplies_generation_io_at_its_own_root",
    "tests/test_supported_topologies.py::test_builtin_remote_roles_expose_only_the_qualified_participants",
    "tests/contracts/test_topology_lifecycle.py::test_remote_inventory_without_snapshot_attribution_is_indeterminate",
    "tests/test_phase6_policy_contract.py::test_replacement_conflict_preserves_winner_and_returns_typed_retryable_result",
    "tests/test_unified_cache_lifecycle_authority.py::test_policy_replacement_keeps_the_latest_authoritative_generation",
    "tests/test_unified_cache_adversarial_lifecycle.py::test_replacement_after_an_invalidation_snapshot_is_not_deleted",
    "tests/test_phase6_removal_contract.py::test_predicate_and_global_clear_delegate_exact_removal_to_blob_store",
    "tests/test_phase6_decorator_contract.py::test_function_clear_preserves_a_concurrently_replaced_generation",
    "tests/test_migration_public_contract.py::test_external_api_coverage_declaration_is_detector_backed",
    "tests/test_public_api_contract.py::test_optional_sqlcache_surface_remains_separate_when_dependency_is_blocked",
    "tests/test_full_suite_environment.py::test_phase071_runtime_extras_keep_boto3_in_test_tooling_only",
    "tests/qualification/test_live_evidence.py::test_live_qualification_requires_an_explicit_region_without_owner_pinning",
    "tests/test_phase071_contract_verifier.py::test_fixed_manifest_covers_all_phase_plans_decisions_and_non_claims",
)
PHASE071_ALL_NODES = tuple(dict.fromkeys((*PHASE071_TEST_NODES, *_ALL_ONLY_NODES)))

PHASE8_NON_CLAIMS = (
    "real AWS",
    "live PostgreSQL",
    "native platforms",
    "full independent advertised optional-group packaging matrix",
    "RSS/performance budgets",
    "SHA-256-versus-XXH3 benchmarks",
)


def _top_level_test_names(path: Path) -> set[str]:
    """Return test functions from a parsed test module without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def validate_selector(selector: str, root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Reject selectors that are malformed, unowned, or no longer defined."""
    if selector.count("::") != 1:
        return (f"selector must contain one path::test separator: {selector}",)
    relative_path, test_name = selector.split("::", 1)
    path = Path(relative_path)
    if (
        not relative_path.startswith("tests/")
        or not test_name.startswith("test_")
        or not test_name.isidentifier()
        or path.is_absolute()
        or ".." in path.parts
    ):
        return (f"selector is not an owned tests/path::test_name node: {selector}",)
    source_path = root / path
    if not source_path.is_file():
        return (f"selector source is absent: {selector}",)
    try:
        names = _top_level_test_names(source_path)
    except (OSError, SyntaxError) as error:
        return (f"selector source cannot be parsed: {selector}: {error}",)
    if test_name not in names:
        return (f"selector test is absent or renamed: {selector}",)
    return ()


def validate_mapping_inventory(
    *,
    decisions: Mapping[str, Sequence[str]],
    threats: Mapping[str, Sequence[str]],
    plans: Sequence[str],
) -> tuple[str, ...]:
    """Check closed mapping shapes before running the selected evidence."""
    errors: list[str] = []
    if tuple(plans) != _REVIEWED_PLAN_PATHS:
        errors.append("plan inventory is not the reviewed literal Plan 01-11 tuple")
    expected_decisions = {f"D-{number:02d}" for number in range(1, 17)}
    if set(decisions) != expected_decisions:
        errors.append("decision inventory is not the locked D-01 through D-16 set")
    if set(threats) != set(_REVIEWED_THREAT_NODES):
        errors.append("threat inventory does not match the reviewed threat registry")

    for kind, mapping in (("decision", decisions), ("threat", threats)):
        for identifier, selectors in mapping.items():
            if not selectors:
                errors.append(f"{kind} {identifier} has no owned selector")
            duplicates = [
                selector
                for selector, count in Counter(selectors).items()
                if count > 1
            ]
            if duplicates:
                errors.append(
                    f"{kind} {identifier} repeats selector(s): {', '.join(duplicates)}"
                )
    return tuple(errors)


def _planned_requirements(plan_text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"^  - ([A-Z]+-\d+)$", plan_text, flags=re.MULTILINE))


def _planned_threats(plan_text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"T-07\.1-[A-Za-z0-9-]+", plan_text))))


def validate_fixed_manifest(root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Validate plans, requirements, threats, decisions, and selectors fail closed."""
    errors = list(
        validate_mapping_inventory(
            decisions=DECISION_NODES,
            threats=THREAT_NODES,
            plans=PHASE071_PLAN_PATHS,
        )
    )
    if PHASE071_REQUIREMENTS != _REVIEWED_REQUIREMENTS:
        errors.append("requirement inventory is not the reviewed literal phase set")
    if THREAT_NODES != _REVIEWED_THREAT_NODES:
        errors.append("threat nodes were changed from the reviewed literal registry")

    observed_requirements: set[str] = set()
    for plan_path in _REVIEWED_PLAN_PATHS:
        source_path = root / plan_path
        if not source_path.is_file():
            errors.append(f"reviewed plan is absent: {plan_path}")
            continue
        plan_text = source_path.read_text(encoding="utf-8")
        observed_requirements.update(_planned_requirements(plan_text))
        plan_number = plan_path.rsplit("-", 2)[-2]
        observed_threats = _planned_threats(plan_text)
        if observed_threats != _REVIEWED_PLAN_THREATS[plan_number]:
            errors.append(f"threat source inventory drifted for Plan {plan_number}")
    if tuple(sorted(observed_requirements)) != tuple(sorted(_REVIEWED_REQUIREMENTS)):
        errors.append("requirement source inventory drifted from the reviewed phase set")

    context_path = root / PHASE071_CONTEXT_PATH
    if not context_path.is_file():
        errors.append("Phase 07.1 context is absent")
    else:
        context_text = context_path.read_text(encoding="utf-8")
        missing_decisions = [
            decision for decision in DECISION_NODES if f"**{decision}:" not in context_text
        ]
        if missing_decisions:
            errors.append(
                "context lacks locked decisions: " + ", ".join(missing_decisions)
            )

    owned_nodes = set(PHASE071_ALL_NODES)
    for selector in owned_nodes:
        errors.extend(validate_selector(selector, root))
    for mapping in (DECISION_NODES, THREAT_NODES, COVERAGE_NODES):
        for selectors in mapping.values():
            for selector in selectors:
                if selector not in owned_nodes:
                    errors.append(f"mapping selector is not in the execution inventory: {selector}")
    return tuple(dict.fromkeys(errors))


class _SourceAudit(ast.NodeVisitor):
    """AST rules for structural cutover regressions, independent of comments."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        if self.filename.startswith("src/") and any(
            alias.name == "boto3" or alias.name.startswith("boto3.")
            for alias in node.names
        ):
            self.errors.append("production boto3 import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.filename.startswith("src/") and (node.module or "").startswith("boto3"):
            self.errors.append("production boto3 import")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(
            isinstance(target, ast.Name)
            and target.id in {"payloads", "payload_backends", "blob_backends"}
            for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            self.errors.append("custom payload dictionary")
        if (
            self.filename.endswith("obstore_generation_io.py")
            and any(
                isinstance(target, ast.Name) and target.id == "use_multipart"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ):
            self.errors.append("multipart payload lifecycle")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            self.filename.endswith("composition.py")
            and "legacy" in node.id.lower()
            and "backend" in node.id.lower()
        ):
            self.errors.append("runtime payload selector or fallback")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            self.filename.endswith("lifecycle.py")
            and node.name.endswith("LifecycleAuthority")
            and node.name != "LifecycleAuthority"
        ):
            self.errors.append("duplicate lifecycle authority")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.filename.endswith("handlers.py") and node.name in {"put", "get"}:
            arguments = {argument.arg for argument in node.args.args}
            if "stream" in arguments or "locator" in arguments:
                self.errors.append("handler signature drift")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return source-contract errors without importing production modules."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return (f"unparseable production source: {filename}: {error}",)
    visitor = _SourceAudit(filename)
    visitor.visit(tree)
    return tuple(dict.fromkeys(visitor.errors))


def audit_runtime_source(root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Apply source gates to the cutover-owned runtime modules."""
    source_paths = (
        "src/cacheness/storage/obstore_generation_io.py",
        "src/cacheness/storage/composition.py",
        "src/cacheness/storage/lifecycle.py",
        "src/cacheness/handlers.py",
        "src/cacheness/interfaces.py",
    )
    errors: list[str] = []
    for relative_path in source_paths:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"source audit target is absent: {relative_path}")
            continue
        errors.extend(audit_source(path.read_text(encoding="utf-8"), relative_path))

    # D-06 is package-wide: a boto3 import in an unlisted runtime module is
    # still a production escape hatch even if no cutover-owned file imports it.
    for path in (root / "src" / "cacheness").rglob("*.py"):
        relative_path = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        except (OSError, SyntaxError) as error:
            errors.append(f"production source cannot be parsed: {relative_path}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "boto3" or alias.name.startswith("boto3.")
                for alias in node.names
            ):
                errors.append(f"production boto3 import: {relative_path}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("boto3"):
                errors.append(f"production boto3 import: {relative_path}")

    pyproject_path = root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        errors.append(f"pyproject cannot be audited: {error}")
        return tuple(dict.fromkeys(errors))
    project = pyproject.get("project", {})
    runtime_dependencies = list(project.get("dependencies", ()))
    runtime_dependencies.extend(
        dependency
        for dependencies in project.get("optional-dependencies", {}).values()
        for dependency in dependencies
    )
    if any("boto3" in dependency.lower() for dependency in runtime_dependencies):
        errors.append("boto3 remains in a production dependency boundary")
    return tuple(dict.fromkeys(errors))


def audit_documentation(root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Require the public D-16 and Phase 8 boundary to be stated consistently."""
    requirements = {
        "README.md": (
            "128 MiB",
            "explicit bucket and region",
            "SHA-256",
            "opaque transport evidence",
            "Phase 8",
            "store.handlers.register_handler",
        ),
        "docs/API_REFERENCE.md": (
            "128 MiB",
            "explicit bucket and region",
            "owner pinning",
            "SHA-256",
            "Phase 8",
            "store.handlers.register_handler",
        ),
        "docs/PLUGIN_DEVELOPMENT.md": (
            "128 MiB",
            "explicit bucket and region",
            "custom endpoint",
            "SHA-256",
            "Phase 8",
            "store.handlers.register_handler",
        ),
        "docs/SECURITY.md": (
            "128 MiB",
            "explicit bucket and region",
            "ExpectedBucketOwner",
            "stable bucket",
            "name whose ownership",
            "IAM",
            "bucket policy",
            "custom endpoint",
            "SHA-256",
            "Phase 8",
        ),
    }
    forbidden = ("S3BlobBackend", "BlobBackend", "register_blob_backend")
    errors: list[str] = []
    for relative_path, terms in requirements.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"documentation target is absent: {relative_path}")
            continue
        text = " ".join(path.read_text(encoding="utf-8").split())
        for term in terms:
            if term not in text:
                errors.append(f"{relative_path} lacks required contract term: {term}")
        for term in forbidden:
            if term in text:
                errors.append(f"{relative_path} revives removed payload API: {term}")
    return tuple(dict.fromkeys(errors))


def _run_pytest(nodes: Iterable[str], *, postgresql_extra: bool = False) -> int:
    command = [sys.executable, "-m", "pytest", "-q", "-o", "log_cli=false", *nodes, "-x"]
    if postgresql_extra:
        command = [
            "uv",
            "run",
            "--extra",
            "postgresql",
            "python",
            "-m",
            "pytest",
            "-q",
            "-o",
            "log_cli=false",
            *nodes,
            "-x",
        ]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


def _run_ruff() -> int:
    targets = (
        "src/cacheness/storage/obstore_generation_io.py",
        "src/cacheness/storage/composition.py",
        "src/cacheness/storage/lifecycle.py",
        "src/cacheness/handlers.py",
        "src/cacheness/interfaces.py",
        "tests/test_phase071_contract_verifier.py",
        "tests/test_public_api_contract.py",
        "tests/test_full_suite_environment.py",
        "tests/qualification/conftest.py",
        "tests/qualification/test_live_evidence.py",
        "tools/verify_phase071_contracts.py",
    )
    command = [sys.executable, "-m", "ruff", "check", *targets]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


def _report_phase8_non_claims() -> None:
    print("PHASE 8 NOT RUN / NOT QUALIFIED:")
    for non_claim in PHASE8_NON_CLAIMS:
        print(f"- {non_claim}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run static proof plus quick or full deterministic selected evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="run the sub-30-second core")
    mode.add_argument("--all", action="store_true", help="run all deterministic phase evidence")
    arguments = parser.parse_args(argv)

    static_errors = (
        *validate_fixed_manifest(REPOSITORY_ROOT),
        *audit_runtime_source(REPOSITORY_ROOT),
        *audit_documentation(REPOSITORY_ROOT),
    )
    if static_errors:
        print("Phase 07.1 contract audit failed:", file=sys.stderr)
        for error in static_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    started = time.monotonic()
    nodes = (
        PHASE071_QUICK_NODES
        if arguments.quick
        else tuple(node for node in PHASE071_ALL_NODES if node not in POSTGRESQL_NODES)
    )
    if _run_pytest(nodes):
        return 1
    if arguments.all and _run_pytest(POSTGRESQL_NODES, postgresql_extra=True):
        return 1
    if arguments.all and _run_ruff():
        return 1
    elapsed = time.monotonic() - started
    if arguments.quick and elapsed >= 30:
        print(f"quick contract exceeded 30 seconds ({elapsed:.2f}s)", file=sys.stderr)
        return 1
    print(f"Phase 07.1 {'quick' if arguments.quick else 'all'} contract passed in {elapsed:.2f}s")
    _report_phase8_non_claims()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
