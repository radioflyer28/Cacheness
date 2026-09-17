"""Fail-closed contracts for Phase 9 evidence-only maintenance."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE3_VERIFICATION = (
    ROOT
    / ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VERIFICATION.md"
)
PHASE8_VALIDATION = (
    ROOT
    / ".planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md"
)


def test_phase3_verification_is_historical_with_named_local_closure_evidence():
    """The stale report must not become a reason to reopen lifecycle work."""
    report = PHASE3_VERIFICATION.read_text(encoding="utf-8")

    for marker in (
        "historical",
        "03-21 through 03-25",
        "5282dca",
        "phase3-direct-implementation-2026-09-06.md",
        "07.1",
        "Phase 8",
        "SQLite/local filesystem",
        "memory authority",
        "Windows remains UNAVAILABLE/NOT_QUALIFIED",
        "not a fresh independent verifier verdict",
    ):
        assert marker in report


def test_phase8_validation_records_completed_local_evidence_without_promoting_nonclaims():
    """Local evidence cannot qualify services, platforms, performance, or release."""
    validation = PHASE8_VALIDATION.read_text(encoding="utf-8")

    for marker in (
        "status: complete",
        "Wave 0",
        "Nyquist",
        "LOCAL_READY",
        "NOT_QUALIFIED",
        "NOT_PUBLISHED",
        "BACK-05",
        "QUAL-06",
        "controlled-Linux",
        "Windows",
        "live services",
        "publication",
    ):
        assert marker in validation
