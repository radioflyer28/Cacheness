"""Executable contract tests for the public serializer trust boundary."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SECURITY_GUIDE = PROJECT_ROOT / "docs" / "SECURITY.md"
README = PROJECT_ROOT / "README.md"


def _section(document: str, heading: str) -> str:
    """Return one second-level Markdown section without snapshotting its prose."""
    start = document.index(heading)
    next_heading = document.find("\n## ", start + len(heading))
    if next_heading == -1:
        return document[start:]
    return document[start:next_heading]


def test_security_guide_defines_the_trusted_payload_boundary() -> None:
    """Pickle/dill risk and integrity limits must be explicit operator guidance."""
    guide = SECURITY_GUIDE.read_text(encoding="utf-8")
    boundary = _section(guide, "## Trusted Payload and Executable Serializer Boundary")
    normalized = boundary.lower()

    assert "pickle" in normalized
    assert "dill" in normalized
    assert "arbitrary code" in normalized
    assert "trusted application payload" in normalized
    assert "hmac" in normalized
    assert "integrity" in normalized
    assert "does not" in normalized
    assert "sandbox" in normalized


def test_security_guide_lists_safe_array_default_and_every_object_array_predicate() -> None:
    """Object arrays require the full explicit trusted-object configuration."""
    guide = SECURITY_GUIDE.read_text(encoding="utf-8")
    boundary = _section(guide, "## Trusted Payload and Executable Serializer Boundary")

    assert "native NPZ" in boundary
    assert "allow_pickle=False" in boundary
    assert "allow_trusted_object_arrays=True" in boundary
    assert "enable_object_pickle=True" in boundary
    assert "enable_entry_signing=True" in boundary
    assert "verify_cache_integrity=True" in boundary
    assert "allow_unsigned_entries=False" in boundary
    assert "malformed metadata" in boundary.lower()
    assert "legacy header" in boundary.lower()


def test_security_guide_preserves_native_format_ownership_without_future_promises() -> None:
    """The custom raw header is legacy read-only, not a future format promise."""
    guide = SECURITY_GUIDE.read_text(encoding="utf-8")
    boundary = _section(guide, "## Trusted Payload and Executable Serializer Boundary")
    normalized = boundary.lower()

    assert "read-only compatibility" in normalized
    assert "native handler" in normalized
    assert "format owner" in normalized
    assert "canonical manifest" not in normalized
    assert "migration runner" not in normalized
    assert "phase 2" not in normalized


def test_security_guide_preserves_fail_closed_parsing_and_contained_staging() -> None:
    """Trust guidance keeps the payload boundary separate from release qualification."""
    guide = SECURITY_GUIDE.read_text(encoding="utf-8")
    boundary = _section(guide, "## Trusted Payload and Executable Serializer Boundary")
    normalized = boundary.lower()

    for phrase in (
        "safe parsing",
        "path containment",
        "fail closed",
        "private staging",
        "regular artifact",
        "sha-256 plus size",
        "opaque corroborating transport evidence",
    ):
        assert phrase in normalized

    assert "## Evidence matrix" not in guide
    assert "[Release qualification](RELEASE_QUALIFICATION.md)" in guide


def test_readme_points_to_the_canonical_security_boundary_without_serializer_claims() -> None:
    """README readers receive a truthful handoff instead of a second contract."""
    readme = README.read_text(encoding="utf-8")
    security_section = _section(readme, "## Security and Integrity")

    assert "[Security Guide](docs/SECURITY.md)" in readme
    assert "trusted application payload" in security_section.lower()
    assert "pickle" in security_section.lower()
    assert "dill" in security_section.lower()
    assert "signature" in security_section.lower()
    assert "safe to deserialize hostile" not in security_section.lower()
