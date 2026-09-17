"""Static contracts for the Phase 9 canonical example CI gate."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "quality.yml"
EXAMPLES_INDEX_PATH = ROOT / "examples" / "README.md"

CANONICAL_EXAMPLES = (
    "memory_blob_store.py",
    "durable_catalog_store.py",
    "unified_cache.py",
    "custom_mcap_format.py",
)
REMOVED_EXAMPLES = (
    "api_request_caching.py",
    "checkpoint_storage.py",
    "configurable_serialization_demo.py",
    "custom_metadata_demo.py",
    "dill_class_caching_demo.py",
    "ml_model_versioning.py",
    "pipeline_artifact_storage.py",
    "s3_caching.py",
    "simple_api_caching.py",
    "simple_config_demo.py",
    "simple_function_caching.py",
    "simple_ml_pipeline.py",
    "simple_object_caching.py",
)


def _linux_stable_job(workflow: str) -> str:
    """Return the ordinary Linux matrix job, excluding later workflow paths."""

    match = re.search(
        r"^  linux-stable:(.*?)(?=^  tensorflow-compatible:)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_linux_stable_ci_runs_only_the_exact_phase9_example_harness() -> None:
    """The public example gate cannot be omitted, broadened, or made advisory."""

    linux_stable = _linux_stable_job(WORKFLOW_PATH.read_text(encoding="utf-8"))

    expected_step = """      - name: Run exact Phase 9 published examples
        if: matrix.python == '3.13'
        run: |
          uv run --isolated --all-extras --group dev --frozen pytest \\
            -q -o log_cli=false tests/test_phase9_examples.py
"""
    assert expected_step in linux_stable
    assert linux_stable.count("tests/test_phase9_examples.py") == 1
    assert "continue-on-error:" not in linux_stable
    assert "secrets." not in linux_stable.casefold()
    assert "live_qualification" not in linux_stable
    assert "run_phase8_qualification.py" not in linux_stable


def test_examples_index_links_only_the_canonical_local_journeys() -> None:
    """The supported index and CI allowlist describe the same four files."""

    index = EXAMPLES_INDEX_PATH.read_text(encoding="utf-8")
    linked_python_files = tuple(
        re.findall(r"\[[^\]]+\]\(([^)]+\.py)\)", index)
    )

    assert linked_python_files == CANONICAL_EXAMPLES
    assert "sqlcache" not in index.casefold()
    for path in REMOVED_EXAMPLES:
        assert path not in index
