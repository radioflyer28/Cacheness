"""Adversarial source fixtures for the Phase 4 executable-consumer audit."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_VERIFIER_PATH = Path(__file__).parents[1] / "tools" / "verify_phase4_cutover.py"
_SPEC = spec_from_file_location("phase4_cutover_verifier", _VERIFIER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VERIFIER = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VERIFIER)
audit_source = _VERIFIER.audit_source


def test_verifier_keeps_only_the_retained_owned_matrix_live() -> None:
    """Historical deferred SqlCache paths are not executable verifier inputs."""
    assert _VERIFIER.PHASE4_MATRIX == _VERIFIER.load_owned_matrix()
    assert all("sql_cache" not in path for path in _VERIFIER.PHASE4_MATRIX)
    assert not hasattr(_VERIFIER, "DEFERRED_SQL_CACHE_PATHS")
    assert not hasattr(_VERIFIER, "EXPECTED_DEFERRED_PATHS")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from cacheness import create_blob_backend\n",
            ("retired import cacheness.create_blob_backend as create_blob_backend",),
        ),
        (
            "from cacheness import get_blob_backend as getter\ngetter()\n",
            (
                "retired import cacheness.get_blob_backend as getter",
                "retired bound alias use: getter",
            ),
        ),
        (
            "import cacheness as c\nc.register_blob_backend()\n",
            ("retired package attribute: cacheness.register_blob_backend",),
        ),
        (
            "import cacheness\ncacheness.unregister_blob_backend()\n",
            ("retired package attribute: cacheness.unregister_blob_backend",),
        ),
        (
            "from cacheness.storage.backends import blob_backends as b\n"
            "b.list_blob_backends()\n",
            (
                "retired package attribute: "
                "cacheness.storage.backends.blob_backends.list_blob_backends",
            ),
        ),
        (
            "import cacheness.storage.backends.blob_backends\n"
            "cacheness.storage.backends.blob_backends.create_blob_backend()\n",
            (
                "retired package attribute: "
                "cacheness.storage.backends.blob_backends.create_blob_backend",
            ),
        ),
        (
            "from cacheness.storage.backends.blob_backends import "
            "get_blob_backend as getter\ngetter()\n",
            (
                "retired import cacheness.storage.backends.blob_backends."
                "get_blob_backend as getter",
                "retired bound alias use: getter",
            ),
        ),
        (
            "from cacheness.metadata import JsonBackend as backend\nbackend()\n",
            (
                "retired import cacheness.metadata.JsonBackend as backend",
                "retired bound alias use: backend",
            ),
        ),
        (
            "from cacheness.metadata import *\n",
            ("retired star import cacheness.metadata",),
        ),
        (
            "from cacheness.storage.backends import *\n",
            ("retired star import cacheness.storage.backends",),
        ),
        (
            "from cacheness.storage.backends.blob_backends import *\n",
            ("retired star import cacheness.storage.backends.blob_backends",),
        ),
    ],
)
def test_audit_source_detects_retired_executable_consumer_forms(source, expected) -> None:
    """Every supported import spelling uses the repository audit visitor."""
    assert audit_source(source, "fixture.py") == expected


@pytest.mark.parametrize(
    "source",
    [
        "from cacheness import *\n",
        'snippet = "from cacheness import create_blob_backend"\n',
        '"""import cacheness as c; c.get_blob_backend()"""\n',
        "# from cacheness.metadata import JsonBackend\n",
    ],
)
def test_audit_source_ignores_clean_root_star_and_string_only_text(source) -> None:
    """The audit follows executable bindings rather than textual coincidence."""
    assert audit_source(source, "negative_fixture.py") == ()
