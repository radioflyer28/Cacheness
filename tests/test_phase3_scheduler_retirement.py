"""Negative-reachability gates for the retired file-native scheduler."""

from importlib.util import find_spec

import pytest


RETIRED_SCHEDULER_MODULES = (
    "cacheness.storage.operation_repository",
    "cacheness.storage.operation_record",
    "cacheness.storage.clear_recovery",
)


@pytest.mark.parametrize("module_name", RETIRED_SCHEDULER_MODULES)
def test_retired_scheduler_modules_are_not_importable(module_name: str) -> None:
    """The public package cannot resolve a second lifecycle authority."""
    assert find_spec(module_name) is None
