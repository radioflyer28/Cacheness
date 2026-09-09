"""Current custom-handler registry contracts independent of public re-exports."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.config import CacheConfig, HandlerConfig
from cacheness.handlers import HandlerRegistry


class _MappingHandler:
    """A minimal user handler selected ahead of the built-in object fallback."""

    @property
    def data_type(self) -> str:
        return "test_mapping"

    def can_handle(self, data, config=None) -> bool:
        del config
        return isinstance(data, dict) and "custom" in data

    def put(self, data, file_path: Path, config):
        del data, config
        return {"actual_path": str(file_path), "file_size": 0}

    def get(self, file_path: Path, metadata):
        del file_path, metadata
        return {"custom": True}

    def get_file_extension(self, config) -> str:
        del config
        return ".mapping"


class _IncompleteHandler:
    """Deliberately misses the persistence interface for validation coverage."""

    @property
    def data_type(self) -> str:
        return "incomplete"


def _registry() -> HandlerRegistry:
    """Use a predictable built-in baseline for priority assertions."""

    return HandlerRegistry(
        CacheConfig(
            handlers=HandlerConfig(
                enable_pandas_dataframes=False,
                enable_polars_dataframes=False,
                enable_pandas_series=False,
                enable_polars_series=False,
            )
        )
    )


def test_custom_handler_registration_controls_selection_priority() -> None:
    """A caller can insert a valid handler before canonical built-ins."""

    registry = _registry()
    handler = _MappingHandler()
    registry.register_handler(handler, priority=0)

    assert registry.get_handler({"custom": "value"}) is handler
    assert registry.list_handlers()[0] == {
        "name": "test_mapping",
        "priority": 0,
        "class": "_MappingHandler",
        "is_builtin": False,
    }


def test_unregister_reports_truthful_current_registry_state() -> None:
    """Registration and removal never imply a second global handler authority."""

    registry = _registry()
    registry.register_handler(_MappingHandler())

    assert registry.unregister_handler("test_mapping") is True
    assert registry.unregister_handler("test_mapping") is False


def test_duplicate_and_incomplete_handlers_are_rejected() -> None:
    """Handler contract failures remain explicit at the direct registry boundary."""

    registry = _registry()
    registry.register_handler(_MappingHandler())

    with pytest.raises(ValueError, match="already registered"):
        registry.register_handler(_MappingHandler())
    with pytest.raises(ValueError, match="missing required"):
        registry.register_handler(_IncompleteHandler())
