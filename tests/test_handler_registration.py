"""Current custom-handler registry contracts independent of public re-exports."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.config import CacheConfig, HandlerConfig
from cacheness.error_handling import CacheManifestUnsupportedVersionError
from cacheness.handlers import HandlerRegistry
from cacheness.interfaces import FormatHandler, PayloadTransformationEdge
import cacheness.storage as storage
import cacheness.storage.handlers as storage_handlers


class _MappingHandler:
    """A minimal user handler selected ahead of the built-in object fallback."""

    @property
    def data_type(self) -> str:
        return "test_mapping"

    @property
    def payload_format(self) -> str:
        return "test-mapping"

    @property
    def payload_format_version(self) -> int:
        return 1

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) == (
            self.payload_format,
            self.payload_format_version,
        )

    def payload_transformation_edges(self) -> tuple[PayloadTransformationEdge, ...]:
        return ()

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


class _McapHandler(_MappingHandler):
    """Third-party handler declaring one readable native-format upgrade edge."""

    @property
    def data_type(self) -> str:
        return "mcap"

    @property
    def payload_format(self) -> str:
        return "mcap-v2"

    @property
    def payload_format_version(self) -> int:
        return 2

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) in {
            ("mcap-v1", 1),
            ("mcap-v2", 2),
        }

    def payload_transformation_edges(self) -> tuple[PayloadTransformationEdge, ...]:
        return (
            PayloadTransformationEdge("mcap-v1", 1, "mcap-v2", 2),
        )

    def transform_payload(self, snapshot, edge, *, destination_io, key: str, config):
        del snapshot, edge, destination_io, key, config
        return {"payload_format": "mcap-v2", "payload_format_version": 2}


class _DuplicateMcapHandler(_McapHandler):
    """Declares an ambiguous edge so registration must fail before migration."""

    def payload_transformation_edges(self) -> tuple[PayloadTransformationEdge, ...]:
        edge = PayloadTransformationEdge("mcap-v1", 1, "mcap-v2", 2)
        return (edge, edge)


class _DeclaredButRejectingMcapHandler(_MappingHandler, FormatHandler):
    """Declares an edge while inheriting FormatHandler's rejecting default."""

    @property
    def data_type(self) -> str:
        return "rejecting-mcap"

    @property
    def payload_format(self) -> str:
        return "mcap-v2"

    @property
    def payload_format_version(self) -> int:
        return 2

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) in {
            ("mcap-v1", 1),
            ("mcap-v2", 2),
        }

    def payload_transformation_edges(self) -> tuple[PayloadTransformationEdge, ...]:
        return (PayloadTransformationEdge("mcap-v1", 1, "mcap-v2", 2),)


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


def test_storage_handler_barrels_expose_only_format_handler_contract() -> None:
    """Storage consumers import the generic protocol with no retired alias."""

    assert storage.FormatHandler is FormatHandler
    assert storage_handlers.FormatHandler is FormatHandler
    assert storage_handlers.FormatHandlerError.__name__ == "FormatHandlerError"
    assert "FormatHandler" in storage.__all__
    assert "FormatHandlerError" in storage.__all__
    assert "FormatHandler" in storage_handlers.__all__
    assert "FormatHandlerError" in storage_handlers.__all__
    assert not hasattr(storage, "CacheHandler")
    assert not hasattr(storage_handlers, "CacheHandler")


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


def test_registered_custom_handler_resolves_one_exact_directed_transformation() -> None:
    """A store-local third-party handler owns native-format conversion eligibility."""

    registry = _registry()
    handler = _McapHandler()
    registry.register_handler(handler)

    resolved_handler, edge = registry.resolve_payload_transformation(
        "mcap", "mcap-v1", 1, "mcap-v2", 2
    )

    assert resolved_handler is handler
    assert edge == PayloadTransformationEdge("mcap-v1", 1, "mcap-v2", 2)
    with pytest.raises(CacheManifestUnsupportedVersionError, match="transformation"):
        registry.resolve_payload_transformation("mcap", "mcap-v2", 2, "mcap-v1", 1)
    with pytest.raises(CacheManifestUnsupportedVersionError, match="transformation"):
        registry.resolve_payload_transformation("mcap", "mcap-v1", 1, "other", 1)


def test_invalid_transformation_edges_are_rejected_without_global_fallback() -> None:
    """Malformed and duplicate edges cannot be registered or resolved broadly."""

    with pytest.raises(ValueError, match="wildcard"):
        PayloadTransformationEdge("mcap-*", 1, "mcap-v2", 2)
    with pytest.raises(ValueError, match="distinct"):
        PayloadTransformationEdge("mcap-v1", 1, "mcap-v1", 1)

    registry = _registry()
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_handler(_DuplicateMcapHandler())


def test_registered_handler_rejects_declared_edge_without_concrete_transform() -> None:
    """An advertised conversion cannot resolve to FormatHandler's rejection stub."""
    registry = _registry()

    with pytest.raises(ValueError, match="concrete transform"):
        registry.register_handler(_DeclaredButRejectingMcapHandler())
