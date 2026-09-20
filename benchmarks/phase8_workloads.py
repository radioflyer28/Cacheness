"""Deterministic, layer-separated workloads for Phase 8 performance evidence.

The workload catalogue is deliberately small: one topology per reviewed tier,
not a format-by-topology matrix.  Fixture construction, store initialization,
and cleanup are outside timed callbacks so the distribution harness measures
only the named handler, storage lifecycle, or cache-policy boundary.

Legacy Blosc2 array files remain input-only compatibility evidence in the
current architecture.  New writes use the native NPZ ``ArrayHandler`` path,
so this catalogue intentionally does not resurrect a legacy Blosc2 writer or
route a benchmark around ``BlobStore``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Literal

from cacheness.config import CacheConfig, CacheStorageConfig, CompressionConfig
from cacheness.core import UnifiedCache
from cacheness.handlers import HandlerRegistry
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


LAYERS = ("handler", "blobstore", "unified-cache")
_ACCESS_LABELS = ("cold-put", "warm-get")
_MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class _TierDefinition:
    """One format and size tier before it expands into named boundaries."""

    name: str
    tier: str
    format_name: str
    canonical_size: int
    size_unit: Literal["bytes", "rows"]
    topology: Literal["memory-memory", "sqlite-filesystem"]


@dataclass(frozen=True)
class WorkloadDescriptor:
    """One reviewed measurement boundary with explicit workload metadata."""

    name: str
    tier: str
    format_name: str
    canonical_size: int
    size_unit: Literal["bytes", "rows"]
    topology: Literal["memory-memory", "sqlite-filesystem"]
    layer: Literal["handler", "blobstore", "unified-cache"]
    access: Literal["cold-put", "warm-get"]

    @property
    def identifier(self) -> str:
        """Return the stable evidence key for this exact measured boundary."""
        return "__".join((self.name, self.layer, self.access))


_TIERS = (
    _TierDefinition(
        name="small-generic-object",
        tier="small",
        format_name="generic-object",
        canonical_size=4 * 1024,
        size_unit="bytes",
        topology="memory-memory",
    ),
    _TierDefinition(
        name="numpy-npz-medium",
        tier="medium",
        format_name="npz",
        canonical_size=16 * _MEBIBYTE,
        size_unit="bytes",
        topology="memory-memory",
    ),
    _TierDefinition(
        name="numpy-npz-large",
        tier="large",
        format_name="npz",
        canonical_size=128 * _MEBIBYTE,
        size_unit="bytes",
        topology="sqlite-filesystem",
    ),
    _TierDefinition(
        name="pandas-parquet",
        tier="dataframe",
        format_name="parquet",
        canonical_size=100_000,
        size_unit="rows",
        topology="memory-memory",
    ),
    _TierDefinition(
        name="polars-parquet",
        tier="dataframe",
        format_name="parquet",
        canonical_size=100_000,
        size_unit="rows",
        topology="memory-memory",
    ),
)


def access_labels() -> tuple[str, str]:
    """Return the explicit operation labels shared by every reviewed tier."""
    return _ACCESS_LABELS


def reviewed_workloads() -> tuple[WorkloadDescriptor, ...]:
    """Expand each representative format tier across the three measured layers."""
    return tuple(
        WorkloadDescriptor(
            name=tier.name,
            tier=tier.tier,
            format_name=tier.format_name,
            canonical_size=tier.canonical_size,
            size_unit=tier.size_unit,
            topology=tier.topology,
            layer=layer,
            access=access,
        )
        for tier in _TIERS
        for layer in LAYERS
        for access in _ACCESS_LABELS
    )


def _reduced_size(descriptor: WorkloadDescriptor) -> int:
    """Return a deterministic test-scale size without changing catalogue facts."""
    if descriptor.size_unit == "rows":
        return min(descriptor.canonical_size, 128)
    return min(descriptor.canonical_size, 64 * 1024)


def _fixture_for(descriptor: WorkloadDescriptor, *, reduced: bool) -> Any:
    """Build one deterministic fixture before the benchmark timer starts."""
    requested_size = _reduced_size(descriptor) if reduced else descriptor.canonical_size
    if descriptor.name == "small-generic-object":
        payload_length = max(0, requested_size - 256)
        return {
            "kind": "phase8-generic-object",
            "nested": {
                "labels": ["cache", "blob", "integrity"],
                "sequence": list(range(16)),
                "payload": "c" * payload_length,
            },
        }
    if descriptor.name.startswith("numpy-npz-"):
        import numpy as np

        return np.arange(requested_size, dtype=np.uint8).reshape((-1, 1))
    if descriptor.name == "pandas-parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("pandas-parquet workload requires the dataframe extra") from error
        return pd.DataFrame(
            {
                "sequence": range(requested_size),
                "label": [f"tier-{index % 13}" for index in range(requested_size)],
                "occurred_at": pd.date_range(
                    "2024-01-01", periods=requested_size, freq="min"
                ),
            }
        )
    if descriptor.name == "polars-parquet":
        try:
            import polars as pl
        except ImportError as error:
            raise RuntimeError("polars-parquet workload requires the dataframe extra") from error
        return pl.DataFrame(
            {
                "sequence": range(requested_size),
                "label": [f"tier-{index % 13}" for index in range(requested_size)],
                "occurred_at": [
                    f"2024-01-{(index % 28) + 1:02d}T00:00:00"
                    for index in range(requested_size)
                ],
            }
        ).with_columns(pl.col("occurred_at").str.to_datetime())
    raise ValueError(f"No Phase 8 fixture factory exists for {descriptor.name!r}")


def _fixture_bytes(descriptor: WorkloadDescriptor, fixture: Any) -> bytes:
    """Create a deterministic fixture fingerprint input outside timed work."""
    if descriptor.name == "small-generic-object":
        return json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if descriptor.name.startswith("numpy-npz-"):
        return fixture.tobytes(order="C")
    if descriptor.name == "pandas-parquet":
        return fixture.to_csv(index=False).encode("utf-8")
    if descriptor.name == "polars-parquet":
        return fixture.write_csv().encode("utf-8")
    raise ValueError(f"No fixture byte encoding exists for {descriptor.name!r}")


def _topology(descriptor: WorkloadDescriptor, root: Path) -> StoreTopology:
    """Build the one reviewed public topology selected for a workload tier."""
    if descriptor.topology == "memory-memory":
        return StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        )
    if descriptor.topology == "sqlite-filesystem":
        return StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )
    raise ValueError(f"No reviewed topology exists for {descriptor.topology!r}")


def _workload_config(root: Path) -> CacheConfig:
    """Create a stable handler/cache config without legacy Blosc2 write hints."""
    return CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(root)),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )


class LayerWorkload:
    """Own one fixture and public-composition boundary for repeated sampling.

    ``prepare_for_sample`` and ``cleanup_after_sample`` are intentionally outside
    the timer.  The benchmark runner invokes only ``timed_callback`` between
    ``perf_counter_ns`` calls.
    """

    uses_public_composition = True

    def __init__(self, descriptor: WorkloadDescriptor, *, reduced: bool) -> None:
        self.descriptor = descriptor
        self.fixture = _fixture_for(descriptor, reduced=reduced)
        self.fixture_bytes = _fixture_bytes(descriptor, self.fixture)
        self.fixture_digest = hashlib.sha256(self.fixture_bytes).hexdigest()
        self._temporary = tempfile.TemporaryDirectory(prefix="cacheness-phase8-workload-")
        self._root = Path(self._temporary.name)
        self._config = _workload_config(self._root / "config")
        self._sample_number = 0
        self._handler = None
        self._handler_metadata: dict[str, Any] | None = None
        self._handler_path: Path | None = None
        self._store: BlobStore | None = None
        self._cache: UnifiedCache | None = None
        self._key: str | None = None
        self._prepare_boundary()
        self.prepare_for_sample()

    @property
    def layer(self) -> str:
        """Expose the named boundary without inferring it from a callback."""
        return self.descriptor.layer

    def _prepare_boundary(self) -> None:
        if self.descriptor.layer == "handler":
            self._handler = HandlerRegistry(self._config).get_handler(self.fixture)
            return
        topology = _topology(self.descriptor, self._root / "topology")
        if self.descriptor.layer == "blobstore":
            self._store = BlobStore(topology, cache_dir=self._root / "blobstore")
            self._store.initialize()
            return
        if self.descriptor.layer == "unified-cache":
            self._cache = UnifiedCache(self._config, store=topology)
            self._cache.initialize()
            return
        raise ValueError(f"Unsupported benchmark layer {self.descriptor.layer!r}")

    def prepare_for_sample(self) -> None:
        """Stage fixture/state before one callback without contaminating timing."""
        self._sample_number += 1
        self._key = f"phase8-{self.descriptor.identifier}-{self._sample_number:06d}"
        if self.descriptor.layer == "handler":
            assert self._handler is not None
            self._handler_path = self._root / f"handler-{self._sample_number:06d}.payload"
            self._handler_metadata = None
            if self.descriptor.access == "warm-get":
                self._handler_metadata = self._handler.put(
                    self.fixture, self._handler_path, self._config
                )
            return
        if self.descriptor.access != "warm-get":
            return
        if self._store is not None:
            self._store.put(self.fixture, key=self._key)
            return
        assert self._cache is not None
        result = self._cache.put(self.fixture, request_id=self._key)
        self._key = result.receipt.key

    def timed_callback(self) -> Any:
        """Perform only the configured handler, storage, or policy operation."""
        assert self._key is not None
        if self.descriptor.layer == "handler":
            assert self._handler is not None and self._handler_path is not None
            if self.descriptor.access == "cold-put":
                return self._handler.put(self.fixture, self._handler_path, self._config)
            assert self._handler_metadata is not None
            actual_path = Path(self._handler_metadata["actual_path"])
            return self._handler.get(actual_path, self._handler_metadata)
        if self._store is not None:
            if self.descriptor.access == "cold-put":
                return self._store.put(self.fixture, key=self._key)
            return self._store.get(self._key)
        assert self._cache is not None
        if self.descriptor.access == "cold-put":
            return self._cache.put(self.fixture, request_id=self._key)
        return self._cache.lookup(cache_key=self._key)

    def cleanup_after_sample(self) -> None:
        """Discard only private handler staging after a sample has completed."""
        if self.descriptor.layer != "handler" or self._handler_path is None:
            return
        for candidate in self._root.glob(f"{self._handler_path.stem}*"):
            if candidate.is_file():
                candidate.unlink()

    def close(self) -> None:
        """Close caller-owned benchmark resources and remove private staging."""
        try:
            if self._cache is not None:
                self._cache.close()
            elif self._store is not None:
                self._store.close()
        finally:
            self._temporary.cleanup()


def build_layer_workload(
    descriptor: WorkloadDescriptor, *, reduced: bool = False
) -> LayerWorkload:
    """Materialize one named workload with all setup complete before timing."""
    if descriptor not in reviewed_workloads():
        raise ValueError("Benchmark workload descriptor is not part of the reviewed inventory")
    return LayerWorkload(descriptor, reduced=reduced)


def with_workload(
    descriptor: WorkloadDescriptor,
    *,
    reduced: bool,
    callback: Callable[[LayerWorkload], Any],
) -> Any:
    """Run a caller callback while guaranteeing private benchmark cleanup."""
    workload = build_layer_workload(descriptor, reduced=reduced)
    try:
        return callback(workload)
    finally:
        workload.close()


__all__ = [
    "LAYERS",
    "LayerWorkload",
    "WorkloadDescriptor",
    "access_labels",
    "build_layer_workload",
    "reviewed_workloads",
    "with_workload",
]
