"""Security contract for legacy arrays and trusted object-array routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cacheness.error_handling import CacheLegacyFormatError, CacheReason
from cacheness.handlers import ArrayHandler, _parse_legacy_array_shape


def _legacy_payload(
    path: Path,
    array: np.ndarray,
    *,
    shape_bytes: bytes | None = None,
    dtype_bytes: bytes | None = None,
    compressed: bytes | None = None,
) -> None:
    """Write the historical length-framed Blosc2 array layout directly."""
    import blosc2

    raw_shape = shape_bytes if shape_bytes is not None else str(array.shape).encode("ascii")
    raw_dtype = dtype_bytes if dtype_bytes is not None else str(array.dtype).encode("ascii")
    raw_payload = (
        compressed
        if compressed is not None
        else blosc2.compress2(array.tobytes(), cparams={"typesize": array.dtype.itemsize})
    )
    path.write_bytes(
        len(raw_shape).to_bytes(4, "little")
        + raw_shape
        + len(raw_dtype).to_bytes(4, "little")
        + raw_dtype
        + raw_payload
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"()", ()),
        (b"(3,)", (3,)),
        (b"(2, 3)", (2, 3)),
        (b" ( 2 , 3 , ) \t", (2, 3)),
    ],
)
def test_parse_legacy_array_shape_accepts_only_bounded_tuple_grammar(
    raw: bytes, expected: tuple[int, ...]
) -> None:
    """Supported historical tuple shapes are decoded without evaluation."""
    assert _parse_legacy_array_shape(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"[2, 3]",
        b"(2, __import__('os'))",
        b"(2, 3) trailing",
        b"(-1,)",
        b"(1,,2)",
        b"(1" + b",1" * 32 + b")",
        b"(999999999999999999999999999999999999999999,)",
    ],
)
def test_parse_legacy_array_shape_rejects_unsafe_or_invalid_grammar(raw: bytes) -> None:
    """The legacy parser never accepts executable or resource-abusive metadata."""
    with pytest.raises(CacheLegacyFormatError) as error:
        _parse_legacy_array_shape(raw)

    assert error.value.context["reason"] == CacheReason.INVALID_LEGACY_ARRAY.value


@pytest.mark.parametrize(
    "array",
    [
        np.array(7, dtype=np.int32),
        np.array([1, 2, 3], dtype=np.int32),
        np.arange(6, dtype=np.int32).reshape(2, 3),
    ],
)
def test_legacy_reader_reconstructs_checked_scalar_vector_and_matrix(
    tmp_path: Path, array: np.ndarray
) -> None:
    """Legacy frames reconstruct only when framing and payload agree exactly."""
    path = tmp_path / "payload.b2nd"
    _legacy_payload(path, array)

    actual = ArrayHandler()._read_blosc2_array(path)

    np.testing.assert_array_equal(actual, array)
    assert actual.dtype == array.dtype


def test_legacy_reader_rejects_byte_mismatch_before_array_reconstruction(
    tmp_path: Path
) -> None:
    """A shape whose checked size disagrees with decompressed bytes is rejected."""
    path = tmp_path / "payload.b2nd"
    _legacy_payload(path, np.arange(6, dtype=np.int32), shape_bytes=b"(2, 4)")

    with pytest.raises(CacheLegacyFormatError) as error:
        ArrayHandler()._read_blosc2_array(path)

    assert error.value.context["reason"] == CacheReason.INVALID_LEGACY_ARRAY.value


def test_legacy_reader_rejects_object_dtype_before_deserialization(tmp_path: Path) -> None:
    """Historical frames with object dtype cannot cross the pickle boundary."""
    path = tmp_path / "payload.b2nd"
    _legacy_payload(path, np.arange(1, dtype=np.int64), dtype_bytes=b"object")

    with pytest.raises(CacheLegacyFormatError) as error:
        ArrayHandler()._read_blosc2_array(path)

    assert error.value.context["reason"] == CacheReason.UNSAFE_OBJECT_ARRAY.value


def test_declared_blosc2_failure_never_probes_valid_npz_sidecar(tmp_path: Path) -> None:
    """A persisted storage format selects exactly one reader."""
    path = tmp_path / "payload.b2nd"
    _legacy_payload(path, np.arange(6, dtype=np.int32), shape_bytes=b"not-a-tuple")
    np.savez_compressed(tmp_path / "payload.npz", value=np.arange(6, dtype=np.int32))

    with pytest.raises(CacheLegacyFormatError):
        ArrayHandler().get(path, {"storage_format": "blosc2"})
