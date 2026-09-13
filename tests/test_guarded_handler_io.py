"""Focused security contracts for private path-based handler I/O."""

from __future__ import annotations

from pathlib import Path

from cacheness.storage.guarded_handler_io import GuardedHandlerIO


class _NativePathHandler:
    """Minimal handler that records only its private stage path."""

    def __init__(self) -> None:
        self.received_path: Path | None = None

    def put(self, data: bytes, file_path: Path, config: object) -> dict[str, object]:
        del config
        self.received_path = file_path
        artifact = file_path.with_suffix(".native")
        artifact.write_bytes(data)
        return {"actual_path": str(artifact), "file_size": len(data)}


def test_stage_retains_a_private_suffix_preserving_regular_file(tmp_path: Path) -> None:
    """Handlers see a private staging path and publication keeps its descriptor."""
    handler = _NativePathHandler()
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    guarded = GuardedHandlerIO(managed_root)
    try:
        with guarded.stage(handler, b"native", config=None) as staged:
            assert handler.received_path is not None
            assert not handler.received_path.is_relative_to(tmp_path / "managed")
            assert staged.suffix == ".native"
            with staged.open() as (source, byte_size):
                assert source.read() == b"native"
                assert byte_size == len(b"native")
    finally:
        guarded.close()
