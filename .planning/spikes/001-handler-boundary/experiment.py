"""Prove path-based built-in and custom handlers across obstore participants."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer
from obstore.store import LocalStore, MemoryStore

from cacheness.config import CacheConfig
from cacheness.handlers import HandlerRegistry
from cacheness.interfaces import CacheHandler
from cacheness.storage.guarded_handler_io import GuardedHandlerIO


@dataclass(frozen=True)
class McapRecording:
    """Small user-domain value intentionally unknown to Cacheness."""

    topic: str
    messages: tuple[dict[str, Any], ...]


class McapHandler(CacheHandler):
    """Realistic third-party handler implemented only with paths and MCAP."""

    @property
    def data_type(self) -> str:
        return "mcap_recording"

    @property
    def payload_format(self) -> str:
        return "mcap"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, McapRecording)

    def get_file_extension(self, config: Any) -> str:
        return ".mcap"

    def put(self, data: McapRecording, file_path: Path, config: Any) -> dict[str, Any]:
        actual_path = file_path.with_suffix(".mcap")
        schema_bytes = json.dumps(
            {"type": "object", "additionalProperties": True}, sort_keys=True
        ).encode()
        with actual_path.open("wb") as output:
            writer = Writer(output)
            writer.start()
            schema_id = writer.register_schema(
                name="cacheness-spike", encoding="jsonschema", data=schema_bytes
            )
            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic=data.topic,
                message_encoding="json",
            )
            for index, message in enumerate(data.messages, start=1):
                writer.add_message(
                    channel_id=channel_id,
                    log_time=index,
                    publish_time=index,
                    data=json.dumps(message, sort_keys=True).encode(),
                )
            writer.finish()
        return {
            "actual_path": str(actual_path),
            "file_size": actual_path.stat().st_size,
            "storage_format": "mcap",
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "metadata": {"topic": data.topic, "messages": len(data.messages)},
        }

    def get(self, file_path: Path, metadata: dict[str, Any]) -> McapRecording:
        messages: list[dict[str, Any]] = []
        topic = ""
        with file_path.open("rb") as source:
            for _schema, channel, message in make_reader(source).iter_messages():
                topic = channel.topic
                messages.append(json.loads(message.data))
        return McapRecording(topic, tuple(messages))


def safe_locator(locator: str) -> str:
    """Apply the minimal namespace contract expected above obstore."""
    if not locator or "\\" in locator or locator.startswith("/"):
        raise ValueError("unsafe locator")
    parts = locator.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe locator")
    normalized = PurePosixPath(locator)
    if normalized.parts != tuple(parts):
        raise ValueError("unsafe locator")
    return normalized.as_posix()


class ObstoreHandlerBridge:
    """Spike-only bridge: private path staging around an obstore participant."""

    def __init__(self, store: Any, private_root: Path):
        private_root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.staging = GuardedHandlerIO(private_root)

    def close(self) -> None:
        self.staging.close()

    def put(
        self,
        handler: CacheHandler,
        data: Any,
        config: CacheConfig,
        locator_base: str,
    ) -> tuple[str, dict[str, Any]]:
        with self.staging.stage(handler, data, config) as staged:
            locator = safe_locator(f"{locator_base}{staged.suffix}")
            with staged.open() as (source, file_size):
                self.store.put(locator, source, mode="create")
            result = staged.result_for(Path(locator), file_size)
            return locator, result

    @contextmanager
    def snapshot(self, locator: str, suffix: str) -> Iterator[Path]:
        safe_locator(locator)
        with tempfile.TemporaryDirectory(prefix="cacheness-obstore-read-") as root:
            snapshot = Path(root) / f"payload{suffix}"
            with snapshot.open("xb") as output:
                for chunk in self.store.get(locator):
                    output.write(chunk)
            yield snapshot

    def get(
        self,
        handler: CacheHandler,
        locator: str,
        suffix: str,
        metadata: dict[str, Any],
    ) -> Any:
        with self.snapshot(locator, suffix) as snapshot:
            return handler.get(snapshot, metadata)


def exercise(store_name: str, store: Any, private_root: Path) -> dict[str, Any]:
    config = CacheConfig()
    registry = HandlerRegistry(config)
    custom = McapHandler()
    registry.register_handler(custom, priority=0)
    bridge = ObstoreHandlerBridge(store, private_root)
    try:
        recording = McapRecording(
            "robot/pose",
            ({"x": 1.25, "y": -4}, {"x": 2.5, "y": 8}),
        )
        chosen = registry.get_handler(recording)
        assert chosen is custom
        mcap_locator, mcap_meta = bridge.put(
            chosen, recording, config, "generations/custom-deadbeef"
        )
        restored_recording = bridge.get(chosen, mcap_locator, ".mcap", mcap_meta)
        assert restored_recording == recording
        assert mcap_meta["metadata"] == {"topic": "robot/pose", "messages": 2}

        array = np.arange(12, dtype=np.int32).reshape(3, 4)
        array_handler = registry.get_handler(array)
        array_locator, array_meta = bridge.put(
            array_handler, array, config, "generations/array-cafebabe"
        )
        restored_array = bridge.get(
            array_handler, array_locator, ".npz", array_meta
        )
        np.testing.assert_array_equal(restored_array, array)

        return {
            "store": store_name,
            "custom_handler": chosen.data_type,
            "custom_locator": mcap_locator,
            "custom_size": mcap_meta["file_size"],
            "built_in_handler": array_handler.data_type,
            "built_in_locator": array_locator,
            "built_in_size": array_meta["file_size"],
            "round_trips": 2,
        }
    finally:
        bridge.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="obstore-handler-spike-") as root_text:
        root = Path(root_text)
        results = [
            exercise("memory", MemoryStore(), root / "memory-private"),
            exercise(
                "local",
                LocalStore(root / "objects", mkdir=True),
                root / "local-private",
            ),
        ]
    print(json.dumps({"verdict": "VALIDATED", "results": results}, indent=2))


if __name__ == "__main__":
    main()

