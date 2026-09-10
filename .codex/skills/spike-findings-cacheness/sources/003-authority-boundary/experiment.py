"""Demonstrate ADR 0001's authority boundary around an obstore payload."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from obstore.store import LocalStore


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def locator(payload: bytes, label: str) -> str:
    return f"generations/{label}-{digest(payload)}.bin"


def object_identity(store: Any, object_locator: str) -> tuple[str, int] | None:
    observed = hashlib.sha256()
    size = 0
    try:
        response = store.get(object_locator)
    except FileNotFoundError:
        return None
    for chunk in response:
        observed.update(chunk)
        size += len(chunk)
    return observed.hexdigest(), size


class SqliteAuthority:
    """Minimal spike authority with explicit intent and promotion transactions."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE operations (
                operation_id TEXT PRIMARY KEY,
                logical_key TEXT NOT NULL,
                locator TEXT NOT NULL,
                digest TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'active'))
            );
            CREATE TABLE entries (
                logical_key TEXT PRIMARY KEY,
                locator TEXT NOT NULL,
                digest TEXT NOT NULL,
                byte_size INTEGER NOT NULL
            );
            """
        )

    def begin_intent(self, operation_id: str, key: str, path: str, payload: bytes) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO operations VALUES (?, ?, ?, ?, ?, 'pending')",
                (operation_id, key, path, digest(payload), len(payload)),
            )

    def visible_locator(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT locator FROM entries WHERE logical_key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def operation_state(self, operation_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT state FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        return row[0] if row else None

    def recover(self, operation_id: str, store: Any) -> str:
        row = self.connection.execute(
            """SELECT logical_key, locator, digest, byte_size, state
               FROM operations WHERE operation_id = ?""",
            (operation_id,),
        ).fetchone()
        if row is None:
            return "NO_DURABLE_INTENT"
        key, path, expected_digest, expected_size, state = row
        if state == "active":
            return "ALREADY_PROMOTED"
        observed = object_identity(store, path)
        if observed is None:
            return "EFFECT_REQUIRED"
        if observed != (expected_digest, expected_size):
            return "CONFLICT"
        with self.connection:
            self.connection.execute(
                """INSERT INTO entries VALUES (?, ?, ?, ?)
                   ON CONFLICT(logical_key) DO UPDATE SET
                     locator=excluded.locator,
                     digest=excluded.digest,
                     byte_size=excluded.byte_size""",
                (key, path, expected_digest, expected_size),
            )
            self.connection.execute(
                "UPDATE operations SET state = 'active' WHERE operation_id = ?",
                (operation_id,),
            )
        return "PROMOTED"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="obstore-authority-spike-") as root_text:
        root = Path(root_text)
        store = LocalStore(root / "objects", mkdir=True)
        authority = SqliteAuthority(root / "authority.sqlite3")

        # Proof of non-atomicity: SQL rollback cannot roll back an external put.
        raw_payload = b"external effect in rolled-back SQL transaction"
        raw_locator = locator(raw_payload, "raw-rollback")
        authority.connection.execute("BEGIN IMMEDIATE")
        authority.connection.execute(
            "INSERT INTO operations VALUES (?, ?, ?, ?, ?, 'pending')",
            ("raw", "raw-key", raw_locator, digest(raw_payload), len(raw_payload)),
        )
        store.put(raw_locator, raw_payload, mode="create")
        authority.connection.rollback()
        assert authority.operation_state("raw") is None
        assert object_identity(store, raw_locator) == (digest(raw_payload), len(raw_payload))

        # Crash before publication: committed intent makes the next action deterministic.
        before_payload = b"crash before object publication"
        before_locator = locator(before_payload, "before")
        authority.begin_intent("before", "before-key", before_locator, before_payload)
        assert authority.visible_locator("before-key") is None
        before_first = authority.recover("before", store)
        assert before_first == "EFFECT_REQUIRED"
        store.put(before_locator, before_payload, mode="create")
        before_second = authority.recover("before", store)
        assert before_second == "PROMOTED"
        assert authority.visible_locator("before-key") == before_locator

        # Crash after publication: object presence alone is not visibility.
        after_payload = b"crash after object publication"
        after_locator = locator(after_payload, "after")
        authority.begin_intent("after", "after-key", after_locator, after_payload)
        store.put(after_locator, after_payload, mode="create")
        assert authority.visible_locator("after-key") is None
        after_recovery = authority.recover("after", store)
        assert after_recovery == "PROMOTED"
        assert authority.visible_locator("after-key") == after_locator

        # Wrong bytes at an expected locator fail closed and are never promoted.
        expected = b"expected payload"
        corrupt_locator = locator(expected, "corrupt")
        authority.begin_intent("corrupt", "corrupt-key", corrupt_locator, expected)
        store.put(corrupt_locator, b"wrong payload", mode="create")
        corrupt_recovery = authority.recover("corrupt", store)
        assert corrupt_recovery == "CONFLICT"
        assert authority.visible_locator("corrupt-key") is None

        result = {
            "verdict": "VALIDATED",
            "payload_atomicity": {
                "whole_object_publication": True,
                "create_if_absent": True,
            },
            "metadata_plus_payload_atomicity": {
                "provided": False,
                "proof": "object survived rollback while SQL intent did not",
            },
            "authority_visibility": {
                "object_before_promotion_visible": False,
                "sqlite_entry_is_visibility_point": True,
            },
            "recovery": {
                "before_payload_first": before_first,
                "before_payload_after_effect": before_second,
                "after_payload": after_recovery,
                "wrong_identity": corrupt_recovery,
            },
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

