"""Validate the immutable, production-independent compatibility fixture corpus.

The validator intentionally does not import Cacheness.  It checks fixture bytes,
provenance, and bounded container framing before a production reader is allowed to
consume a fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import blosc2
import numpy as np
import xxhash


CORPUS_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
FIXTURE_SCHEMA_VERSION = 1
FIXED_DTYPE = "int32"
FIXED_SHAPE = [2, 3]
FIXED_VALUES = [0, 1, 2, 3, 4, 5]
FIXED_ARRAY = np.arange(6, dtype=np.int32).reshape(2, 3)
PROVENANCE_KEYS = {
    "schema_version",
    "fixture_id",
    "source_commit",
    "source_version",
    "generator_command",
    "environment",
    "logical_input",
    "discriminators",
    "files",
}
DECORATOR_PROVENANCE_KEYS = PROVENANCE_KEYS | {"decorator_key"}
DECORATOR_KEY_KEYS = {
    "module",
    "qualname",
    "args",
    "kwargs",
    "key_prefix",
    "serialized_args",
    "serialized_kwargs",
    "xxh3_64",
    "cache_entry_key",
}
MANIFEST_KEYS = {"schema_version", "fixtures"}
MANIFEST_RECORD_KEYS = {
    "source_commit",
    "source_version",
    "variant",
    "logical_input",
    "discriminators",
    "files",
    "production_reader",
}
ENVIRONMENT_KEYS = {"cacheness", "python", "numpy", "blosc2"}
LOGICAL_INPUT_KEYS = {"dtype", "shape", "values"}
DIGEST_KEYS = {"source_sha256", "copied_sha256"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_SHAPE_RE = re.compile(r"^\([0-9]+, [0-9]+\)$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")

SPLIT_JSON_KEYS = [
    "entries",
    "access_times",
    "creation_times",
    "file_sizes",
    "data_types",
    "cache_key_params",
    "cache_hits",
    "cache_misses",
]
SIGNED_SPLIT_JSON_KEYS = [*SPLIT_JSON_KEYS, "entry_signature"]
NESTED_JSON_KEYS = ["entries", "cache_hits", "cache_misses"]
LEGACY_SQLITE_COLUMNS = [
    "cache_key",
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "file_hash",
    "entry_signature",
    "cache_key_params",
    "metadata_json",
]
CURRENT_SQLITE_COLUMNS = [
    "cache_key",
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "file_hash",
    "entry_signature",
    "object_type",
    "storage_format",
    "serializer",
    "compression_codec",
    "actual_path",
    "cache_key_params",
]

# This table is deliberately complete before later fixture plans append files.  It
# is the normative matrix, not a manifest inferred from whatever happens to exist.
EXPECTED_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "array-raw-v035-compress",
        "commit": "041c930fb66c7aa23f53d1f9f524e9fafdd20e68",
        "version": "0.3.5",
        "variant": "raw-array",
        "files": ["payload.b2nd", "provenance.json"],
        "production_reader": "legacy_raw_blosc2",
        "discriminators": {
            "storage_format": "blosc2",
            "frame_api": "compress",
            "shape_text": "(2, 3)",
            "dtype_text": "int32",
            "signature_state": "not_applicable",
        },
    },
    {
        "id": "array-raw-v037-compress2",
        "commit": "a756d70c858cec13ff1c885e2316c0fe725c4949",
        "version": "0.3.7",
        "variant": "raw-array",
        "files": ["payload.b2nd", "provenance.json"],
        "production_reader": "legacy_raw_blosc2",
        "discriminators": {
            "storage_format": "blosc2",
            "frame_api": "compress2",
            "shape_text": "(2, 3)",
            "dtype_text": "int32",
            "signature_state": "not_applicable",
        },
    },
    {
        "id": "json-split-unsigned-v037",
        "commit": "a756d70c858cec13ff1c885e2316c0fe725c4949",
        "version": "0.3.7",
        "variant": "json-split-npz",
        "files": ["metadata.json", "payload.npz", "provenance.json"],
        "production_reader": "legacy_split_json_metadata",
        "discriminators": {
            "metadata_kind": "json_split",
            "json_top_level_keys": SPLIT_JSON_KEYS,
            "signature_state": "absent",
        },
    },
    {
        "id": "json-split-signed-v038",
        "commit": "71e4ba04cbcd7dfbcf74e5651f131213b7d45ab9",
        "version": "0.3.8",
        "variant": "json-split-npz",
        "files": ["metadata.json", "payload.npz", "provenance.json"],
        "production_reader": "legacy_split_json_metadata",
        "discriminators": {
            "metadata_kind": "json_split",
            "json_top_level_keys": SIGNED_SPLIT_JSON_KEYS,
            "signature_state": "present",
        },
    },
    {
        "id": "sqlite-metadata-json-v039",
        "commit": "6e2f9933a9aa66fb6629aec0125f84a0c114c35f",
        "version": "0.3.9",
        "variant": "sqlite-metadata-json-npz",
        "files": ["metadata.sqlite3", "payload.npz", "provenance.json"],
        "production_reader": "legacy_sqlite_metadata_json",
        "discriminators": {
            "metadata_kind": "sqlite",
            "table": "cache_entries",
            "sqlite_columns": LEGACY_SQLITE_COLUMNS,
            "signature_state": "column_present",
        },
    },
    {
        "id": "decorator-key-v0313",
        "commit": "76a469f3e090a99d9f9c119c3190a97b11a5e68a",
        "version": "0.3.13",
        "variant": "decorator-key-json-npz",
        "files": ["metadata.json", "payload.npz", "provenance.json"],
        "production_reader": "legacy_decorator_key",
        "discriminators": {
            "metadata_kind": "json_nested",
            "json_top_level_keys": NESTED_JSON_KEYS,
            "signature_state": "entry_present",
            "key_scheme": "decorator_pre_unified",
        },
    },
    {
        "id": "json-nested-v0314",
        "commit": "a22f4b4575cb8213d9783ed388d2a70727563db1",
        "version": "0.3.14",
        "variant": "json-nested-npz",
        "files": ["metadata.json", "payload.npz", "provenance.json"],
        "production_reader": "current_json_metadata",
        "discriminators": {
            "metadata_kind": "json_nested",
            "json_top_level_keys": NESTED_JSON_KEYS,
            "signature_state": "entry_present",
            "key_scheme": "unified",
        },
    },
    {
        "id": "sqlite-columns-v0314",
        "commit": "a22f4b4575cb8213d9783ed388d2a70727563db1",
        "version": "0.3.14",
        "variant": "sqlite-columns-npz",
        "files": ["metadata.sqlite3", "payload.npz", "provenance.json"],
        "production_reader": "current_sqlite_columns",
        "discriminators": {
            "metadata_kind": "sqlite",
            "table": "cache_entries",
            "sqlite_columns": CURRENT_SQLITE_COLUMNS,
            "signature_state": "column_present",
        },
    },
)


class CorpusValidationError(ValueError):
    """Raised when an immutable fixture fails its pre-read safety gate."""


def fail(message: str) -> None:
    raise CorpusValidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {path.relative_to(CORPUS_ROOT)}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path.relative_to(CORPUS_ROOT)} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digests(paths: list[Path]) -> dict[Path, str]:
    return {path: sha256(path) for path in paths}


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        fail(f"{label} keys differ: expected {sorted(expected)}, got {sorted(actual)}")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty relative path")
    if "\\" in value or value.startswith("//") or DRIVE_RE.match(value):
        fail(f"{label} is not a portable relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        fail(f"{label} is not a normalized contained path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        fail(f"{label} is not normalized: {value!r}")
    return normalized


def validate_logical_input(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    require_exact_keys(value, LOGICAL_INPUT_KEYS, label)
    if (
        value["dtype"] != FIXED_DTYPE
        or value["shape"] != FIXED_SHAPE
        or value["values"] != FIXED_VALUES
    ):
        fail(f"{label} does not describe the fixed int32 (2, 3) semantic input")


def validate_environment(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    require_exact_keys(value, ENVIRONMENT_KEYS, label)
    if not all(isinstance(version, str) and version for version in value.values()):
        fail(f"{label} contains an empty toolchain version")


def expected_prefix(expected_through: str) -> tuple[dict[str, Any], ...]:
    ids = [fixture["id"] for fixture in EXPECTED_FIXTURES]
    if expected_through not in ids:
        fail(f"unknown --expected-through fixture: {expected_through}")
    return EXPECTED_FIXTURES[: ids.index(expected_through) + 1]


def validate_manifest(expected: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != FIXTURE_SCHEMA_VERSION:
        fail("manifest schema_version is not the supported fixture schema")
    fixtures = manifest["fixtures"]
    if not isinstance(fixtures, dict):
        fail("manifest fixtures must be an insertion-ordered object")
    expected_ids = [fixture["id"] for fixture in expected]
    if list(fixtures) != expected_ids:
        fail(f"manifest fixture order must be exactly {expected_ids}")
    return manifest


def validate_manifest_record(
    fixture: dict[str, Any], expected: dict[str, Any]
) -> tuple[Path, list[Path]]:
    fixture_id = expected["id"]
    if not isinstance(fixture, dict):
        fail(f"manifest fixture {fixture_id} must be an object")
    require_exact_keys(fixture, MANIFEST_RECORD_KEYS, f"manifest fixture {fixture_id}")
    if fixture["source_commit"] != expected["commit"]:
        fail(f"{fixture_id} source commit does not match the normative matrix")
    if fixture["source_version"] != expected["version"]:
        fail(f"{fixture_id} source version does not match the normative matrix")
    if fixture["variant"] != expected["variant"]:
        fail(f"{fixture_id} variant does not match the normative matrix")
    if fixture["production_reader"] != expected["production_reader"]:
        fail(f"{fixture_id} production reader does not match the normative matrix")
    validate_logical_input(fixture["logical_input"], f"manifest {fixture_id} logical_input")
    if fixture["discriminators"] != expected["discriminators"]:
        fail(f"{fixture_id} discriminators do not match the complete normative matrix")
    files = fixture["files"]
    if not isinstance(files, dict):
        fail(f"manifest fixture {fixture_id} files must be an object")
    expected_files = expected["files"]
    if list(files) != expected_files:
        fail(f"{fixture_id} files must be exactly {expected_files}")
    fixture_dir = CORPUS_ROOT / fixture_id
    if not fixture_dir.is_dir():
        fail(f"fixture directory is missing: {fixture_id}")
    files_on_disk: list[Path] = []
    for relative_path, file_record in files.items():
        safe_path = validate_relative_path(relative_path, f"{fixture_id} manifest file")
        if not isinstance(file_record, dict):
            fail(f"{fixture_id}/{safe_path} manifest record must be an object")
        require_exact_keys(file_record, {"sha256"}, f"{fixture_id}/{safe_path}")
        manifest_digest = require_sha256(
            file_record["sha256"], f"{fixture_id}/{safe_path} manifest digest"
        )
        candidate = fixture_dir / safe_path
        if not candidate.is_file():
            fail(f"listed fixture file is missing: {fixture_id}/{safe_path}")
        if sha256(candidate) != manifest_digest:
            fail(f"manifest digest drift: {fixture_id}/{safe_path}")
        files_on_disk.append(candidate)
    actual_files = sorted(
        path.relative_to(fixture_dir).as_posix()
        for path in fixture_dir.rglob("*")
        if path.is_file()
    )
    if actual_files != sorted(expected_files):
        fail(f"{fixture_id} contains unlisted or missing files: {actual_files}")
    return fixture_dir, files_on_disk


def validate_decorator_key_provenance(
    fixture_dir: Path, provenance: dict[str, Any]
) -> None:
    """Recompute the sole allowed legacy decorator candidate without scanning."""
    decorator_key = provenance["decorator_key"]
    if not isinstance(decorator_key, dict):
        fail("decorator-key-v0313 decorator_key must be an object")
    require_exact_keys(
        decorator_key,
        DECORATOR_KEY_KEYS,
        "decorator-key-v0313 decorator_key",
    )
    expected_values = {
        "module": "compat_fixture_v0313",
        "qualname": "fixture_array",
        "args": [6],
        "kwargs": {"offset": 0},
        "key_prefix": "compat-v0313",
        "serialized_args": "tuple:[int:6]",
        "serialized_kwargs": "dict:[str:offset:int:0]",
    }
    for name, value in expected_values.items():
        if decorator_key[name] != value:
            fail(f"decorator-key-v0313 decorator_key {name} differs")

    function_id = f"{decorator_key['module']}.{decorator_key['qualname']}"
    candidate_base = (
        f"{decorator_key['key_prefix']}:{function_id}:"
        f"args:{decorator_key['serialized_args']}:"
        f"kwargs:{decorator_key['serialized_kwargs']}"
    )
    candidate = xxhash.xxh3_64(candidate_base.encode()).hexdigest()
    if decorator_key["xxh3_64"] != candidate:
        fail("decorator-key-v0313 decorator candidate digest differs")

    storage_base = f"__decorator_cache_key:str:{candidate}"
    storage_key = xxhash.xxh3_64(storage_base.encode()).hexdigest()[:16]
    if decorator_key["cache_entry_key"] != storage_key:
        fail("decorator-key-v0313 cache entry key differs")

    metadata = read_json(fixture_dir / "metadata.json")
    entries = metadata.get("entries")
    if not isinstance(entries, dict):
        fail("decorator-key-v0313 metadata entries must be an object")
    entry = entries.get(storage_key)
    if not isinstance(entry, dict):
        fail("decorator-key-v0313 exact candidate is absent from metadata")
    if entry.get("description") != f"Cached result for {candidate}":
        fail("decorator-key-v0313 candidate entry description differs")


def validate_provenance(
    fixture_dir: Path, manifest_record: dict[str, Any], expected: dict[str, Any]
) -> None:
    fixture_id = expected["id"]
    provenance = read_json(fixture_dir / "provenance.json")
    provenance_keys = (
        DECORATOR_PROVENANCE_KEYS
        if fixture_id == "decorator-key-v0313"
        else PROVENANCE_KEYS
    )
    require_exact_keys(provenance, provenance_keys, f"{fixture_id} provenance")
    if provenance["schema_version"] != FIXTURE_SCHEMA_VERSION:
        fail(f"{fixture_id} provenance schema_version is unsupported")
    for key, expected_value in (
        ("fixture_id", fixture_id),
        ("source_commit", expected["commit"]),
        ("source_version", expected["version"]),
    ):
        if provenance[key] != expected_value:
            fail(f"{fixture_id} provenance {key} does not match the normative matrix")
    if not isinstance(provenance["generator_command"], str) or not provenance[
        "generator_command"
    ]:
        fail(f"{fixture_id} provenance must describe its writer-only generator command")
    validate_environment(provenance["environment"], f"{fixture_id} provenance environment")
    validate_logical_input(
        provenance["logical_input"], f"{fixture_id} provenance logical_input"
    )
    if provenance["discriminators"] != expected["discriminators"]:
        fail(f"{fixture_id} provenance discriminators do not match the normative matrix")
    provenance_files = provenance["files"]
    if not isinstance(provenance_files, dict):
        fail(f"{fixture_id} provenance files must be an object")
    evidence_files = [name for name in expected["files"] if name != "provenance.json"]
    if list(provenance_files) != evidence_files:
        fail(f"{fixture_id} provenance files must be exactly {evidence_files}")
    for relative_path, file_record in provenance_files.items():
        safe_path = validate_relative_path(relative_path, f"{fixture_id} provenance file")
        if not isinstance(file_record, dict):
            fail(f"{fixture_id}/{safe_path} provenance record must be an object")
        require_exact_keys(file_record, DIGEST_KEYS, f"{fixture_id}/{safe_path}")
        source_digest = require_sha256(
            file_record["source_sha256"], f"{fixture_id}/{safe_path} source digest"
        )
        copied_digest = require_sha256(
            file_record["copied_sha256"], f"{fixture_id}/{safe_path} copied digest"
        )
        if source_digest != copied_digest:
            fail(f"{fixture_id}/{safe_path} source and copied digests differ")
        current_digest = sha256(fixture_dir / safe_path)
        if current_digest != source_digest:
            fail(f"{fixture_id}/{safe_path} differs from its source/copy digest")
    provenance_digest = sha256(fixture_dir / "provenance.json")
    if manifest_record["files"]["provenance.json"]["sha256"] != provenance_digest:
        fail(f"{fixture_id} provenance digest is not independently recorded in manifest")
    if fixture_id == "decorator-key-v0313":
        validate_decorator_key_provenance(fixture_dir, provenance)


def validate_raw_fixture(fixture_dir: Path, expected: dict[str, Any]) -> None:
    raw = (fixture_dir / "payload.b2nd").read_bytes()
    if len(raw) < 13:
        fail(f"{expected['id']} raw frame is too short")
    offset = 0
    for field in ("shape", "dtype"):
        if offset + 4 > len(raw):
            fail(f"{expected['id']} raw {field} length is truncated")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if not 1 <= length <= 64 or offset + length > len(raw):
            fail(f"{expected['id']} raw {field} length is out of bounds")
        try:
            text = raw[offset : offset + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            fail(f"{expected['id']} raw {field} is not UTF-8: {exc}")
        offset += length
        expected_text = expected["discriminators"][f"{field}_text"]
        if text != expected_text:
            fail(f"{expected['id']} raw {field} discriminator differs")
        if field == "shape" and not RAW_SHAPE_RE.fullmatch(text):
            fail(f"{expected['id']} raw shape grammar is unsafe")
        if field == "dtype" and text != FIXED_DTYPE:
            fail(f"{expected['id']} raw dtype grammar is unsafe")
    compressed = raw[offset:]
    if not compressed:
        fail(f"{expected['id']} has no compressed raw payload")
    try:
        if expected["discriminators"]["frame_api"] == "compress":
            decompressed = blosc2.decompress(compressed)
        else:
            decompressed = blosc2.decompress2(compressed)
    except Exception as exc:
        fail(f"{expected['id']} raw frame cannot be decompressed: {exc}")
    if decompressed != FIXED_ARRAY.tobytes(order="C"):
        fail(f"{expected['id']} raw frame bytes do not match the fixed semantic input")


def validate_npz_fixture(fixture_dir: Path, expected: dict[str, Any]) -> None:
    path = fixture_dir / "payload.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["data"]:
                fail(f"{expected['id']} NPZ members must be exactly ['data']")
            array = archive["data"]
    except (OSError, ValueError, KeyError) as exc:
        fail(f"{expected['id']} cannot be safely inspected as NPZ: {exc}")
    if array.dtype.hasobject:
        fail(f"{expected['id']} NPZ contains an object dtype")
    if str(array.dtype) != FIXED_DTYPE or list(array.shape) != FIXED_SHAPE:
        fail(f"{expected['id']} NPZ dtype/shape differs from fixed semantic input")
    if not np.array_equal(array, FIXED_ARRAY):
        fail(f"{expected['id']} NPZ values differ from fixed semantic input")


def validate_json_fixture(fixture_dir: Path, expected: dict[str, Any]) -> None:
    metadata = read_json(fixture_dir / "metadata.json")
    expected_keys = expected["discriminators"]["json_top_level_keys"]
    if set(metadata) != set(expected_keys):
        fail(f"{expected['id']} JSON discriminator set is incomplete or mixed")
    signature_state = expected["discriminators"]["signature_state"]
    has_signature = "entry_signature" in metadata
    if signature_state == "absent" and has_signature:
        fail(f"{expected['id']} unexpectedly contains entry_signature")
    if signature_state == "present" and not has_signature:
        fail(f"{expected['id']} is missing entry_signature")


def validate_sqlite_fixture(fixture_dir: Path, expected: dict[str, Any]) -> None:
    source = fixture_dir / "metadata.sqlite3"
    source_digest_before = sha256(source)
    with tempfile.TemporaryDirectory(prefix="cacheness-compat-") as temporary_dir:
        copy = Path(temporary_dir) / "metadata.sqlite3"
        shutil.copy2(source, copy)
        copy_digest_before = sha256(copy)
        if source_digest_before != copy_digest_before:
            fail(f"{expected['id']} SQLite inspection copy differs from source")
        uri = f"file:{copy.as_posix()}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True)
            try:
                data_version_before = connection.execute("PRAGMA data_version").fetchone()[0]
                columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(cache_entries)").fetchall()
                ]
                data_version_after = connection.execute("PRAGMA data_version").fetchone()[0]
            finally:
                connection.close()
        except sqlite3.Error as exc:
            fail(f"{expected['id']} SQLite inspection failed: {exc}")
        if columns != expected["discriminators"]["sqlite_columns"]:
            fail(f"{expected['id']} SQLite schema is incomplete, reordered, or mixed")
        if data_version_before != data_version_after:
            fail(f"{expected['id']} SQLite data_version changed during read-only inspection")
        if sha256(copy) != copy_digest_before or sha256(source) != source_digest_before:
            fail(f"{expected['id']} SQLite source or inspection copy mutated")


def validate_fixture(manifest_record: dict[str, Any], expected: dict[str, Any]) -> None:
    fixture_dir, files_on_disk = validate_manifest_record(manifest_record, expected)
    before = snapshot_digests(files_on_disk)
    validate_provenance(fixture_dir, manifest_record, expected)
    if "payload.b2nd" in expected["files"]:
        validate_raw_fixture(fixture_dir, expected)
    if "payload.npz" in expected["files"]:
        validate_npz_fixture(fixture_dir, expected)
    if "metadata.json" in expected["files"]:
        validate_json_fixture(fixture_dir, expected)
    if "metadata.sqlite3" in expected["files"]:
        validate_sqlite_fixture(fixture_dir, expected)
    after = snapshot_digests(files_on_disk)
    if before != after:
        fail(f"{expected['id']} fixture bytes changed during inspection")


def validate_corpus(expected_through: str) -> None:
    expected = expected_prefix(expected_through)
    manifest = validate_manifest(expected)
    for fixture in expected:
        validate_fixture(manifest["fixtures"][fixture["id"]], fixture)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-through",
        required=True,
        choices=[fixture["id"] for fixture in EXPECTED_FIXTURES],
        help="Require exactly the accumulated normative fixture prefix ending at this ID.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_corpus(args.expected_through)
    except CorpusValidationError as exc:
        print(f"compatibility corpus validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"compatibility corpus validated through {args.expected_through}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
