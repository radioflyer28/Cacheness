# Phase 23: Encryption Schema & Storage — Validation Strategy

**Source:** Extracted from 23-RESEARCH.md Validation Architecture section
**Requirements:** ENC-01, ENC-02

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + pytest-xdist |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/test_sqlite_schema_versioning.py tests/test_pg_schema_versioning.py -x -q --ignore=tests/test_tensorflow_handler.py` |
| Full suite command | `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` |

## Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Wave |
|--------|----------|-----------|-------------------|------|
| ENC-01 | v3→v4 migration adds encryption columns (SQLite) | unit | `uv run pytest tests/test_sqlite_schema_versioning.py -x -q -k "v3_to_v4 or v4"` | 1 |
| ENC-01 | v3→v4 migration adds encryption columns (PG) | unit | `uv run pytest tests/test_pg_schema_versioning.py -x -q -k "v3_to_v4 or v4"` | 1 |
| ENC-01 | put_entry preserves encryption fields (SQLite) | unit | `uv run pytest tests/test_sqlite_schema_versioning.py -x -q -k "encryption"` | 1 |
| ENC-01 | put_entry preserves encryption fields (PG) | integration | `uv run pytest tests/test_pg_schema_versioning.py -x -q -k "encryption"` | 1 |
| ENC-01 | get_entry returns encryption fields (SQLite) | unit | `uv run pytest tests/test_sqlite_schema_versioning.py -x -q -k "encryption"` | 1 |
| ENC-02 | Encrypted roundtrip with SQLite backend | integration | `uv run pytest tests/test_sqlite_schema_versioning.py -x -q -k "roundtrip"` | 1 |
| ENC-02 | Encrypted roundtrip with PG backend | integration | `uv run pytest tests/test_pg_schema_versioning.py -x -q -k "roundtrip"` | 1 |

## Sampling Rate

- **Per task commit:** `uv run pytest tests/test_sqlite_schema_versioning.py tests/test_pg_schema_versioning.py -x -q --ignore=tests/test_tensorflow_handler.py`
- **Per wave merge:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
- **Phase gate:** Full suite green before `/gsd-verify-work`

## Wave Gaps (to be filled by plans)

- `tests/test_sqlite_schema_versioning.py` — add v3→v4 migration tests (Plan 01, Task 2)
- `tests/test_pg_schema_versioning.py` — add v3→v4 migration tests (Plan 02, Task 2)
