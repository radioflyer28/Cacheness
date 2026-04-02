# Phase 2: Metadata Package Split - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 02-metadata-package-split
**Areas discussed:** Module structure, PostgresBackend location

---

## Module Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror Phase 1 pattern | _compat.py (shared imports + ORM models + namespace utils), base.py (ABC + CachedMetadataBackend), json_backend.py, sqlite_backend.py, __init__.py (re-exports + factory) | ✓ |
| Separate ORM models | models.py (Base, CacheEntry, etc.), _compat.py (just imports), base.py, cached.py, json_backend.py, sqlite_backend.py, __init__.py | |
| Minimal split | Only json_backend.py + sqlite_backend.py extracted, everything else in __init__.py | |

**User's choice:** Mirror Phase 1 pattern
**Notes:** Consistent with established pattern from Phase 1. Keeps ORM models in _compat.py for simplicity.

---

## PostgresBackend Location

| Option | Description | Selected |
|--------|-------------|----------|
| Leave it | Keep PostgresBackend in storage/backends/postgresql_backend.py | ✓ |
| Move into metadata/ | Move to metadata/postgresql_backend.py alongside JSON and SQLite | |

**User's choice:** Leave it
**Notes:** Already works, avoid unnecessary churn.

---

## Agent's Discretion

- CachedMetadataBackend placement (with base.py vs own file)
- Import organization within _compat.py

## Deferred Ideas

None
