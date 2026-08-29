---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 4
total_count: 4
last_updated: 2026-08-29T22:17:14.936Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | tests/test_blob_backend_registry.py |  | Registry test updated to reject a slash-delimited direct filesystem ID under the strict opaque-ID contract. | fixed |  | 2026-08-29T20:21:54.242Z | 2026-08-29T20:23:22.401Z |
| 2 | 01 | deviation | src/cacheness/sql_cache.py |  | Removed duplicate best-effort gap failure logging so each unresolved condition emits one structured record. | fixed |  | 2026-08-29T20:34:13.738Z | 2026-08-29T20:35:05.358Z |
| 3 | 01 | deviation | tests/fixtures/compat/validate_corpus.py |  | Added direct verification for the declared 0.3.14 unified JSON key and signed entry. | fixed |  | 2026-08-29T22:17:11.298Z | 2026-08-29T22:17:14.838Z |
| 4 | 01 | deviation | tests/fixtures/compat/validate_corpus.py |  | Added read-only provenance verification for the 0.3.14 SQLite schema and data_version. | fixed |  | 2026-08-29T22:17:11.407Z | 2026-08-29T22:17:14.936Z |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "01",
    "file": "tests/test_blob_backend_registry.py",
    "line": null,
    "description": "Registry test updated to reject a slash-delimited direct filesystem ID under the strict opaque-ID contract.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T20:21:54.242Z",
    "resolved_at": "2026-08-29T20:23:22.401Z"
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "01",
    "file": "src/cacheness/sql_cache.py",
    "line": null,
    "description": "Removed duplicate best-effort gap failure logging so each unresolved condition emits one structured record.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T20:34:13.738Z",
    "resolved_at": "2026-08-29T20:35:05.358Z"
  },
  {
    "id": 3,
    "kind": "deviation",
    "phase": "01",
    "file": "tests/fixtures/compat/validate_corpus.py",
    "line": null,
    "description": "Added direct verification for the declared 0.3.14 unified JSON key and signed entry.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T22:17:11.298Z",
    "resolved_at": "2026-08-29T22:17:14.838Z"
  },
  {
    "id": 4,
    "kind": "deviation",
    "phase": "01",
    "file": "tests/fixtures/compat/validate_corpus.py",
    "line": null,
    "description": "Added read-only provenance verification for the 0.3.14 SQLite schema and data_version.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T22:17:11.407Z",
    "resolved_at": "2026-08-29T22:17:14.936Z"
  }
]
````
