---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 2
total_count: 2
last_updated: 2026-08-29T20:35:05.358Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | tests/test_blob_backend_registry.py |  | Registry test updated to reject a slash-delimited direct filesystem ID under the strict opaque-ID contract. | fixed |  | 2026-08-29T20:21:54.242Z | 2026-08-29T20:23:22.401Z |
| 2 | 01 | deviation | src/cacheness/sql_cache.py |  | Removed duplicate best-effort gap failure logging so each unresolved condition emits one structured record. | fixed |  | 2026-08-29T20:34:13.738Z | 2026-08-29T20:35:05.358Z |

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
  }
]
````
