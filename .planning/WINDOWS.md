---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 1
total_count: 1
last_updated: 2026-08-29T20:23:22.401Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | tests/test_blob_backend_registry.py |  | Registry test updated to reject a slash-delimited direct filesystem ID under the strict opaque-ID contract. | fixed |  | 2026-08-29T20:21:54.242Z | 2026-08-29T20:23:22.401Z |

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
  }
]
````
