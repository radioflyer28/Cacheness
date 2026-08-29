---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 8
total_count: 8
last_updated: 2026-08-29T23:08:07.482Z
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
| 5 | 01 | deviation | tests/test_stored_compatibility.py |  | Decorator fixture startup cleanup is disabled so strict generic locator preflight remains intact and the exact guarded fallback owns in-memory rebasing. | fixed |  | 2026-08-29T23:03:37.544Z | 2026-08-29T23:03:46.637Z |
| 6 | 01 | deviation | src/cacheness/core.py |  | Invalid current signatures remain fail-closed unless the exact signed split-map discriminator selects the documented legacy verifier. | fixed |  | 2026-08-29T23:03:43.045Z | 2026-08-29T23:03:46.749Z |
| 7 | 01 | deviation | tests/test_stored_compatibility.py |  | Phase-owned compatibility tests use named local invariance helpers and pass their zero-findings Ruff gate. | fixed |  | 2026-08-29T23:03:43.152Z | 2026-08-29T23:03:46.861Z |
| 8 | 01 | deviation | src/cacheness/core.py |  | Unrecognized legacy-signature metadata now rejects before unsigned policy and cannot reach a handler. | fixed |  | 2026-08-29T23:08:02.399Z | 2026-08-29T23:08:07.482Z |

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
  },
  {
    "id": 5,
    "kind": "deviation",
    "phase": "01",
    "file": "tests/test_stored_compatibility.py",
    "line": null,
    "description": "Decorator fixture startup cleanup is disabled so strict generic locator preflight remains intact and the exact guarded fallback owns in-memory rebasing.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T23:03:37.544Z",
    "resolved_at": "2026-08-29T23:03:46.637Z"
  },
  {
    "id": 6,
    "kind": "deviation",
    "phase": "01",
    "file": "src/cacheness/core.py",
    "line": null,
    "description": "Invalid current signatures remain fail-closed unless the exact signed split-map discriminator selects the documented legacy verifier.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T23:03:43.045Z",
    "resolved_at": "2026-08-29T23:03:46.749Z"
  },
  {
    "id": 7,
    "kind": "deviation",
    "phase": "01",
    "file": "tests/test_stored_compatibility.py",
    "line": null,
    "description": "Phase-owned compatibility tests use named local invariance helpers and pass their zero-findings Ruff gate.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T23:03:43.152Z",
    "resolved_at": "2026-08-29T23:03:46.861Z"
  },
  {
    "id": 8,
    "kind": "deviation",
    "phase": "01",
    "file": "src/cacheness/core.py",
    "line": null,
    "description": "Unrecognized legacy-signature metadata now rejects before unsigned policy and cannot reach a handler.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-29T23:08:02.399Z",
    "resolved_at": "2026-08-29T23:08:07.482Z"
  }
]
````
