---
phase: 04-metadata-composition-and-topology-contracts
verified: 2026-09-08T08:59:08Z
status: passed
score: 18/18 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
re_verification:
  previous_status: gaps_found
  previous_score: 14/16
  gaps_closed:
    - "Every advertised built-in projection now satisfies the strict ProjectionSink composition contract; PostgreSQL is no longer advertised before Phase 5 qualification."
    - "The executable-consumer audit now detects direct, aliased, bound, and implementation-star retired API uses through the same tested visitor used by the repository audit."
    - "The accidental repository-root metadata.py near-copy is absent."
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "The three pandas-dependent SQL-cache modules collect and execute in a qualified optional-dependency installation."
    addressed_in: "Phase 8"
    evidence: "Phase 8 owns advertised optional-dependency installation/import qualification; Phase 4 keeps these exact paths diagnostic-only and explicitly non-green."
  - truth: "Live PostgreSQL and S3 lifecycle/topology behavior is qualified."
    addressed_in: "Phase 5"
    evidence: "Phase 5 owns real-service and supported-pair qualification. PostgreSQL remains role-classified but unregistered until a real derived sink is qualified."
---

# Phase 4: Metadata Composition and Topology Contracts Verification Report

**Phase Goal:** Users can customize BlobStore catalog metadata without implementing lifecycle sequencing, select advertised metadata backends through one composition root, and receive only guarantees supported by the topology.
**Verified:** 2026-09-08T08:59:08Z
**Status:** passed
**Re-verification:** Yes — final verification after Plan 04-14

## Goal Achievement

Phase 4 achieves its goal. Direct users can define and query native catalog metadata through `BlobStore`; the exact selected payload and authority participants perform the work; one topology-owned registry resolves built-in and application roles; construction rejects unsupported claims before I/O; projections remain derived and preserve committed receipts; and current-format/version boundaries remain explicit and non-mutating.

Plan 04-14 closes the last composition and evidence gaps without reopening lifecycle coordination. JSON is a real replay-safe `ProjectionSink`, PostgreSQL is honestly deferred rather than advertised as a placeholder, the cutover audit catches every claimed retired-import form, and the accidental root `metadata.py` is gone.

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Memory/SQLite authorities and JSON/PostgreSQL projection families have explicit, truthful roles and supported capability tiers. | ✓ VERIFIED | Memory and SQLite satisfy `LifecycleAuthority`; JSON is the sole constructible built-in projection; PostgreSQL remains derived-role classified but unregistered until Phase 5. |
| 2 | Every advertised built-in projection resolves through `StoreTopology` and satisfies strict `ProjectionSink` validation. | ✓ VERIFIED | Direct construction reports `JSON_SINK True`; named JSON topology resolution succeeds without bypassing `_validate_participant_role`. |
| 3 | JSON projection apply/checkpoint/reopen behavior is bounded, idempotent, and derived-only. | ✓ VERIFIED | Behavioral tests persist a real `CatalogEntry`, reopen the checkpoint, and replay a pending batch after simulated checkpoint failure without duplication. |
| 4 | PostgreSQL is not prematurely advertised as constructible. | ✓ VERIFIED | `RoleRegistry.resolve(PROJECTION, "postgresql")` raises typed `CompositionValidationError`, while `resolve_metadata_role("postgresql")` grants no authority permissions. |
| 5 | One composition root preserves exact injected and application-registered participants. | ✓ VERIFIED | High-level custom named payload/authority/projection tests pass through `StoreTopology.role_registry`; no alternate registry remains. |
| 6 | Capability reports describe the participants that actually perform payload and authority operations. | ✓ VERIFIED | Selected-root A/B tests pass and `BlobStore` obtains generation I/O from the selected payload participant. |
| 7 | Invalid topology claims and participants fail before I/O, and store-owned resources unwind once by identity. | ✓ VERIFIED | Structural role, capability-minimum, duplicate-identity, caller-owned, and reverse close-order tests pass. |
| 8 | Direct users define, validate, write, update, reopen, and query declared catalog metadata. | ✓ VERIFIED | Public memory/SQLite put-update-query-reopen tests pass; invalid schema values do not reach handler, payload, authority, or projection work. |
| 9 | Native schema values distinguish stored absence, materialized defaults, explicit null, and bounded opaque fields. | ✓ VERIFIED | Catalog schema/value tests exercise all four states through the public lifecycle. |
| 10 | Canonical queries are authenticated, revision-bound, deterministic, complete for the authority snapshot, and work-capped. | ✓ VERIFIED | Memory/SQLite dense, sparse, tampered, stale-cursor, and public round-trip tests pass. |
| 11 | Cursor inputs are bounded before base64, JSON, HMAC, manifest loading, and authority dispatch. | ✓ VERIFIED | Encoded, decoded, field, signature, exact-shape, and zero-dispatch boundary tests pass. |
| 12 | Projection failures preserve committed receipts and named rebuild checks use the selected sink's capability. | ✓ VERIFIED | Arbitrary `Exception`, `BaseException`, exact receipt, committed-partial, dirty outcome, and mixed-capability rebuild tests pass. |
| 13 | Format/schema versions remain independent and unsupported layouts fail typed without mutation. | ✓ VERIFIED | Current-format reopen and obsolete/foreign/corrupt layout tests pass; no runtime upgrade path was added. |
| 14 | UnifiedCache remains a policy facade over one internal BlobStore engine without the Phase 6 policy redesign. | ✓ VERIFIED | Core and lifecycle-authority regression modules pass through the selected BlobStore topology. |
| 15 | Superseded factories, registries, result shapes, and runtime ORM authority hooks are absent. | ✓ VERIFIED | Public absence tests and the strengthened executable-tree audit pass. |
| 16 | The cutover audit detects direct root imports, root aliases, backend-submodule aliases, bound aliases, and implementation-module star imports without string/root-star false positives. | ✓ VERIFIED | Fifteen table-driven positive/negative fixtures call the same `audit_source()` path as repository scanning; direct adversarial repros now produce findings. |
| 17 | Retained lifecycle, integrity, containment, concurrency, initialization, and cleanup regressions execute on current seams. | ✓ VERIFIED | The exact 42-module owned matrix passes on Python 3.11 and 3.13. |
| 18 | Phase 4 adds no Ruff findings and preserves honest service/platform/optional-dependency non-claims. | ✓ VERIFIED | Ruff delta passes on both interpreters; PostgreSQL/S3/Windows and pandas SQL-cache limitations remain explicit. |

**Score:** 18/18 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/catalog.py` | Native schema/query/cursor contract | ✓ VERIFIED | Substantive, wired through BlobStore, bounded, and behaviorally exercised. |
| `src/cacheness/storage/composition.py` | One truthful role-aware topology resolver | ✓ VERIFIED | JSON-only built-in projection inventory, strict role validation, exact selection, and owned-resource ledger are wired. |
| `src/cacheness/storage/blob_store.py` | Sole public payload/catalog facade | ✓ VERIFIED | Public catalog, selected payload, authority, projection-controller, and UnifiedCache links are live. |
| `src/cacheness/metadata.py` | Concrete derived JSON projection | ✓ VERIFIED | Implements all three sink methods, versioned exact-shape state, atomic sibling replacement, pending-batch replay, and validated checkpoint loading. |
| `src/cacheness/storage/projections.py` | Bounded derived projection controller | ✓ VERIFIED | Pull/checkpoint/partial/rebuild behavior remains subordinate to canonical authority. |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | Format-2 local canonical authority | ✓ VERIFIED | Substantive, wired, and covered by lifecycle/catalog tests. |
| `src/cacheness/storage/memory_lifecycle_authority.py` | Same-process canonical authority | ✓ VERIFIED | Substantive, wired, and covered by behavior tests. |
| `src/cacheness/storage/read_contract.py` | Frozen `BlobReceipt` result | ✓ VERIFIED | Immutable receipts carry exact expectations and named projection outcomes. |
| `tools/verify_phase4_cutover.py` | Consumer audit and bounded release matrix | ✓ VERIFIED | Shared `audit_source()` visitor passes adversarial fixtures and current-tree audit; matrix/deferred parsing remains disjoint. |
| `tests/test_phase4_cutover_verifier.py` | Audit regression fixtures | ✓ VERIFIED | Exists, substantive, imported against the actual tool, and exercises positive and negative syntax variants. |
| `docs/CATALOG_AND_TOPOLOGY.md` | Published consistency/topology boundaries | ✓ VERIFIED | Accurately documents JSON as the only built-in projection and PostgreSQL as Phase 5-deferred. |
| `04-VALIDATION.md` | Reproducible two-interpreter evidence | ✓ VERIFIED | Exact 42-path owned matrix, separate three-path diagnostic set, cleanup record, and non-claims are present. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `BlobStore.put_entry/update_catalog` | `CatalogSchema` and authority descriptor transaction | validated values and exact expectation | ✓ WIRED | Commit, patch, stale conflict, query, and reopen tests pass. |
| selected payload participant | lifecycle engine handler I/O | `materialize_handler_io()` | ✓ WIRED | Distinct-root test proves no cache-dir substitution. |
| `StoreTopology` | application `RoleRegistry` | topology-owned resolution | ✓ WIRED | Custom participants reach high-level BlobStore construction. |
| `RoleRegistry` JSON built-in | `JsonProjection` / `ProjectionSink` | `_construct_json_projection` | ✓ WIRED | Factory constructs a protocol-valid sink and topology resolution succeeds. |
| `JsonProjection` | `ProjectionBatch` / `ProjectionCheckpoint` | apply then checkpoint | ✓ WIRED | Persisted pending identity makes restart replay a no-op before matching checkpoint advancement. |
| `BlobStore` | `ProjectionController` | exact receipt and per-sink capabilities | ✓ WIRED | Post-commit outcome and named rebuild tests pass. |
| `tools/verify_phase4_cutover.py` | executable consumers | shared alias-aware AST visitor | ✓ WIRED | Fixture tests and repository audit call the same `audit_source()` implementation. |
| `UnifiedCache` | `BlobStore` | narrow internal composition | ✓ WIRED | Cache-policy facade tests pass without a second lifecycle path. |

### Data-Flow Trace

| Artifact | Data | Source | Status |
|---|---|---|---|
| `BlobStore.query_catalog` | `CatalogPage.entries` | Authenticated authority descriptors written by public catalog lifecycle | ✓ FLOWING |
| selected filesystem/memory payload | immutable generation bytes | Exact composed payload participant | ✓ FLOWING |
| `ProjectionController` | bounded batches/checkpoints | Canonical query pages after authority commit | ✓ FLOWING |
| `JsonProjection` | derived entry/checkpoint document | Validated `ProjectionBatch` plus matching `ProjectionCheckpoint` | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Final projection/audit contracts | Focused pytest over JSON, roles, PostgreSQL deferral, composition, and cutover-verifier modules | Passed | ✓ PASS |
| Direct JSON topology | Construct JSON through `RoleRegistry` and memory/memory `StoreTopology` | Protocol-valid sink; topology resolves | ✓ PASS |
| PostgreSQL deferral | Resolve named PostgreSQL projection | Typed `CompositionValidationError` | ✓ PASS |
| Prior AST false negatives | Direct root, aliased root, submodule alias, and direct backend import snippets | All now produce retired-use findings | ✓ PASS |
| AST negatives | Root package star import and string-only snippet | No false-positive findings | ✓ PASS |
| Python 3.11 owned gate | `uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all` | 607 passed, 6 skipped across 42 modules | ✓ PASS |
| Python 3.13 owned gate | `uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all` | 607 passed, 6 skipped across 42 modules | ✓ PASS |
| Ruff delta | `verify_phase4_ruff_delta.py` on Python 3.11 and 3.13 | Passed on both | ✓ PASS |
| Root draft cleanup | `test ! -e metadata.py` | Exit 0; no tracked or untracked root module | ✓ PASS |

### Probe Execution

No Phase 4 probe scripts are declared or present. The checked-in cutover verifier was executed independently on both supported interpreters.

### Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| BACK-02 | ✓ SATISFIED | Authority/projection roles are explicit; JSON is a protocol-valid derived sink; PostgreSQL is explicitly derived but unregistered pending qualification. |
| BACK-03 | ✓ SATISFIED | Exact injected and valid registered participants remain selected through one topology registry. |
| BACK-06 | ✓ SATISFIED | Active participants expose truthful capabilities, invalid minima fail before I/O, and no placeholder backend is advertised. |
| BACK-07 | ✓ SATISFIED | Direct public schema/value put, query, update, exact conflict, partial projection, and reopen behavior passes. |

No Phase 4 requirement is orphaned; all four roadmap IDs appear in phase plan frontmatter.

### Test Quality Audit

| Test surface | Linked requirement | Assertion level | Verdict |
|---|---|---|---|
| Catalog/composition/projection suites | BACK-02/03/06/07 | Behavioral multi-step | PASS |
| JSON projection restart tests | BACK-02/07 | Behavioral state transition | PASS — apply/checkpoint interruption and reopen are exercised. |
| Metadata/PostgreSQL role tests | BACK-02/06 | Value plus construction behavior | PASS — tests no longer stop at role-label presence. |
| Cutover verifier fixtures | D-17 clean cutover | Exact value assertions | PASS — shared visitor output is checked for every supported positive/negative form. |

Plan 04-14 requirement-linked tests contain no disabled cases. Platform/capability skips in the larger owned matrix are not sole evidence for any Phase 4 requirement. Fixture writes create hostile inputs or derived state and are not circular expected-value generation.

### Anti-Patterns and Prohibition Checks

No unreferenced `TBD`, `FIXME`, or `XXX` markers were found in the final Plan 04-14 scope. The repository-root `metadata.py` draft is absent, and `src/cacheness/metadata.py` is the sole package implementation.

Non-authoritative judgment confirms the phase adds no lifecycle lock, queue, scheduler, readiness registry/event, lease, sidecar, second authority, compatibility selector, implicit migration, or universal cross-resource/topology guarantee. JSON consumes derived batches only and exposes no canonical read, delete, cleanup, reconciliation, or repair authority.

### Decision Coverage

The non-blocking GSD decision-coverage check recognizes all 18 trackable Phase 4 decisions. Behavioral evidence now supports the formerly weak D-09/D-10/D-13/D-17 projection and cutover seams.

### Deferred Items

| Item | Addressed In | Evidence |
|---|---|---|
| Optional pandas/SQL-cache installation and import qualification | Phase 8 | The exact three paths remain a separately classified non-green diagnostic and never enter the owned matrix. |
| Live PostgreSQL/S3 topology qualification | Phase 5 | Phase 4 makes no live-service claim and does not advertise a placeholder PostgreSQL sink. |
| Offline migration/rebuild execution | Phase 7 | Format/schema detection and typed non-mutating failure remain intact; normal open performs no implicit migration. |

### Human Verification Required

N/A — infrastructure/core-library phase with no user-facing visual flow. Every Phase 4 success criterion and behavior-dependent invariant has executable evidence.

### Gaps Summary

**No gaps found.** Phase goal achieved and ready to proceed. The Roadmap's unchecked Plan 04-14 marker is completion bookkeeping for the orchestrator; the plan's three commits, summary, implementation, tests, review, and verification evidence are present.

---

_Verified: 2026-09-08T08:59:08Z_
_Verifier: the agent (gsd-verifier)_
