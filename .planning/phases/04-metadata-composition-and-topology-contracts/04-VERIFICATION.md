---
phase: 04-metadata-composition-and-topology-contracts
verified: 2026-09-08T08:00:47Z
status: gaps_found
score: 14/16 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
re_verification:
  previous_status: gaps_found
  previous_score: 6/15
  gaps_closed:
    - "Public BlobStore catalog write, update, query, and reopen are wired through the canonical descriptor lifecycle."
    - "The selected filesystem or memory payload participant supplies the engine's actual generation I/O."
    - "Application registrations reach BlobStore through the topology-owned RoleRegistry and the legacy blob registry is absent."
    - "Structural participant validation and identity-deduplicated owned-resource unwind fail closed."
    - "Ordinary projection failures preserve the committed receipt and named rebuild uses per-sink capabilities."
    - "Catalog cursors are bounded before decode, parsing, HMAC, manifest loading, or authority dispatch."
    - "Retained regression consumers pass at stable paths on the clean public surface."
  gaps_remaining:
    - "Advertised built-in JSON/PostgreSQL projection registrations do not satisfy the topology's ProjectionSink contract."
    - "The cutover AST audit misses common root and submodule-alias uses of retired public APIs."
  regressions:
    - "The previously verified explicit metadata-role truth is no longer sufficient: role labels resolve, but the registry's own advertised projection objects cannot be resolved as topology participants."
gaps:
  - truth: "Every advertised built-in metadata projection resolves through the one StoreTopology composition path as a structurally valid derived-only participant."
    status: failed
    reason: "RoleRegistry advertises json and postgresql projection factories, but JsonProjection and PostgresBackend do not implement ProjectionSink. JSON construction reaches structural validation and fails; PostgreSQL fails earlier on unavailable optional dependencies in the locked base environment and would fail the same protocol check if constructed."
    artifacts:
      - path: "src/cacheness/storage/composition.py"
        issue: "Registers json/postgresql as built-ins at lines 392-402, then rejects their factory results at the ProjectionSink check around lines 666-672."
      - path: "src/cacheness/metadata.py"
        issue: "JsonProjection is a placeholder with no apply_projection_batch/save_projection_checkpoint/load_projection_checkpoint methods."
      - path: "src/cacheness/storage/backends/postgresql_backend.py"
        issue: "PostgresBackend is explicitly a placeholder and does not implement the derived sink protocol."
      - path: "tests/test_metadata_role_contract.py"
        issue: "The built-in test checks only RoleRegistration.role and never constructs or resolves the advertised participant through StoreTopology."
    missing:
      - "Either implement the narrow derived ProjectionSink contract for a Phase 4-qualified JSON projection or stop advertising json as constructible."
      - "Remove/defer the PostgreSQL built-in registration until its projection contract is supplied in Phase 5, or implement and qualify the narrow derived contract now."
      - "Add a topology-level test for every advertised built-in projection; do not weaken structural validation or make a projection authoritative."
  - truth: "The checked-in executable-consumer audit rejects retired public API imports and attribute use across every supported import spelling it claims to cover."
    status: failed
    reason: "Direct AST probes show false negatives for a retired symbol imported from the package root, an aliased cacheness root module, and an aliased blob_backends submodule. The audit nevertheless reports success, so the release-evidence artifact does not enforce its Plan 04-13 contract."
    artifacts:
      - path: "tools/verify_phase4_cutover.py"
        issue: "Root cacheness imports/aliases are not tracked for blob selectors, and an unreachable later elif prevents binding blob_backends aliases imported from cacheness.storage.backends."
      - path: ".planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md"
        issue: "Claims the executable tree audit rejects retired imports, bound aliases, and package attributes more broadly than the visitor actually detects."
    missing:
      - "Track direct retired imports from cacheness and aliases created by import cacheness as name."
      - "Bind blob_backends submodule aliases inside the reachable backend ImportFrom branch."
      - "Add table-driven tests for direct imports, root aliases, submodule aliases, bound aliases, star imports, and string-only negative assertions."
deferred:
  - truth: "The three pandas-dependent SQL-cache modules collect and execute in a qualified optional-dependency installation."
    addressed_in: "Phase 8"
    evidence: "Phase 8 success criteria require each advertised optional dependency group to install and import independently; Phase 4 keeps these exact paths diagnostic-only and explicitly non-green."
  - truth: "Live PostgreSQL and S3 lifecycle/topology behavior is qualified."
    addressed_in: "Phase 5"
    evidence: "Phase 5 owns real-service and supported-pair qualification. This does not justify advertising a structurally invalid Phase 4 built-in participant."
---

# Phase 4: Metadata Composition and Topology Contracts Verification Report

**Phase Goal:** Users can customize BlobStore catalog metadata without implementing lifecycle sequencing, select advertised metadata backends through one composition root, and receive only guarantees supported by the topology.
**Verified:** 2026-09-08T08:00:47Z
**Status:** gaps_found
**Re-verification:** Yes — after gap-closure Plans 04-09 through 04-13

## Goal Achievement

The original six blockers and two warnings are closed in the actual code. Public catalog values now flow through the canonical signed descriptor; the selected payload participant performs storage I/O; application registrations reach one topology-owned registry; construction/unwind, projection failure, per-sink rebuild, and cursor limits fail closed; and the retained two-interpreter matrix is green.

Phase 4 still misses its complete composition and release-evidence contracts. The sole registry advertises JSON and PostgreSQL projections that its own structural validator rejects, and the checked-in cutover audit misses ordinary retired-API import forms. Neither gap calls for a lock, queue, retry coordinator, second authority, or stronger topology guarantee.

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | JSON/PostgreSQL are explicit derived-only roles whose advertised registrations are valid for their declared Phase 4 capability tier. | ✗ FAILED | Role labels resolve, but direct topology resolution of built-in `json` raises `CompositionValidationError: A projection participant must satisfy ProjectionSink`; `postgresql` is registered despite being a deferred placeholder. |
| 2 | One composition root preserves exact injected and registered participants. | ✓ VERIFIED | High-level custom named payload/authority/projection tests pass; `StoreTopology.role_registry` is the only registry path. |
| 3 | Capability reports describe the participants that actually supply payload and authority operations. | ✓ VERIFIED | Selected-root A/B tests pass and `BlobStore` obtains handler I/O from `self.payload_backend.materialize_handler_io()`. |
| 4 | Invalid topology claims/participants fail during construction and owned resources unwind once by identity. | ✓ VERIFIED | Structural role and reverse close-once tests pass for factory-, transfer-, caller-, and duplicate-owned cases. |
| 5 | Direct users define, validate, write, update, reopen, and query declared catalog metadata. | ✓ VERIFIED | Public memory/SQLite put-update-query-reopen tests pass; validation precedes authority/payload calls. |
| 6 | Projection APIs preserve committed receipts and enforce per-projection failure/rebuild behavior. | ✓ VERIFIED | Arbitrary `Exception`, `BaseException`, exact receipt, dirty outcome, committed-partial, and mixed-capability named-rebuild tests pass. |
| 7 | Native schema values distinguish stored absence/default/null and preserve bounded opaque fields. | ✓ VERIFIED | Catalog schema/value behavioral tests pass on the public lifecycle. |
| 8 | Format/schema versions remain independent and unsupported layouts fail typed without mutation. | ✓ VERIFIED | Current-format reopen and corrupted/obsolete/foreign layout tests pass without implicit migration. |
| 9 | Canonical queries are authenticated, revision-bound, deterministic, complete for the authority snapshot, and work-capped. | ✓ VERIFIED | Memory/SQLite dense, sparse, tampered, stale-cursor, and public round-trip tests pass. |
| 10 | Cursor inputs are bounded before decoding, parsing, HMAC, manifest loading, and authority dispatch. | ✓ VERIFIED | Boundary and no-dispatch tests pass; encoded, decoded, field, signature, and exact-shape limits are present. |
| 11 | UnifiedCache remains a policy facade over one internal BlobStore engine without a Phase 6 redesign. | ✓ VERIFIED | Core and lifecycle-authority regression modules pass through the selected BlobStore topology. |
| 12 | Superseded factories, registries, result shapes, and runtime ORM authority hooks are absent from the production/public tree. | ✓ VERIFIED | Source scan finds no live retired production export or selector; public absence tests pass. |
| 13 | Retained lifecycle, integrity, containment, concurrency, initialization, and cleanup regressions execute on current seams. | ✓ VERIFIED | Focused prior-gap selection passes with 184 passed and 2 optional-dataframe skips; the owned matrix includes all stable paths. |
| 14 | The Phase 4-owned matrix runs unfiltered on Python 3.11 and 3.13 while later optional/service/platform claims remain explicit non-claims. | ✓ VERIFIED | Independent runs: 588 passed, 6 skipped on CPython 3.11.16 and CPython 3.13.15. Full-tree pandas collection remains explicitly deferred/non-green. |
| 15 | Phase 4 adds no Ruff findings relative to its frozen baseline. | ✓ VERIFIED | `verify_phase4_ruff_delta.py` passes on Python 3.13; the recorded two-interpreter gate is consistent. |
| 16 | The executable-consumer audit detects every supported retired public API import/alias form it claims to reject. | ✗ FAILED | Three direct AST probes return no findings: root direct import, aliased root module, and aliased backend submodule. |

**Score:** 14/16 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/catalog.py` | Native schema/query/cursor contract | ✓ VERIFIED | Substantive, wired, bounded, and exercised through public BlobStore behavior. |
| `src/cacheness/storage/composition.py` | One truthful role-aware topology resolver | ✗ FAILED | Custom and authority/payload roles are substantive and wired, but two advertised built-in projections fail the required structural contract. |
| `src/cacheness/storage/blob_store.py` | Sole public payload/catalog facade | ✓ VERIFIED | Public catalog, selected payload, authority, projection-controller, and UnifiedCache links are live. |
| `src/cacheness/storage/projections.py` | Derived-only bounded projection controller | ✓ VERIFIED | Ordinary external failures preserve receipts; named rebuild is capability-local. |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | Format-2 local canonical authority | ✓ VERIFIED | Substantive, wired, and covered by lifecycle/catalog tests. |
| `src/cacheness/storage/memory_lifecycle_authority.py` | Same-process canonical authority | ✓ VERIFIED | Substantive, wired, and behaviorally exercised. |
| `src/cacheness/storage/read_contract.py` | Frozen BlobReceipt result | ✓ VERIFIED | Immutable receipts carry exact expectations and named projection outcomes. |
| `tools/verify_phase4_cutover.py` | Honest consumer audit and bounded release matrix | ✗ FAILED | Matrix/set parsing works, but the AST visitor has reproducible retired-use false negatives. |
| `docs/CATALOG_AND_TOPOLOGY.md` | Published consistency/topology boundaries | ⚠ PARTIAL | Correct about authority and ADR limits, but the registry implementation cannot realize the implied JSON derived participant. |
| `04-VALIDATION.md` | Reproducible release evidence | ⚠ PARTIAL | Interpreter/matrix evidence is reproducible; its consumer-audit capability statement overclaims the current visitor. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `BlobStore.put_entry/update_catalog` | `CatalogSchema` and authority descriptor transaction | Validated catalog values/exact expectation | ✓ WIRED | Public lifecycle tests prove commit, patch, query, stale conflict, and reopen. |
| selected payload participant | `AuthorityLifecycleEngine` handler I/O | `materialize_handler_io()` | ✓ WIRED | Distinct-root test proves no cache-dir substitution. |
| `StoreTopology` | application `RoleRegistry` | topology-owned registry resolved once | ✓ WIRED | Custom names reach high-level BlobStore construction. |
| `RoleRegistry` built-in projections | `ProjectionSink` | factory construction plus structural role validation | ✗ NOT WIRED | JSON/PostgreSQL factory results omit all three required sink methods. |
| `BlobStore` | `ProjectionController` | exact receipt and per-sink capabilities | ✓ WIRED | Post-commit behavior tests pass. |
| `tools/verify_phase4_cutover.py` | executable consumer tree | AST import/alias resolution | ⚠ PARTIAL | Direct backend imports are caught; root and submodule aliases are not. |
| `UnifiedCache` | `BlobStore` | narrow internal topology composition | ✓ WIRED | Policy facade regression modules pass. |

### Data-Flow Trace

| Artifact | Data | Source | Status |
|---|---|---|---|
| `BlobStore.query_catalog` | `CatalogPage.entries` | Authenticated authority descriptors written by public catalog lifecycle | ✓ FLOWING |
| selected filesystem/memory payload | handler generation bytes | Exact composed payload participant | ✓ FLOWING |
| custom `ProjectionController` | bounded batches/checkpoints | Canonical `query_catalog` pages after authority commit | ✓ FLOWING |
| built-in JSON/PostgreSQL projection | derived batches/checkpoints | No structurally valid sink exists | ✗ DISCONNECTED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Prior-gap focused contracts | `uv run --frozen pytest -q` over ten catalog/composition/projection/integrity/lifecycle modules | 184 passed, 2 optional dataframe skips | ✓ PASS |
| Python 3.11 owned matrix | `uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all` | 588 passed, 6 skipped; deferred pandas diagnostic classified non-green | ✓ PASS |
| Python 3.13 owned matrix | `uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all` | 588 passed, 6 skipped; deferred pandas diagnostic classified non-green | ✓ PASS |
| Advertised JSON projection | Resolve memory/memory topology with named `json` projection | `CompositionValidationError`: projection does not satisfy `ProjectionSink` | ✗ FAIL |
| Advertised PostgreSQL projection | Resolve memory/memory topology with named `postgresql` projection | `ImportError` in locked base environment; implementation also lacks `ProjectionSink` methods | ✗ FAIL |
| AST detector adversarial probes | Parse four retired-selector snippets with `_RetiredConsumerVisitor` | Only direct backend import caught; 3/4 missed | ✗ FAIL |
| Current executable tree audit | `uv run --frozen python tools/verify_phase4_cutover.py --audit` | Reports pass | ⚠ INSUFFICIENT |
| Ruff delta | `uv run --frozen python tools/verify_phase4_ruff_delta.py` | Pass | ✓ PASS |

### Probe Execution

No Phase 4 probe scripts are declared or present. The checked-in cutover verifier was executed independently as release-evidence tooling.

### Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| BACK-02 | ✗ BLOCKED | Role classification is explicit and projections remain derived-only, but the registry advertises JSON/PostgreSQL projection implementations that cannot satisfy the composition contract. |
| BACK-03 | ✓ SATISFIED | Injected and application-registered valid participants remain exact and selected through one registry. |
| BACK-06 | ✓ SATISFIED | Valid active participants expose truthful capabilities; impossible minima and structurally invalid participants fail before I/O. The advertised built-in inconsistency is separately blocking BACK-02/composition completeness. |
| BACK-07 | ✓ SATISFIED | Direct public schema/value put, query, update, exact conflict, partial projection, and reopen behavior passes. |

No Phase 4 requirement is orphaned; all four roadmap IDs appear in plan frontmatter.

### Prior Finding Re-Test

| Prior finding | Verdict | Evidence |
|---|---|---|
| Public catalog write/update absent | CLOSED | Public put/update/query/reopen and no-dispatch validation tests pass. |
| Selected filesystem payload ignored | CLOSED | Selected-root A/B storage and delete tests pass. |
| Application registrations unreachable | CLOSED | Named custom participants reach BlobStore through topology registry. |
| Unexpected projection exceptions lose receipt | CLOSED | `Exception` translation and `BaseException` propagation tests pass. |
| Cursor input unbounded | CLOSED | Finite boundary/no-dispatch tests pass. |
| Invalid participant/resource unwind | CLOSED | Protocol and identity-ledger tests pass. |
| Aggregate rebuild capability used per sink | CLOSED | Mixed-capability named rebuild test passes. |
| Stale retained tests | CLOSED | Stable modules pass in both owned matrices. |

### Test Quality Audit

| Test surface | Linked requirement | Assertion level | Verdict |
|---|---|---|---|
| Catalog/composition/projection focused suites | BACK-02/03/06/07 | Behavioral multi-step | PASS for the original gaps. |
| `tests/test_metadata_role_contract.py` | BACK-02 | Registration metadata only | INSUFFICIENT: verifies that `json` has role text `projection`, not that its factory result satisfies or resolves through that role. |
| `tests/test_postgresql_backend.py` | BACK-02 | Registration/optional-error status | INSUFFICIENT: never performs topology structural validation of the advertised participant. |
| `tools/verify_phase4_cutover.py --audit` | D-17 release evidence | Current-tree status only | BLOCKER: no detector self-tests; adversarial retired-import forms are false negatives. |

Platform/capability skips are not sole evidence for any Phase 4 requirement. Test fixture writes construct hostile/corrupt inputs and are not circular expected-value generation.

### Anti-Patterns and Workspace Findings

| File | Pattern | Severity | Impact |
|---|---|---|---|
| `src/cacheness/storage/composition.py` | Advertised built-in factories contradict enforced structural protocol | 🛑 Blocker | The one composition root cannot construct its own JSON/PostgreSQL projection names. |
| `tools/verify_phase4_cutover.py` | Incomplete alias/import tracking plus unreachable `elif` | 🛑 Blocker | A green release audit can miss executable use of retired APIs. |
| repository-root `metadata.py` | Untracked near-copy of `src/cacheness/metadata.py` created just before the intended package module | ⚠ Warning | `import metadata` can resolve an accidental draft and the file could be committed unintentionally; remove or archive after confirming no human work depends on it. |

No unreferenced `TBD`, `FIXME`, or `XXX` marker was found in the Phase 4 implementation scope. No new lock, queue, readiness mechanism, lease, sidecar, second authority, or universal-progress/cross-resource-ACID claim was found.

### Decision Coverage

The non-blocking decision-coverage gate recognizes all 18 trackable Phase 4 decisions. D-09/D-10/D-13/D-17 are mentioned in artifacts but the built-in projection and audit failures above show that mention alone is not behavioral fulfillment.

### Human Verification Required

N/A — infrastructure/core-library phase with no user-facing visual flow. All relevant behavior, including the two remaining failures, is reproducible programmatically.

### Gaps Summary

Two finite gaps block Phase 4:

1. Make advertised built-in projection registrations honest. Qualify a real derived-only JSON sink or stop registering it; defer/remove PostgreSQL registration until its Phase 5 participant exists. Keep the strict protocol and single authority.
2. Repair and test the AST visitor's root/direct/submodule alias tracking so the executable-consumer audit enforces its stated clean-cutover boundary.

The untracked root `metadata.py` should be removed or archived as workspace cleanup, but it does not justify compatibility code and is not a lifecycle guarantee gap. The remaining work is bounded composition/evidence repair, not another concurrency cycle.

---

_Verified: 2026-09-08T08:00:47Z_
_Verifier: the agent (gsd-verifier)_
