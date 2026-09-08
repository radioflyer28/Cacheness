---
phase: 04-metadata-composition-and-topology-contracts
verified: 2026-09-08T04:40:39Z
status: gaps_found
score: 6/15 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
gaps:
  - truth: "Direct BlobStore users can define, validate, persist, update, reopen, and query application catalog metadata through the supported public lifecycle."
    status: failed
    reason: "BlobStore exposes no catalog schema/value input on put_entry or update_metadata. Normal writes hard-code the default schema and empty catalog values, so the public path cannot create a match for a declared CatalogSchema."
    artifacts:
      - path: "src/cacheness/storage/blob_store.py"
        issue: "put_entry accepts only data/key/metadata and update_metadata changes only user_metadata."
      - path: "src/cacheness/storage/lifecycle.py"
        issue: "Normal put hard-codes cacheness.default with catalog_values={} and catalog_presence=()."
      - path: "tests/test_catalog_query_contract.py"
        issue: "The passing end-to-end query test seeds catalog descriptors through private authority mutation primitives instead of BlobStore."
    missing:
      - "A public BlobStore catalog write/update input validated by CatalogSchema before staging."
      - "Canonical same-generation persistence of schema identity, values, and exact stored presence."
      - "Public put/update/query/reopen behavioral tests."
  - truth: "The selected payload participant supplies storage operations to the BlobStore engine and an injected participant is not silently replaced."
    status: failed
    reason: "Every non-memory payload participant is discarded by _materialize_authority_store in favor of GuardedHandlerIO(cache_dir). A selected FilesystemBlobBackend(base_dir=A) leaves A empty while payload bytes are written under cache_dir=B."
    artifacts:
      - path: "src/cacheness/storage/blob_store.py"
        issue: "Lines 566-573 use only InMemoryBlobBackend; every other selected payload is replaced by cache_dir-backed handler I/O."
    missing:
      - "A narrow handler/generation I/O primitive supplied by the selected filesystem participant."
      - "Construction-time rejection for payload participants that cannot supply that primitive."
  - truth: "Caller-registered backend names resolve through the same BlobStore composition root, with no overlapping hidden registry."
    status: failed
    reason: "BlobStore always invokes topology.resolve() without an application RoleRegistry, while StoreTopology creates a fresh built-in-only registry. The still-public legacy blob registry is separate and disconnected."
    artifacts:
      - path: "src/cacheness/storage/composition.py"
        issue: "StoreTopology.resolve creates a fresh RoleRegistry when none is supplied; StoreTopology carries no registry."
      - path: "src/cacheness/storage/blob_store.py"
        issue: "BlobStore.__init__ has no registry input and calls topology.resolve() directly."
      - path: "src/cacheness/storage/backends/blob_backends.py"
        issue: "The separate _blob_backend_registry and register/get/list functions remain."
      - path: "src/cacheness/__init__.py"
        issue: "Legacy blob registry functions remain publicly exported."
    missing:
      - "One application-extensible registry reachable from BlobStore construction."
      - "Removal of the disconnected legacy blob registry and exports rather than a synchronization shim."
  - truth: "Invalid topology participants fail during composition and all store-owned resources unwind exactly once."
    status: failed
    reason: "Role validation rejects only known cross-role types and None. StoreTopology(object(), object()).resolve() succeeds. Named factory results and transferred injected resources are validated before entering the ownership ledger, so validation failure can leak them; duplicate owned references can also be closed more than once."
    artifacts:
      - path: "src/cacheness/storage/composition.py"
        issue: "_validate_participant_role does not require LifecycleAuthority or a payload I/O protocol, and ownership is recorded after validation."
    missing:
      - "Runtime-checkable authority and payload participant protocols enforced before store use."
      - "An identity-deduplicated ownership guard that records newly owned objects before validation and closes them on every failure path."
  - truth: "Projection failure always preserves and reports the committed receipt, and named rebuild checks use the selected projection's own capability."
    status: failed
    reason: "ProjectionController catches only a selected exception tuple. A custom Exception escapes put_entry after authority commit while the entry remains present and no BlobReceipt is returned. Controllers also receive aggregate topology projection capabilities, so one incapable sink can incorrectly block rebuilding a capable named sink."
    artifacts:
      - path: "src/cacheness/storage/projections.py"
        issue: "best_effort and refresh catch only CacheError/OSError/RuntimeError/TypeError/ValueError, not arbitrary external sink Exception subclasses."
      - path: "src/cacheness/storage/blob_store.py"
        issue: "Each ProjectionController receives aggregate self.capabilities rather than the selected sink's ParticipantCapabilities."
    missing:
      - "Catch Exception (not BaseException) at the external projection boundary and translate it to dirty/committed-partial evidence carrying the exact receipt."
      - "Per-sink capability checks for named rebuild operations, while retaining aggregate topology reporting for whole-store guarantees."
  - truth: "Portable catalog cursor inputs are bounded before decoding and authority dispatch."
    status: failed
    reason: "Cursor validation requires only a non-empty string. CatalogCursor.inspect base64-decodes the complete caller-controlled value before any encoded or decoded size bound, and cursor string fields have no finite length bound."
    artifacts:
      - path: "src/cacheness/storage/catalog.py"
        issue: "Lines 523-530 decode and parse an unbounded cursor."
    missing:
      - "A maximum encoded cursor length checked before base64 decode."
      - "A decoded-byte bound and finite bounds for every cursor string field."
      - "Over-limit tests proving authority dispatch is never reached."
  - truth: "Retained Phase 4 regressions and the complete supported-runtime suite are green without compatibility shims or exclusions."
    status: failed
    reason: "The independent full Python 3.13 run stops with seven collection errors. Four modules still import intentionally removed metadata-authority symbols, three SQL-cache modules require absent pandas, and the focused retained integrity/lifecycle run has six stale-test failures. Python 3.11 need not be rerun to falsify the conjunctive two-version criterion."
    artifacts:
      - path: "tests/test_cached_query_meta.py"
        issue: "Imports removed CacheEntry/CachedMetadataBackend/SqliteBackend symbols."
      - path: "tests/test_phase3_gap_acceptance.py"
        issue: "Imports removed SqliteBackend."
      - path: "tests/test_sqlite_metadata_bootstrap_atomicity.py"
        issue: "Imports removed Base/SqliteBackend."
      - path: "tests/test_stored_compatibility.py"
        issue: "Imports removed JsonBackend/SqliteBackend."
      - path: "tests/test_blob_store_integrity.py"
        issue: "Three tests call the removed private _manifest_key helper."
      - path: "tests/test_unified_cache_lifecycle_authority.py"
        issue: "Three tests retain obsolete _lock, conflict-propagation, or authored-relative-path representation assertions."
    missing:
      - "Retire or rewrite development-only compatibility tests against the clean public contract without restoring old exports."
      - "Repair retained tests at current private test seams while preserving the security/concurrency invariant."
      - "Keep optional pandas/SQL-cache qualification explicit; Phase 8 may finish packaging-extra evidence, but Phase 4's own no-exclusion suite criterion remains failed until collection is clean."
deferred:
  - truth: "Default-environment SQL-cache modules collect without pandas."
    addressed_in: "Phase 8"
    evidence: "Phase 8 success criteria require each advertised optional dependency group to install and import independently. This does not defer the separate Phase 4 failures caused by stale removed-symbol imports."
---

# Phase 4: Metadata Composition and Topology Contracts Verification Report

**Phase Goal:** Users can customize BlobStore catalog metadata without implementing lifecycle sequencing, select advertised metadata backends through one composition root, and receive only guarantees supported by the topology.
**Verified:** 2026-09-08T04:40:39Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

The phase establishes useful catalog/schema/query values, explicit metadata roles, signed canonical scans, version boundaries, and a BlobStore-backed UnifiedCache seam. It does not yet achieve the phase goal end to end. Most importantly, catalog customization cannot enter the public BlobStore lifecycle, the selected filesystem payload participant does not power payload I/O, registered application participants cannot reach BlobStore, and composition/projection input boundaries do not fail closed as specified.

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | JSON/PostgreSQL are projection-only; memory/SQLite are authorities; adapters do not duplicate lifecycle sequencing. | ✓ VERIFIED | `resolve_metadata_role`, `ProjectionRole`, `LifecycleAuthority`, and the 185-test focused matrix preserve one authority boundary. |
| 2 | One composition root preserves exact injected and registered participants. | ✗ FAILED | Direct repro: an application `RoleRegistry` registration cannot be supplied to `BlobStore`; a selected filesystem participant is retained as an attribute but not used for I/O. |
| 3 | Users inspect the actual durability, sharing, CAS, streaming, and listing capabilities of the active pair. | ✗ FAILED | Capability objects exist, but the reported filesystem participant is not the participant actually performing payload I/O. |
| 4 | Invalid topology claims/participants fail during construction, with owned-resource unwind. | ✗ FAILED | `StoreTopology(object(), object()).resolve()` succeeds; validation is nominal rather than protocol-complete and occurs before ownership is recorded. |
| 5 | Direct users define, validate, write, update, reopen, and query declared catalog metadata. | ✗ FAILED | Public put/update accepts only `metadata`; a put with `metadata={"rank": 7}` stores default schema/empty catalog and query returns zero entries. |
| 6 | Projection APIs preserve committed receipts and enforce per-projection failure/rebuild behavior. | ✗ FAILED | A custom external-sink `Exception` escapes after commit without a receipt; aggregate capabilities are used for each named controller. |
| 7 | Native schema values distinguish stored absence/default/null and preserve bounded opaque fields. | ✓ VERIFIED | `CatalogSchema` behavior and exact scalar validation pass in the focused matrix. |
| 8 | Format/schema versions are independent and unsupported layouts fail typed without mutation. | ✓ VERIFIED | Format 2 and independent dimensions are explicit; non-mutating layout tests pass. |
| 9 | Canonical queries are authenticated, revision-bound, deterministic, and work-capped for seeded descriptors. | ✓ VERIFIED | Memory/SQLite dense, sparse, tampered, and stale-cursor behavioral tests pass through `BlobStore.query_catalog`. |
| 10 | Cursor input itself is bounded before decode/dispatch. | ✗ FAILED | A two-million-character cursor is fully decoded before being rejected; no encoded/decoded/string-field cap exists. |
| 11 | UnifiedCache remains a policy facade over one internal BlobStore engine without a Phase 6 policy redesign. | ✓ VERIFIED | `core.py` composes `StoreTopology`/`BlobStore`; focused configuration/local-workflow/decorator tests pass. |
| 12 | Superseded factories, registries, filters, result shapes, and runtime ORM authority hooks are absent. | ✗ FAILED | Most retired metadata surfaces are gone, but the public/disconnected blob backend registry and exports remain. |
| 13 | Retained lifecycle, integrity, containment, and concurrency regressions are executable on the clean surface. | ✗ FAILED | Focused integrity/lifecycle run: six failures from stale test assumptions. |
| 14 | Complete Python 3.11 and 3.13 suites pass without exclusions; external-service/platform non-claims stay honest. | ✗ FAILED | Independent Python 3.13 full run stops with seven collection errors. The documented PostgreSQL/S3/Windows non-claims are honest. |
| 15 | Phase 4 introduces no new Ruff findings relative to its frozen baseline. | ✓ VERIFIED | `uv run --frozen python tools/verify_phase4_ruff_delta.py` passes. |

**Score:** 6/15 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/catalog.py` | Native schema/query/cursor contract | ⚠️ PARTIAL | Substantive and wired, but cursor decode is unbounded and public writes cannot populate declared catalog values. |
| `src/cacheness/storage/composition.py` | One role-aware topology resolver | ✗ FAILED | Substantive, but invalid objects resolve and application registration is not wired to BlobStore. |
| `src/cacheness/storage/blob_store.py` | Sole public payload/catalog facade | ✗ FAILED | Substantive and used, but declared catalog input is absent and filesystem payload selection is ignored. |
| `src/cacheness/storage/projections.py` | Derived-only bounded projection controller | ⚠️ PARTIAL | Pull/checkpoint behavior exists; unexpected sink exceptions and per-sink rebuild capability are mishandled. |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | Format-2 local authority and canonical scan | ✓ VERIFIED | Exists, substantive, wired through StoreTopology, and covered by focused behavior tests. |
| `src/cacheness/storage/memory_lifecycle_authority.py` | Same-process authority and canonical scan | ✓ VERIFIED | Exists, substantive, wired, and covered by memory behavior tests. |
| `src/cacheness/storage/read_contract.py` | Frozen BlobReceipt/BlobEntry results | ✓ VERIFIED | Receipt is frozen and carries named projection outcomes. |
| `docs/CATALOG_AND_TOPOLOGY.md` | Published consistency/topology boundaries | ⚠️ PARTIAL | Correctly rejects extra authority and universal guarantees, but claims pre-use role validation that the implementation does not enforce and omits a usable public catalog-write workflow. |
| `04-VALIDATION.md` | Honest interpreter/release evidence | ⚠️ PARTIAL | Honestly records failures, but its sign-off remains incomplete and the required full-suite gate is not green. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `BlobStore.put_entry/update_metadata` | `CatalogSchema` / canonical manifest catalog fields | Validated public catalog lifecycle | ✗ NOT_WIRED | Normal lifecycle hard-codes the default empty catalog. |
| `StoreTopology` | application `RoleRegistry` | High-level BlobStore construction | ✗ NOT_WIRED | Registry works only when callers invoke `topology.resolve(registry)` themselves. |
| selected filesystem payload | `AuthorityLifecycleEngine` handler I/O | Active participant supplies payload storage | ✗ NOT_WIRED | BlobStore substitutes `GuardedHandlerIO(cache_dir)`. |
| `BlobStore.query_catalog` | lifecycle authority | Bounded canonical `catalog_page` | ✓ WIRED | Focused query behavior passes for memory/SQLite seeded descriptors. |
| `BlobStore` | `ProjectionController` | post-commit best effort / explicit refresh | ⚠️ PARTIAL | Wiring exists, but exception translation is incomplete and capabilities are aggregate. |
| `UnifiedCache` | `BlobStore` | narrow internal composition | ✓ WIRED | Policy facade constructs and delegates to one BlobStore. |
| `cacheness.__init__` | storage barrels | clean public surface | ⚠️ PARTIAL | Clean catalog exports exist, but legacy blob registry exports survive. |

The automated PLAN link checker reported 10/11 declared links matched. The unmatched Plan 04-01 fake-participant regex is a naming mismatch (`_Payload`/`_Authority`, not `Fake*`) rather than an additional runtime gap. Manual wiring checks found the four material failures above that presence-based pattern matching missed.

### Data-Flow Trace (Level 4)

| Artifact | Data variable | Source | Produces real data | Status |
|---|---|---|---|---|
| `BlobStore.query_catalog` | `CatalogPage.entries` | Authenticated descriptors from authority `catalog_page` | Yes, when descriptors contain catalog values | ✓ FLOWING |
| `BlobStore.put_entry` | manifest `catalog_values` | Hard-coded `{}` rather than caller schema/value input | No | ✗ DISCONNECTED |
| `ProjectionController` | derived batches/checkpoints | Canonical `BlobStore.query_catalog` pages | Yes for supported error paths | ⚠️ PARTIAL |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Focused Phase 4 contract matrix | `uv run --frozen pytest -q` over the nine Plan 04-08 focused modules | 185 passed | ✓ PASS |
| Public catalog round trip | Direct memory BlobStore put with `metadata={"rank": 7}` then declared `rank == 7` query | 0 entries; manifest uses `cacheness.default` and `{}` | ✗ FAIL |
| Selected filesystem participant | Direct filesystem participant at A with `cache_dir=B`, then put | A remains empty; payload appears under B | ✗ FAIL |
| Application registered name | Register names in application RoleRegistry, then construct BlobStore by those names | `CompositionValidationError`: name not registered in fresh internal registry | ✗ FAIL |
| Unexpected projection failure | Custom `Exception` from projection sink after put | Custom exception escapes; committed entry remains present; no receipt returned | ✗ FAIL |
| Invalid participant validation | `StoreTopology(object(), object()).resolve()` | Resolution succeeds | ✗ FAIL |
| Retained integrity/lifecycle regressions | pytest over `test_blob_store_integrity.py` and `test_unified_cache_lifecycle_authority.py` | 6 failed | ✗ FAIL |
| Complete default Python 3.13 suite | `uv run --frozen pytest -q -o log_cli=false` | 7 collection errors | ✗ FAIL |
| Ruff delta | `uv run --frozen python tools/verify_phase4_ruff_delta.py` | passed | ✓ PASS |

### Probe Execution

No Phase 4 probes were declared and no `scripts/**/tests/probe-*.sh` files exist. Probe execution is not applicable.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| BACK-02 | 04-01 through 04-08 | Explicit authority/projection roles through one composition contract | ✓ SATISFIED | Roles are explicit; JSON/PostgreSQL remain derived and no projection authorizes lifecycle or canonical query completeness. |
| BACK-03 | 04-01, 03, 06, 07, 08 | Injected and registered implementations remain selected | ✗ BLOCKED | Filesystem injection is not used for I/O; application registered names cannot reach BlobStore. |
| BACK-06 | 04-01, 03 through 08 | Truthful capabilities and construction-time rejection | ✗ BLOCKED | Report objects exist, but invalid participants resolve and reported payload capabilities can describe a participant the engine ignores. |
| BACK-07 | 04-01 through 08 | Direct users customize/query/update authoritative catalog metadata | ✗ BLOCKED | Public lifecycle cannot write or update declared catalog values; projection/cursor fail-closed boundaries are incomplete. |

No Phase 4 requirements are orphaned: all four IDs appear in plan frontmatter and in the roadmap mapping.

### Review Finding Assessment

| Finding | Verification verdict | Evidence |
|---|---|---|
| CR-01 public catalog write/update missing | CONFIRMED BLOCKER | Direct public round trip stores empty default catalog and returns no query match. |
| CR-02 filesystem participant ignored | CONFIRMED BLOCKER | Direct A/B root repro leaves selected backend root A empty. |
| CR-03 application registrations unreachable | CONFIRMED BLOCKER | High-level construction cannot receive the application registry; legacy registry is disconnected. |
| CR-04 unexpected projection exceptions escape | CONFIRMED BLOCKER | Custom `Exception` escapes after commit while `get_entry_info` confirms the entry exists. |
| CR-05 cursor decode unbounded | CONFIRMED BLOCKER | Two-million-character input reaches base64 decode; code has no pre-decode/decoded bound. |
| CR-06 invalid participants/resource unwind | CONFIRMED BLOCKER | Arbitrary objects resolve; ownership is appended only after validation/construct. |
| WR-01 aggregate rebuild capability used per sink | CONFIRMED WARNING | `BlobStore` passes `self.capabilities` to every controller; aggregate uses `all(...)`. |
| WR-02 stale retained tests | CONFIRMED WARNING | Independent focused run produced exactly six failures. |

None of these findings requires or justifies a new lifecycle lock, queue, readiness registry, second authority, cross-resource ACID mechanism, or universal-success guarantee. The repair boundary is the existing composition/catalog/projection seam under ADR 0001.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|---|---|---:|---:|---|---|---|
| `tests/test_catalog_schema.py` | BACK-07 | Yes | 0 | No | Value/behavioral | PASS for value objects and format boundaries; does not prove public catalog writes. |
| `tests/test_catalog_query_contract.py` | BACK-07 | Yes | 0 | No | Behavioral | INSUFFICIENT for end-to-end BACK-07 because catalog entries are seeded with private authority primitives. |
| `tests/test_blob_store_composition.py` | BACK-03/BACK-06 | Yes | 0 | No | Behavioral | INSUFFICIENT for registered-name high-level composition because it calls `topology.resolve(registry)` directly, not BlobStore. |
| `tests/test_catalog_projection.py` | BACK-02/BACK-07 | Yes | 0 | No | Behavioral | PARTIAL; covers OSError but not an arbitrary external sink exception or mixed per-sink rebuild capability. |
| `tests/test_blob_store_integrity.py` | BACK-03/BACK-07 | Yes | 3 platform-conditional | No | Behavioral | FAIL: three active cases use a retired private helper. Platform skips are not sole requirement evidence. |
| `tests/test_filesystem_containment.py` | BACK-03/BACK-07 | Yes | platform/capability-conditional | No | Behavioral | PASS for available POSIX paths; skipped platform variants have other active contract evidence. |

**Disabled tests on requirements:** platform/capability skips exist, but none is the only evidence for a Phase 4 requirement; no test-quality blocker from skips.
**Circular patterns detected:** 0.
**Insufficient assertions/wiring:** 3 focused suites omit the public or adversarial path they claim to establish.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `src/cacheness/storage/lifecycle.py` | 252-256 | Hard-coded empty canonical catalog on every public write | 🛑 Blocker | Makes the catalog feature unreachable. |
| `src/cacheness/storage/blob_store.py` | 566-573 | Selected participant retained but bypassed | 🛑 Blocker | Composition identity/capabilities disagree with actual I/O. |
| `src/cacheness/storage/backends/blob_backends.py` | 450+ | Disconnected legacy registry retained | 🛑 Blocker | Violates the single-root clean cutover and misleads extension users. |
| `src/cacheness/storage/projections.py` | 49-55 | Hand-picked external error allowlist | 🛑 Blocker | A committed write can appear failed without its receipt. |
| `src/cacheness/storage/catalog.py` | 523-530 | Unbounded decode before validation | 🛑 Blocker | Caller-controlled allocation/parsing at the public query boundary. |

No unreferenced `TBD`, `FIXME`, or `XXX` markers were found in the Phase 4 implementation/test scope. No new coordinator/queue/sidecar or cross-resource atomicity claim was found.

### Prohibition Checks

- **No new lifecycle coordinator/lock/queue/sidecar or universal cross-resource guarantee:** non-authoritative judgment passes. Phase 4 uses the existing BlobStore admission/lifecycle authority and synchronous derived pulls.
- **No projection authorizes canonical reads/query completeness/delete/repair:** non-authoritative judgment passes; projections consume canonical pages and are not consulted for lifecycle authority.
- **Unsupported layouts are not implicitly reopened or mutated:** behavioral tests pass.
- **No hidden/legacy registry or compatibility path remains:** fails because the old blob backend registry and public exports remain. The fix is deletion/consolidation, not a compatibility shim.
- **No manufactured release matrix:** passes as a documentation honesty check; `04-VALIDATION.md` records failures rather than exclusions. The matrix itself is still red.

### Decision Coverage

All 18 trackable CONTEXT.md decisions are recognized in shipped artifacts by the non-blocking GSD decision-coverage check. This does not override the behavioral failures above: D-10, D-11, D-15, D-16, and D-17 are mentioned in artifacts but are not fully realized by the public runtime path.

### Human Verification Required

N/A — infrastructure/core-library phase with no user-facing visual or external-service flow. All phase-goal failures were reproduced programmatically; no present-but-behavior-unverified truth remains.

### Deferred Items

| Item | Addressed In | Evidence |
|---|---|---|
| Optional pandas/SQL-cache installation and import qualification | Phase 8 | Phase 8 owns independent optional-dependency installation/import release evidence. |

The catalog/composition/projection blockers are not deferred. Phase 5 depends on correct participant wiring, and Phases 6/7 depend on the public catalog/composition foundation; pushing these gaps forward would make later qualification test the wrong engine.

### Gaps Summary

Phase 4 is not ready to close. Seven grouped gaps block the goal: public catalog values are unreachable, selected filesystem storage is bypassed, application registrations are disconnected, invalid participants/resources are not safely validated, projection failures/rebuild capabilities are misclassified, cursor inputs are not bounded, and the retained/full-suite test surface is red. These are finite seam repairs within the existing BlobStore/authority model. ADR 0001 specifically rules out solving them with more coordination machinery or stronger universal concurrency promises.

---

_Verified: 2026-09-08T04:40:39Z_
_Verifier: the agent (gsd-verifier)_
