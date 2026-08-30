---
phase: 02-canonical-storage-and-integrity-contract
status: complete
audited: 2026-08-30
plans: [02-01, 02-02, 02-03, 02-04, 02-05, 02-06, 02-07]
---

# Phase 2 Multi-Source Coverage Audit

All in-scope items from the roadmap goal, requirements, research, context decisions, pattern map, and validation contract are covered. Deferred Phase 3/4/5/6/7 work is excluded rather than silently planned.

## Goal and requirements

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| GOAL | — | Direct BlobStore users interact with one versioned, backend-neutral record and deterministic fail-closed read contract. | 02-01 through 02-07 | COVERED | Tracer establishes the path; expansions cover parity, formats, errors, integrity, all operations, and legacy edges. |
| REQ | STOR-01 | Every stored entry uses one versioned canonical manifest across supported backends. | 02-01, 02-02, 02-03 | COVERED | Phase 2 local JSON/memory/SQLite parity; later advertised backend matrix remains Phase 4/5. |
| REQ | STOR-02 | Normal reads expose only committed entry generations. | 02-01, 02-05, 02-06 | COVERED | Authenticated committed state gates every direct read surface. |
| REQ | STOR-08 | Direct operations distinguish missing, corrupt, conflict, and backend failure. | 02-01, 02-02, 02-04, 02-05, 02-06 | COVERED | Only absent raw record maps to compatible absence. |
| REQ | SECU-03 | Verify manifest authenticity and payload integrity before deserialization. | 02-01, 02-03, 02-05, 02-06 | COVERED | Exact order and one-snapshot tests block bypasses. |
| REQ | SECU-04 | Required signing fails closed for key/signature/permission/configuration faults. | 02-01, 02-05 | COVERED | Strict provider is separate from the legacy auto-generating signer. |
| REQ | SECU-05 | Signed manifests bind locator, handler/type, format, and lifecycle-generation fields. | 02-01, 02-05 | COVERED | Complete fixed projection and per-field mutation matrix. |
| REQ | SECU-08 | BlobStore integrity exceptions remain typed; UnifiedCache has a later translation seam. | 02-04, 02-05, 02-06 | COVERED | Pure classifier only; no cache behavior changes. |
| REQ | MIGR-02 | Metadata schema and payload format have independent version identifiers. | 02-01, 02-03 | COVERED | Exact manifest and handler-owned versions. |
| REQ | MIGR-07 | Unknown future formats fail explicitly without guess/rewrite/delete. | 02-01, 02-03, 02-04, 02-05, 02-07 | COVERED | Typed early dispatch plus fixture immutability. |

## Spec-less edge and prohibition fallbacks

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| FALLBACK | STOR-01 adjacency | Identical logical field sets encode to identical canonical bytes independent of backend; duplicate/conflicting representations are rejected rather than merged. | 02-01, 02-02 | COVERED | Explicit truths, behavior, and golden/parity tests. |
| FALLBACK | STOR-01 empty | Missing repository entry alone is absence; empty/null/malformed bytes are corruption. | 02-01, 02-04 | COVERED | Explicit truth and outcome taxonomy. |
| FALLBACK | STOR-01 ordering | Canonical authenticated bytes use deterministic field order. | 02-01 | COVERED | Golden bytes across insertion order. |
| FALLBACK | STOR-02 | Generic shape classifier returned unclassified. | 02-01, 02-05, 02-06 | COVERED | Flag retained in planner assumptions; requirement/context text becomes acceptance. |
| FALLBACK | STOR-08 | Generic shape classifier returned unclassified. | 02-01, 02-02, 02-04, 02-05, 02-06 | COVERED | Flag retained in planner assumptions; direct outcome matrix is explicit. |
| FALLBACK | SECU-03 | Generic shape classifier returned unclassified. | 02-01, 02-05, 02-06 | COVERED | Flag retained; ordered security pipeline is explicit. |
| FALLBACK | SECU-04 | Generic shape classifier returned unclassified. | 02-01, 02-05 | COVERED | Flag retained; complete key/signature matrix is explicit. |
| FALLBACK | SECU-05 | Generic shape classifier returned unclassified. | 02-01, 02-05 | COVERED | Flag retained; signed projection mutation cases are explicit. |
| FALLBACK | SECU-08 | Generic shape classifier returned unclassified. | 02-04, 02-05, 02-06 | COVERED | Flag retained; pure seam and public integrity types are explicit. |
| FALLBACK | MIGR-02 | Generic shape classifier returned unclassified. | 02-01, 02-03 | COVERED | Flag retained; independent exact identifiers are explicit. |
| FALLBACK | MIGR-07 | Generic shape classifier returned unclassified. | 02-01, 02-03, 02-04, 02-07 | COVERED | Flag retained; unsupported and legacy conversion outcomes are explicit. |
| PROHIBITION | P-01 | No Cacheness wrapper/header around native handler payload formats. | 02-01 through 02-07 | COVERED, FLAGGED-UNVERIFIED | Preserved under every plan's must_haves.prohibitions; semantic proof closes in 02-03 and 02-07. |
| PROHIBITION | P-02 | Reads do not mutate/rewrite/quarantine/delete legacy, corrupt, or future entries. | 02-01 through 02-07 | COVERED, FLAGGED-UNVERIFIED | Preserved under every plan; pre/post evidence tests close in 02-05 and 02-07. |
| PROHIBITION | P-03 | Required signing never downgrades, invents ephemeral replacement, or accepts unsigned manifests. | 02-01 through 02-07 | COVERED, FLAGGED-UNVERIFIED | Preserved under every plan; strict-provider matrix closes in 02-05. |

## Context decisions

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| CONTEXT | D-01 | One canonical manifest schema 1 with all required fields on new writes. | 02-01 | COVERED | Persisted contract rated one-way; gates suppressed per unattended mode. |
| CONTEXT | D-02 | Backend-neutral deterministic typed model. | 02-01, 02-02 | COVERED | Exact canonical bytes across local repositories. |
| CONTEXT | D-03 | Native handlers own payload containers; manifest describes format/version. | 02-01, 02-03 | COVERED | Native-format contract and tests. |
| CONTEXT | D-04 | Unknown future schemas/formats fail typed and non-mutatingly. | 02-01, 02-03, 02-07 | COVERED | Exact dispatch before payload access. |
| CONTEXT | D-05 | Only true absence is a miss; other direct outcomes remain typed. | 02-01, 02-04, 02-06 | COVERED | Stable hierarchy and operation matrix. |
| CONTEXT | D-06 | Only committed manifests are visible. | 02-01, 02-05, 02-06 | COVERED | No Phase 3 state transition protocol is added. |
| CONTEXT | D-07 | Bounded parse → auth → critical validation → snapshot → digest/size → handler. | 02-01, 02-05 | COVERED | Event-order and exactly-one-snapshot assertions. |
| CONTEXT | D-08 | SHA-256 is canonical security digest; XXH3 stays non-security/legacy. | 02-01, 02-05 | COVERED | Streaming SHA-256/size verifier. |
| CONTEXT | D-09 | Deterministic canonical HMAC-SHA256 binds all critical fields. | 02-01, 02-05 | COVERED | Fixed complete projection. |
| CONTEXT | D-10 | Required signing rejects key/signature/permission/configuration failures. | 02-01, 02-05 | COVERED | No silent fallback path. |
| CONTEXT | D-11 | Integrity does not sandbox pickle/dill; trusted payload boundary remains. | 02-03, 02-05, 02-07 | COVERED | Explicit threat acceptance and regression. |
| CONTEXT | D-12 | New writes canonical only; exact legacy adapters read-only. | 02-01, 02-07 | COVERED | No read-time canonical write. |
| CONTEXT | D-13 | Legacy conversion-needed outcome is typed; execution remains Phase 7. | 02-04, 02-07 | COVERED | Inspectable outcome only. |
| CONTEXT | D-14 | BlobStore owns typed outcomes; pure later cache translation seam only. | 02-04 | COVERED | `core.py` remains untouched. |

## Research, pattern, and validation commitments

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| RESEARCH | R-01 | Frozen manifest, strict bounded codec, complete signature projection. | 02-01 | COVERED | Includes bytes/depth/count/string/integer/type bounds. |
| RESEARCH | R-02 | Narrow raw repository above legacy metadata backends; dedicated SQLite table. | 02-02 | COVERED | JSON/memory/SQLite exact bytes and reopen. |
| RESEARCH | R-03 | Authenticate before critical-field use; verify one snapshot. | 02-05 | COVERED | High-severity blocking gate. |
| RESEARCH | R-04 | Exact non-deserializing handler format support. | 02-03 | COVERED | All built-in native paths. |
| RESEARCH | R-05 | Stable direct outcome taxonomy. | 02-04 | COVERED | Public types/reasons and causes. |
| RESEARCH | R-06 | Full public operation integrity matrix. | 02-06 | COVERED | Read-only and mutation surfaces. |
| RESEARCH | R-07 | Pure UnifiedCache translation classifier. | 02-04 | COVERED | No cache imports or mutations. |
| RESEARCH | R-08 | Strict key provider and required signing. | 02-05 | COVERED | POSIX attestation and explicit cross-platform result. |
| RESEARCH | R-09 | Exact read-only legacy compatibility. | 02-07 | COVERED | Eight-fixture hash/mtime proof. |
| RESEARCH | R-10 | No new package dependencies. | all | COVERED | Stdlib plus existing repository primitives only; no package-legitimacy gate needed. |
| PATTERNS | PATT-01 | Follow model/repository/integrity/read-contract/blob-store/barrel/error/interface/handler analogs. | 02-01 through 02-07 | COVERED | Each task has exact read_first files and identified seam. |
| VALIDATION | W0-01 | `tests/test_blob_manifest.py`. | 02-01, 02-03 | COVERED | Golden codec, bounds, versions, native formats. |
| VALIDATION | W0-02 | `tests/test_blob_manifest_backends.py`. | 02-02 | COVERED | Local parity and reopen. |
| VALIDATION | W0-03 | `tests/test_blob_store_read_contract.py`. | 02-01, 02-06, 02-07 | COVERED | Tracer, all operations, final matrix. |
| VALIDATION | W0-04 | `tests/test_blob_store_integrity.py`. | 02-05 | COVERED | Signing/digest/tamper/one-snapshot. |
| VALIDATION | W0-05 | `tests/test_blob_store_legacy_contract.py`. | 02-07 | COVERED | Eight-fixture non-mutation. |
| VALIDATION | W0-06 | `tests/test_blob_store_translation_seam.py`. | 02-04 | COVERED | Pure classifier taxonomy. |

## Explicit exclusions (not gaps)

- Phase 3: general CAS, generation transitions, overwrite/delete races, crash recovery, reconciliation, and idempotent close.
- Phase 4: generalized injected/registered backend composition and topology/capability contracts.
- Phase 5: filesystem/memory/S3 × JSON/memory/SQLite/PostgreSQL full matrix.
- Phase 6: UnifiedCache delegation, policy translation, statistics, TTL, eviction, and invalidation changes.
- Phase 7: inventory, copy-verify-switch migration, resumption, and rebuild confirmation.
- Hostile pickle/dill deserialization remains outside the trusted-application-payload model.
- API detector is false; internal BlobStore APIs are not external integration, so no `COVERAGE.md` is created.

## Audit verdict

**COVERED — no unplanned in-scope items and no phase split required.**
