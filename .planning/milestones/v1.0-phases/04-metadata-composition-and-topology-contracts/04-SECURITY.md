---
phase: 04
slug: metadata-composition-and-topology-contracts
status: verified
threats_open: 0
asvs_level: 1
block_on: high
created: 2026-09-08
updated: 2026-09-08
---

# Phase 04 — Security

> ASVS Level 1 verification of the plan-time STRIDE register for metadata,
> composition, topology, projection, and release-evidence boundaries.

## Trust Boundaries

| Boundary | Description | Data crossing |
| --- | --- | --- |
| Caller input → catalog | Schemas, values, predicates, cursor bytes, and bounds enter canonical validation. | Application metadata and query state |
| Persisted state → lifecycle authority | Existing SQLite identity, signed descriptors, paths, and format versions may be stale, corrupt, or hostile. | Canonical lifecycle records and payload locators |
| Configuration → composition root | Names, instances, factories, options, roles, capabilities, and ownership enter participant resolution. | Backend configuration and live resources |
| Authority → projection | Authenticated canonical pages enter a less-trusted derived sink whose failures occur after commit. | Bounded catalog pages and checkpoints |
| Cache policy → BlobStore | `UnifiedCache` requests storage operations but may not acquire lifecycle ownership. | Cache keys, policy decisions, and receipts |
| Repository → release evidence | Executable imports, validation inventories, and interpreter results determine the qualified cutover claim. | Source syntax, test paths, and test results |

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation evidence | Status |
| --- | --- | --- | --- | --- | --- | --- |
| T-04-01 | Tampering | schema/query/cursor contracts | high | mitigate | Exact-type and bounded pre-dispatch validation plus HMAC-bound cursors in `storage/catalog.py`; catalog contract tests | closed |
| T-04-02 | Elevation of privilege | role/projection contracts | high | mitigate | Structural role checks in `storage/composition.py`; projection authority and capability pre-I/O tests | closed |
| T-04-03 | Repudiation | committed-partial outcomes | medium | mitigate | Frozen `BlobReceipt`, named projection outcome, checkpoint, and remaining-work attribution | closed |
| T-04-04 | Denial of service | Ruff baseline | low | accept | Deterministic, path-bounded local gate with no external input | accepted |
| T-04-02-01 | Tampering | manifest/catalog | high | mitigate | Canonical typed serialization, schema fingerprint, signed manifest verification, and tamper tests | closed |
| T-04-02-02 | Denial of service | metadata/query | medium | mitigate | Finite metadata, predicate, membership, page, and work limits with boundary tests | closed |
| T-04-02-03 | Spoofing | format identity | high | mitigate | Independent version checks and read-only foreign/mixed-layout rejection | closed |
| T-04-02-04 | Repudiation | BlobReceipt | medium | mitigate | Frozen operation, generation, expectation, revision, and projection fields | closed |
| T-04-02-SC | Tampering | package supply chain | low | accept | Plan added no package task; phase range has no dependency-manifest changes | accepted |
| T-04-03-01 | Spoofing | registry | high | mitigate | Typed role/name registry, structural validation, and collision tests | closed |
| T-04-03-02 | Elevation of privilege | capability minima | high | mitigate | Conservative composed capabilities and pre-I/O `CapabilityMinimum` rejection | closed |
| T-04-03-03 | Repudiation | ownership close | medium | mitigate | Explicit identity-deduplicated ownership ledger and close/unwind tests | closed |
| T-04-03-04 | Tampering | duplicate coordinator | high | mitigate | One topology and one `AuthorityLifecycleEngine` per `BlobStore`; tracer tests | closed |
| T-04-03-SC | Tampering | package supply chain | low | accept | Plan added no package task; phase range has no dependency-manifest changes | accepted |
| T-04-13 | Tampering | SQLite canonical descriptor | high | mitigate | SQLite identity checks, transactional promotion, signed descriptors, and rollback tests | closed |
| T-04-14 | Tampering | query evaluator | high | mitigate | Finite fields/operators and authenticated-descriptor evaluation after pre-dispatch validation | closed |
| T-04-15 | Spoofing | opaque cursor | high | mitigate | HMAC plus store, format, schema, query, revision, and identity binding | closed |
| T-04-16 | Denial of service | paging/membership | medium | mitigate | Predicate, membership, page, and work caps with keyset paging | closed |
| T-04-17 | Repudiation | retryable progress outcomes | low | accept | Typed retryable outcomes with cause/context under ADR-qualified local progress semantics | accepted |
| T-04-05-01 | Tampering | checkpoint | high | mitigate | Checkpoint binds source, epoch, schema, query, revision, and cursor; mismatch tests | closed |
| T-04-05-02 | Elevation of privilege | projection role | high | mitigate | Narrow derived-only `ProjectionSink` protocol and role-confusion rejection | closed |
| T-04-05-03 | Denial of service | projection pull | medium | mitigate | Bounded batches, pages, and invocation work | closed |
| T-04-05-04 | Repudiation | partial result | medium | mitigate | Named immutable projection outcomes attached to the committed receipt | closed |
| T-04-05-SC | Tampering | package supply chain | low | accept | Plan added no package task; phase range has no dependency-manifest changes | accepted |
| T-04-22 | Tampering | manifest/integrity regressions | high | mitigate | Canonical HMAC, authenticated reads, rollback, and retained tamper tests | closed |
| T-04-23 | Elevation of privilege | topology fixtures | high | mitigate | Exact injected-role validation and projection-as-authority denial | closed |
| T-04-24 | Denial of service | concurrency tests | medium | mitigate | Deterministic barriers and ADR-qualified retry/conflict outcomes | closed |
| T-04-25 | Information disclosure | containment | high | mitigate | Traversal, symlink, retarget, and authority-locator containment tests | closed |
| T-04-26 | Elevation of privilege | metadata/projection tests | high | mitigate | Shared projection role/protocol and explicit authority denial | closed |
| T-04-27 | Tampering | mixed-scope test retirement | high | mitigate | Stable security and lifecycle modules retained in the exact passing matrix | closed |
| T-04-28 | Repudiation | exception/public contract | medium | mitigate | Stable reason codes/context and optional-import assertions | closed |
| T-04-29 | Denial of service | optional PostgreSQL tests | low | accept | Deterministic local role/import failure only; no live-service claim | accepted |
| T-04-30 | Elevation of privilege | legacy selectors/exports | high | mitigate | Public absence tests and executable-tree retired-consumer audit | closed |
| T-04-31 | Tampering | unsupported layouts | high | mitigate | Typed read-only classification and non-mutation regressions | closed |
| T-04-32 | Tampering | custom/ORM projection | high | mitigate | Projection-only pull/checkpoint contract with no authority methods | closed |
| T-04-33 | Repudiation | release matrix | medium | mitigate | Exact commands, interpreters, results, and non-claims recorded in `04-VALIDATION.md` | closed |
| T-04-34 | Denial of service | package/full-suite gate | low | accept | Exact, marker-bounded 42-module local matrix | accepted |
| T-04-35 | Tampering | catalog write/update | high | mitigate | Pre-mutation validation, signed catalog values/presence, and exact-record CAS | closed |
| T-04-36 | Elevation of privilege | payload participant | high | mitigate | Runtime payload-I/O protocol validation and guarded-root A/B tests | closed |
| T-04-37 | Information disclosure | opaque catalog values | medium | mitigate | Depth, entry, byte, and declared-query-field limits | closed |
| T-04-38 | Repudiation | catalog update result | medium | mitigate | Exact immutable receipt, committed expectation/revision, and typed stale conflict | closed |
| T-04-39 | Denial of service | schema validation | medium | mitigate | Field, nesting, value, and encoded-size bounds before lifecycle dispatch | closed |
| T-04-40 | Elevation of privilege | participant validation | high | mitigate | Runtime-checkable role protocols reject structurally invalid participants before I/O | closed |
| T-04-41 | Denial of service | ownership unwind | high | mitigate | Owned identities recorded before validation and closed once in reverse order | closed |
| T-04-42 | Spoofing | named registry resolution | high | mitigate | One topology-carried role/name registry; no legacy/global fallback | closed |
| T-04-43 | Tampering | projection role | high | mitigate | Projection protocol exposes no mutation, cleanup, or authority path | closed |
| T-04-44 | Repudiation | compatibility cutover | medium | mitigate | Public absence tests, AST audit, and explicit current-format rejection behavior | closed |
| T-04-45 | Repudiation | post-commit projection | high | mitigate | Ordinary projection exceptions preserve exact receipt, name, and cursor | closed |
| T-04-46 | Denial of service | cursor inspection | high | mitigate | Encoded, decoded, and per-field bounds precede decoding, parsing, and dispatch | closed |
| T-04-47 | Tampering | cursor context/signature | high | mitigate | Exact shape/type/signature checks and complete HMAC context binding | closed |
| T-04-48 | Elevation of privilege | projection rebuild | high | mitigate | Named sink capability and isolated publish contract; no authority operations | closed |
| T-04-50 | Spoofing | retired registry consumers | high | mitigate | Local `RoleRegistry`, retired API absence tests, and passing AST audit | closed |
| T-04-51 | Tampering | regression migration | high | mitigate | Integrity, recovery, initialization, version, and cleanup behavior retained | closed |
| T-04-52 | Elevation of privilege | S3 registry test | high | mitigate | Factory-only mocked evidence; no topology/lifecycle qualification | closed |
| T-04-53 | Repudiation | executable consumer audit | high | mitigate | AST scan covers all executable consumer roots independently of pytest collection | closed |
| T-04-54 | Tampering | validation path extraction | high | mitigate | Exact marker parsing, normalization, duplicate/overlap checks, and owned-only matrix | closed |
| T-04-55 | Repudiation | interpreter/collection evidence | high | mitigate | Exact matrix on Python 3.11/3.13 and separately bounded deferred diagnostics | closed |
| T-04-56 | Elevation of privilege | unsupported layout | high | mitigate | Typed non-mutating migration-required failure and explicit offline boundary | closed |
| T-04-57 | Elevation of privilege | JSON projection role | high | mitigate | Derived-only `JsonProjection`, strict `ProjectionSink` resolution, and authority-denial tests | closed |
| T-04-58 | Tampering | JSON projection document | high | mitigate | Exact state/checkpoint validation and same-directory atomic replacement | closed |
| T-04-59 | Repudiation | PostgreSQL participant | high | mitigate | Unqualified factory registration removed; typed unsupported resolution until Phase 5 | closed |
| T-04-60 | Tampering | executable-consumer audit | high | mitigate | Alias-aware AST visitor with table-driven positive and negative fixtures | closed |
| T-04-61 | Spoofing | accidental root module | high | mitigate | Root `metadata.py` absent; only package implementation remains | closed |
| T-04-62 | Repudiation | interpreter/release evidence | high | mitigate | Fresh exact matrix and Ruff-delta passes on both supported interpreters | closed |

All 64 plan-time threats have a final disposition: 57 mitigations are closed
and 7 low-severity risks are explicitly accepted. There are no transfers and no
open threats at any severity.

## Accepted Risks Log

| Risk ID | Threat ref | Rationale | Accepted by | Date |
| --- | --- | --- | --- | --- |
| AR-04-01 | T-04-04 | Ruff verification is deterministic, path-bounded, local, and receives no external input. | Project plan | 2026-09-08 |
| AR-04-02 | T-04-02-SC | Plan 04-02 introduced no dependency or package task. | Project plan | 2026-09-08 |
| AR-04-03 | T-04-03-SC | Plan 04-03 introduced no dependency or package task. | Project plan | 2026-09-08 |
| AR-04-04 | T-04-17 | Local contention may produce a typed retryable outcome; universal progress is outside ADR 0001. | Project plan | 2026-09-08 |
| AR-04-05 | T-04-05-SC | Plan 04-05 introduced no dependency or package task. | Project plan | 2026-09-08 |
| AR-04-06 | T-04-29 | Phase 4 verifies deterministic PostgreSQL role/import behavior only and makes no live-service claim. | Project plan | 2026-09-08 |
| AR-04-07 | T-04-34 | The release gate is intentionally bounded to the exact owned local matrix. | Project plan | 2026-09-08 |

Accepted items are recorded as accepted risks, not misreported as implemented
mitigations. They do not grant any guarantee beyond ADR 0001 or the Phase 4
qualification boundary.

## Verification Evidence

- `uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all`:
  consumer audit passed; 607 passed, 6 skipped across 42 modules on CPython
  3.11.16; the exact three pandas-dependent modules remained a separately
  classified non-green diagnostic.
- `uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all`:
  consumer audit passed; 607 passed, 6 skipped across 42 modules on CPython
  3.13.15; the same deferred diagnostic remained explicit.
- `uv run --frozen --python 3.11 python tools/verify_phase4_ruff_delta.py`: pass.
- `uv run --frozen --python 3.13 python tools/verify_phase4_ruff_delta.py`: pass.
- Phase-range inspection found no `pyproject.toml` or `uv.lock` changes.
- Repository-root `metadata.py` is absent.

The six skipped tests are capability/platform cases and are not sole evidence
for any registered mitigation.

## Security Audit Trail

| Audit date | Threats total | Closed | Accepted | Open | Run by |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026-09-08 | 64 | 57 | 7 | 0 | GSD security auditor |

## Sign-Off

- [x] All threats have a disposition.
- [x] Every mitigated threat has implementation or test evidence.
- [x] Accepted risks are documented separately.
- [x] `threats_open: 0` confirmed.
- [x] `status: verified` set in frontmatter.

**Approval:** verified 2026-09-08
