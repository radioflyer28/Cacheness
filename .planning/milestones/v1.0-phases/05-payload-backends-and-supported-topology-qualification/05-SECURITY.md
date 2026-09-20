---
phase: 05
slug: payload-backends-and-supported-topology-qualification
status: verified
threats_open: 0
asvs_level: 1
register_authored_at_plan_time: true
threats_total: 43
threats_closed: 43
threats_transferred: 5
created: 2026-09-08
---

# Phase 5 — Security

> Per-phase security contract for backend composition, immutable payload
> generations, PostgreSQL authority, S3 payload I/O, reconciliation, and
> qualification evidence.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Caller → topology composition | Selects named or injected lifecycle participants | Backend identity, role, capability request, configuration |
| Lifecycle engine → authority | Executes the canonical lifecycle state machine | Mutation intent, generation identity, receipts, cleanup debt |
| Lifecycle engine → payload backend | Publishes and reads immutable generations | Trusted application payloads, signed manifests, digests and sizes |
| Process → PostgreSQL | Uses the multi-host lifecycle authority | Bound SQL values, lifecycle state, retryable database outcomes |
| Process → Amazon S3 | Uses the remote immutable payload participant | Object keys, payload bytes, provider responses, bounded inventories |
| Qualification runner → release evidence | Converts live observations into a committed, sanitized record | Redacted service identity, test result, revision, cleanup status |

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation / evidence | Status |
|-----------|----------|-----------|----------|-------------|-----------------------|--------|
| T-05-01 | Spoofing | Topology profiles | high | mitigate | Exact immutable identity-pair lookup; injected/named construction tests | closed |
| T-05-02 | Elevation of privilege | Metadata roles | high | mitigate | Lifecycle methods require authority roles; projection roles are rejected | closed |
| T-05-03 | Tampering | Topology declarations | medium | mitigate | Profiles resolve by normalized authority/payload tuple | closed |
| T-05-04 | Denial of service | Composition | medium | mitigate | Capability preflight runs before participant factories | closed |
| T-05-05 | Tampering | Memory payloads | high | mitigate | Exclusive immutable publication followed by digest/size verification | closed |
| T-05-06 | Elevation of privilege | Memory snapshots | high | mitigate | Private `0600` snapshots and integrity verification before dispatch | closed |
| T-05-07 | Repudiation | Payload faults | medium | mitigate | Named fault boundaries produce attributable recovery evidence | closed |
| T-05-08 | Denial of service | Contention progress | medium | accept | ADR 0001 permits bounded typed retryable outcomes; accepted risk below | closed |
| T-05-09 | Tampering | S3 publication | high | mitigate | Conditional single/multipart publication and ambiguity verification | closed |
| T-05-10 | Elevation of privilege | S3 snapshots | high | mitigate | Bounded private materialization and integrity-before-handler dispatch | closed |
| T-05-11 | Information disclosure | S3 credentials/evidence | high | mitigate | Provider-chain boundary plus evidence allow-list and secret rejection | closed |
| T-05-12 | Denial of service | S3 operations | high | mitigate | Explicit object, part, retry, download, and inventory bounds | closed |
| T-05-13 | Tampering | S3 namespace | high | mitigate | Bucket/prefix/locator containment and exact delete/abort targets | closed |
| T-05-14 | Tampering | PostgreSQL SQL | high | mitigate | Identifiers are quoted and values are bound | closed |
| T-05-15 | Spoofing | PostgreSQL authority | high | mitigate | Schema, version, store, capability, and server identity validation | closed |
| T-05-16 | Tampering | PostgreSQL promotion | high | mitigate | Conditional transitions with exact expected state and `RETURNING` | closed |
| T-05-17 | Information disclosure | PostgreSQL errors | high | mitigate | Bounded stage/SQLSTATE context, redaction, and preserved causes | closed |
| T-05-18 | Elevation of privilege | PostgreSQL participant | high | mitigate | Complete authority protocol; no placeholder transitions | closed |
| T-05-19 | Tampering | Recovery scans | high | mitigate | Clear/reconciliation state validation and bounded cursors | closed |
| T-05-20 | Tampering | Cleanup debt | high | mitigate | Exact conditional retirement/deletion predicates with durable debt | closed |
| T-05-21 | Denial of service | PostgreSQL scans | high | mitigate | Page- and work-bounded catalog/reconciliation queries | closed |
| T-05-22 | Repudiation | PostgreSQL progress | medium | mitigate | Explicit SQLSTATE-to-typed-outcome mapping | closed |
| T-05-23 | Information disclosure | PostgreSQL diagnostics | high | mitigate | Errors expose only bounded operation/stage/SQLSTATE metadata | closed |
| T-05-24 | Elevation of privilege | Composition roles | high | mitigate | Structural role validation separates authority and payload participants | closed |
| T-05-25 | Tampering | Inventory attribution | high | mitigate | Authority-mediated attribution; indeterminate objects are report-only | closed |
| T-05-26 | Repudiation | Qualification claims | high | mitigate | Immutable declarations exclude readiness; live status is separate/read-only | closed |
| T-05-27 | Denial of service | Reconciliation | high | mitigate | Resumable cursors, malformed-token rejection, independently bounded inventory | closed |
| T-05-28 | Spoofing | Remote manifests | high | mitigate | Multi-host profile requires a shared external signing key | closed |
| T-05-29 | Information disclosure | Evidence schema | high | mitigate | Exact field/service allow-lists and forbidden secret/URL patterns | closed |
| T-05-30 | Tampering | Qualification cleanup | high | mitigate | Exact ownership markers, contained namespaces, and bounded cleanup pages | closed |
| T-05-31 | Spoofing | Qualification execution | high | mitigate | Fixed suites reject skips, deselection, and zero collection | closed |
| T-05-32 | Denial of service | Live qualification | medium | mitigate | Runner and cleanup operations have explicit bounds | closed |
| T-05-33 | Repudiation | Evidence provenance | medium | mitigate | Exact revision, UTC time, runtime/service identities, result, cleanup status | closed |
| T-05-34 | Spoofing | S3 service identity | high | mitigate | Custom endpoints rejected; standard Amazon S3 identity required | closed |
| T-05-35 | Tampering | Multi-client topology | high | mitigate | PostgreSQL CAS, shared signer, immutable generation, digest/size proof | closed |
| T-05-36 | Repudiation | Live test inventory | high | mitigate | Fixed live modules/markers and incomplete-run rejection | closed |
| T-05-37 | Denial of service | Live contention | medium | mitigate | Statement/lock timeouts, bounded lists, and bounded thread joins | closed |
| T-05-38 | Information disclosure | Signer/evidence | high | mitigate | Signer redaction and exact committed-evidence validation | closed |
| T-05-39 | Repudiation | Published contracts | high | mitigate | Marker-bounded tables are compared with runtime records | closed |
| T-05-40 | Spoofing | Live evidence | high | mitigate | Separate schema validation; `UNAVAILABLE` is explicitly non-passing | closed |
| T-05-41 | Tampering | Architecture gate | high | mitigate | Fixed source inventory and AST-based prohibited-form checks | closed |
| T-05-42 | Information disclosure | Documentation/evidence | high | mitigate | Docs reject mutable service status; evidence permits sanitized fields only | closed |
| T-05-43 | Denial of service | Contract verifier | medium | mitigate | Fixed inventory and 300-second subprocess timeout with fail-closed tests | closed |

All 43 canonical Phase 5 threats are closed. “Closed” includes the explicitly
accepted T-05-08 progress tradeoff; it does not imply that every contender must
succeed under arbitrary scheduling.

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-05-01 | T-05-08 | For SQLite/filesystem and PostgreSQL/S3 topologies, contention may end in a bounded typed retryable outcome. Requiring wait-free, starvation-free, or success-for-all-contenders behavior would contradict ADR 0001 and invite additional coordination authorities. Integrity and deterministic recovery remain mandatory. | Project owner via ADR 0001 and Phase 5 approval | 2026-09-08 |

## Transferred Threats (Excluded from Phase 5 Counts)

These threats belong to the Phase 8 BACK-05 real-service qualification gate.
They are neither Phase 5 passes nor Phase 5 open threats. The committed live
evidence remains `UNAVAILABLE` until that gate runs in eligible infrastructure.

| Threat ID | Category | Severity | Disposition | Transfer target |
|-----------|----------|----------|-------------|-----------------|
| T-05-44 | Spoofing | high | transfer | Phase 8 real PostgreSQL/Amazon S3 service identity |
| T-05-45 | Information disclosure | high | transfer | Phase 8 credential and evidence sanitation |
| T-05-46 | Tampering | high | transfer | Phase 8 isolated namespace and cleanup integrity |
| T-05-47 | Denial of service | medium | transfer | Phase 8 bounded live qualification execution |
| T-05-48 | Repudiation | medium | transfer | Phase 8 clean-revision evidence provenance |

Transfer authority: Phase 5 decision D-23, the superseded `05-10-PLAN.md`,
`ROADMAP.md`, and `REQUIREMENTS.md`.

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-08 | 43 | 41 | 2 non-blocking | GSD security auditor |
| 2026-09-08 | 43 | 43 | 0 | Codex remediation and verification |

Evidence was checked against the canonical Phase 5 plans, implementation,
contract tests, qualification tests, and ADR 0001. T-05-43 was remediated in
commit `9bedfbe`; focused tests and the complete local contract verifier pass.

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter
- [x] Transferred real-service threats remain assigned to Phase 8 and are not counted as passed

**Approval:** verified 2026-09-08
