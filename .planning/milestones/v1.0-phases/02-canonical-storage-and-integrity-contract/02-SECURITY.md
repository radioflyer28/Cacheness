---
phase: 02
slug: canonical-storage-and-integrity-contract
status: verified
threats_open: 0
asvs_level: 1
created: 2026-08-30
---

# Phase 02 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Caller to manifest repository | Caller-controlled keys and metadata enter backend-neutral canonical persistence | Logical keys, typed manifest bytes, user metadata |
| Manifest repository to payload backend | An authenticated committed manifest selects one contained payload snapshot | Locator, declared size, SHA-256 digest |
| Signing-key provider to integrity layer | Application or attested file key material authorizes canonical manifests | HMAC-SHA256 key bytes and signatures |
| Integrity layer to native handler | Handler dispatch occurs only after authenticated identity and same-snapshot verification | Trusted application payload bytes and exact format identity |
| Legacy evidence to compatibility reader | Exact known legacy layouts are inspected without mutation or speculative fallback | Existing manifests, payloads, signatures, hashes, and mtimes |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-02-01 | Tampering | Canonical manifest and signer | high | mitigate | Fixed complete signed projection, deterministic codec, constant-time HMAC verification, and per-field mutation tests | closed |
| T-02-02 | Tampering / Elevation of Privilege | BlobStore read ordering | high | mitigate | Authenticate before critical-field use; verify one private snapshot before handler invocation | closed |
| T-02-03 | Denial of Service | Manifest decoder | medium | mitigate | Independent byte, depth, collection, string, and integer bounds precede typed construction | closed |
| T-02-04 | Repudiation | Direct-read failure evidence | medium | mitigate | Stable typed reasons and non-mutating failure paths preserve evidence | closed |
| T-02-05 | Tampering | JSON/SQLite record persistence | high | mitigate | Exact canonical-byte round trips, durable reopen tests, and dedicated SQLite BLOB storage | closed |
| T-02-06 | Spoofing | Repository selection | high | mitigate | Exact supported local identities admitted; unknown and custom identities rejected before mutation | closed |
| T-02-07 | Repudiation | Repository error translation | medium | mitigate | Narrow backend failures preserve causes and never collapse to absence | closed |
| T-02-08 | Elevation of Privilege / Tampering | Handler dispatch | high | mitigate | Exact authenticated identity tuple and non-deserializing support checks reject unknown versions | closed |
| T-02-09 | Tampering | Native payload formats | medium | mitigate | Native framing and round-trip tests prove handlers retain payload-format ownership | closed |
| T-02-10 | Repudiation | Public error taxonomy | high | mitigate | Stable reason-coded exceptions and exhaustive classification tests distinguish failure from miss | closed |
| T-02-11 | Tampering | Cache translation seam | high | mitigate | Pure closed classifier has no UnifiedCache dependency or mutation capability | closed |
| T-02-12 | Information Disclosure | Error context | low | accept | Errors expose operation, key, category, and exception type but never payload bytes or key material | closed |
| T-02-13 | Spoofing / Tampering | Strict key provider | high | mitigate | Exact algorithm/key contract, no-follow file attestation, and no reopen regeneration | closed |
| T-02-14 | Tampering | Manifest critical fields | high | mitigate | Every structural field is in the fixed HMAC projection and independently mutation-tested | closed |
| T-02-15 | Tampering / Elevation of Privilege | Payload snapshot and handler | high | mitigate | One guarded snapshot is hashed and deserialized in one context after authentication | closed |
| T-02-16 | Repudiation | Failed-read evidence | medium | mitigate | Failed reads cannot update access, delete, rewrite, rotate keys, or clean evidence | closed |
| T-02-17 | Tampering | update_metadata | high | mitigate | Structural keys are rejected; mutable metadata is recanonicalized and re-signed | closed |
| T-02-18 | Tampering / Denial of Service | delete and clear | high | mitigate | Complete-record authentication and whole-set preflight occur before locator use or mutation | closed |
| T-02-19 | Repudiation | exists/list/get_metadata | medium | mitigate | Shared typed pipeline prevents invalid records from appearing absent or being omitted | closed |
| T-02-20 | Tampering / Elevation of Privilege | Legacy discriminator | high | mitigate | Exact layout, signature, and identity matching rejects lookalikes and speculative fallback | closed |
| T-02-21 | Repudiation | Legacy/future read evidence | high | mitigate | Recursive hashes, mtimes, and mutation spies prove read-only inspection | closed |
| T-02-22 | Tampering | Cross-contract regression | high | mitigate | Consolidated adversarial suite blocks canonical-auth, visibility, dispatch, and snapshot bypasses | closed |
| T-02-23 | Elevation of Privilege | Trusted pickle/dill payload | medium | accept | Application payloads are explicitly trusted; integrity does not claim sandboxed deserialization | closed |

*All 23 plan-time threats are closed. The configured blocking threshold is high.*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-02-01 | T-02-12 | Local callers need actionable typed context; payload bytes, signing keys, and secret material remain excluded | Phase 2 plan decision | 2026-08-30 |
| AR-02-02 | T-02-23 | Cacheness stores trusted application objects; sandboxing pickle/dill is outside the documented trust model | Project constraint and Phase 2 D-11 | 2026-08-30 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-08-30 | 23 | 23 | 0 | GSD secure-phase L1 verification |

The plan-time register was complete, all mitigation commands are represented by
the passing Phase 2 consolidated requirement gate, and no threat at or above the
configured `high` threshold remains open.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-08-30
