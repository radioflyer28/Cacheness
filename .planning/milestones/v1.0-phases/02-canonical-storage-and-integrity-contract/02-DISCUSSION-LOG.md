# Phase 2: Canonical Storage and Integrity Contract - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-30
**Phase:** 2-canonical-storage-and-integrity-contract
**Areas discussed:** Canonical manifest and versioning, direct read outcomes, integrity and signing, compatibility and downstream seams

---

## Canonical Manifest and Versioning

| Option | Description | Selected |
|--------|-------------|----------|
| One typed, versioned manifest | Backend-neutral model with independent manifest and payload format versions | ✓ |
| Backend-native records | Let JSON/SQLite/PostgreSQL shapes define the storage contract | |
| New wrapper container | Put payload and metadata behind another Cacheness-specific file header | |

**User's choice:** Auto-selected recommended option under the approved autonomous workflow.
**Notes:** Carries forward the user's explicit direction that native libraries/handlers own payload formats and Cacheness should not invent another container.

---

## Direct Read Outcomes

| Option | Description | Selected |
|--------|-------------|----------|
| Compatible miss plus typed failures | Missing remains the existing normal miss; corrupt/conflict/backend/version failures are distinct exceptions | ✓ |
| Result object everywhere | Replace all reads, including misses, with a new public result wrapper | |
| Miss for every failure | Collapse corruption and backend failures into absence | |

**User's choice:** Auto-selected recommended option under the approved autonomous workflow.
**Notes:** Preserves the public compatibility baseline while satisfying the direct `BlobStore` error distinction.

---

## Integrity and Signing

| Option | Description | Selected |
|--------|-------------|----------|
| SHA-256 payload + canonical HMAC manifest | Version algorithms and bind every security-critical field | ✓ |
| Keep XXH3 as security digest | Reuse the existing fast non-cryptographic content hash | |
| Sign only selected metadata | Preserve the current partial field coverage | |

**User's choice:** Auto-selected recommended option under the approved autonomous workflow.
**Notes:** Required signing fails closed and remains distinct from hostile-deserialization safety.

---

## Compatibility and Downstream Seams

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical new writes, read-only legacy adapters | Expose typed migration/version outcomes without rewriting on read | ✓ |
| Auto-upgrade on read | Rewrite legacy entries when encountered | |
| Remove legacy paths now | Reject all Phase 1 stored-format compatibility | |

**User's choice:** Auto-selected recommended option under the approved autonomous workflow.
**Notes:** Migration execution stays in Phase 7 and `UnifiedCache` rewiring stays in Phase 6.

## the agent's Discretion

- Exact type/module names, canonical codec details, and internal result layering.
- Opaque generation identifier representation without Phase 3 CAS claims.

## Deferred Ideas

- General lifecycle/CAS/recovery, backend composition, full backend matrix, cache-policy delegation, and migration execution remain in Phases 3-7 as recorded in CONTEXT.md.
