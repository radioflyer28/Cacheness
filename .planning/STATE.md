---
gsd_state_version: 1.0
milestone: v1.0
current_phase: 03
current_phase_name: Atomic Lifecycle and Recovery Engine
status: executing
stopped_at: Completed 03-07-PLAN.md
last_updated: "2026-08-31T04:05:06.515Z"
last_activity: 2026-08-30
last_activity_desc: Phase 03 execution started
state_head: 995201be66218bdf14da72d59584113e58abd756
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 32
  completed_plans: 29
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-29)

**Core value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.
**Current focus:** Phase 03 — Atomic Lifecycle and Recovery Engine

## Current Position

Phase: 03 (Atomic Lifecycle and Recovery Engine) — EXECUTING
Plan: 8 of 10
Status: Ready to execute
Last activity: 2026-08-30 — Phase 03 execution started

Progress: [█░░░░░░░░░] 1 of 8 phases complete

## Performance Metrics

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01-compatibility-and-security-baseline P01 | 5min | 2 tasks | 6 files |
| Phase 01-compatibility-and-security-baseline P02 | 14min | 2 tasks | 5 files |
| Phase 01 P06 | 8min | 2 tasks | 3 files |
| Phase 01-compatibility-and-security-baseline P08 | 3min | 3 tasks | 7 files |
| Phase 01 P03 | 12min | 3 tasks | 7 files |
| Phase 01-compatibility-and-security-baseline P09 | 8min | 2 tasks | 7 files |
| Phase 01-compatibility-and-security-baseline P04 | 14min | 2 tasks | 3 files |
| Phase 01 P05 | 6min | 2 tasks | 4 files |
| Phase 01 P10 | 5min | 2 tasks | 8 files |
| Phase 01 P11 | 4min | 2 tasks | 8 files |
| Phase 01 P12 | 35min | 2 tasks | 5 files |
| Phase 01 P07 | 15min | 2 tasks | 5 files |
| Phase 01 P13 | 8min | 2 tasks | 7 files |
| Phase 01 P15 | 35min | 3 tasks | 8 files |
| Phase 02 P01 | 7h 50m | 2 tasks | 8 files |
| Phase 02 P02 | 8 min | 2 tasks | 4 files |
| Phase 02 P03 | 6min | 2 tasks | 3 files |
| Phase 02-canonical-storage-and-integrity-contract P04 | 6 min | 2 tasks | 4 files |
| Phase 02 P05 | 10 min | 2 tasks | 4 files |
| Phase 02 P06 | 6 min | 2 tasks | 2 files |
| Phase 02 P07 | 10min | 2 tasks | 4 files |
| Phase 03 P01 | 18min | 2 tasks | 9 files |
| Phase 03 P02 | 7 min | 2 tasks | 4 files |
| Phase 03-atomic-lifecycle-and-recovery-engine P03 | 18 min | 3 tasks | 8 files |
| Phase 03 P04 | 8 min | 2 tasks | 3 files |
| Phase 03 P05 | 12 min | 2 tasks | 8 files |
| Phase 03 P06 | 8 min | 2 tasks | 4 files |
| Phase 03 P07 | 785 | 3 tasks | 6 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- `BlobStore` is the canonical payload-plus-metadata lifecycle owner; `UnifiedCache` owns cache policy only.
- `SqlCache` remains a separate subsystem, and supported public APIs stay available through compatibility adapters.
- V1 supports same-backend migration plus an explicit rebuild for incompatible or cross-backend data.
- Direct `BlobStore` integrity failures are typed exceptions; `UnifiedCache` may translate them into separately recorded misses.
- AWS S3 semantics are authoritative; compatible services are supported only where explicitly verified.
- [Phase 01]: CacheReason uses stable lower-snake-case string values in typed error contexts.
- [Phase 01]: SQL cache classes remain importable without optional dependencies; construction reports actionable guidance.
- [Phase 01]: CacheStorageConfig preserves authored paths; storage boundaries resolve them at runtime.
- [Phase 01]: FilesystemBlobBackend accepts only strict opaque IDs; Plan 03 owns logical-key encoding.
- [Phase 01]: Resolved storage roots are anchored once and managed descendants are revalidated per operation.
- [Phase 01]: SqlCache strict mode stages all missing-range frames and raises one typed completeness error before storage if any range fails or is empty.
- [Phase 01]: SqlCache best-effort mode is explicit and returns SqlCacheResult with immutable ordered failure records.
- [Phase 01]: Custom gap fallback is best-effort-only, while bulk-to-row upsert uses an observable savepoint-backed fallback.
- [Phase 01]: The independent corpus validator owns the exact eight-fixture matrix and accepts only staged manifest prefixes.
- [Phase 01]: Legacy raw fixtures are verified through bounded framing and direct decompressed-byte comparison without historical readers or metadata evaluation.
- [Phase 01]: Generated evidence records equal source/copy SHA-256 values plus an independent manifest digest for provenance.
- [Phase 01]: High-level handlers serialize in private stages and deserialize only from a live private snapshot copied once through ManagedFileOps.
- [Phase 01]: BlobStore keys and UnifiedCache prefixes remain exact public metadata while payload paths use versioned length-framed SHA-256 identifiers.
- [Phase 01]: Unsafe locator preflight covers complete high-level operation sets before reads, cleanup, metadata mutation, or result exposure.
- [Phase 01]: The independent validator, rather than fixture discovery, owns the exact signed and unsigned split-map schema and accumulated corpus prefix.
- [Phase 01]: Signed split-map evidence uses a fixed, disclosed test-only 32-byte HMAC input and records a distinct wrong-key verification failure expectation in provenance.
- [Phase 01]: Legacy raw-array frames remain read-only compatibility input and are parsed through bounded tuple metadata plus exact byte validation.
- [Phase 01]: New ordinary arrays use native NPZ with pickle disabled; no Cacheness raw-array writer remains.
- [Phase 01]: Object arrays require explicit signing, integrity verification, unsigned-entry rejection, and snapshot authorization before ObjectHandler.
- [Phase 01]: query_meta validates the complete filter-key set before any backend/session access and exposes invalid fields as typed errors.
- [Phase 01]: query_meta binds SQLite JSON paths and values separately while retaining raw filters and legacy serialized-string compatibility.
- [Phase 01]: Historical SQLite fixture inspection is read-only and records the exact legacy schema, data-version invariant, and source/copy digests.
- [Phase 01]: The 0.3.13 decorator compatibility path accepts one recorded candidate and derived storage key; it never scans metadata.
- [Phase 01]: Current JSON fixture provenance records one exact unified key that the independent validator recomputes and looks up directly.
- [Phase 01]: Current SQLite control provenance records the exact denormalized schema, metadata_json absence, and read-only data_version through a copied read-only URI.
- [Phase 01]: Exact legacy metadata layouts are read-only; successful reads use process-local counters only.
- [Phase 01]: Current signature verification remains first; only the signed split-map discriminator enables the exact six-field legacy HMAC fallback.
- [Phase 01]: Decorator compatibility derives one 0.3.13 candidate after a current-key miss and never scans metadata.
- [Phase 01]: Integrity, HMAC, and content digests establish authenticity or tamper evidence; they never sandbox pickle or dill.
- [Phase 01]: Phase quality evidence caps the existing Ruff baseline and requires every Phase 1-created Python file to be clean.
- [Phase 01]: Guarded handler publication rechecks the exact validated regular-file identity after its final open.
- [Phase 01]: query_meta accepts only signed-64 integers before backend/session access and keeps bool exact-match semantics.
- [Phase 01]: Gap-wave validation remains draft and approval pending until the orchestrator renews review after Plans 01-13 through 01-15.
- [Phase 01]: Clear recovery is a bounded BlobStore-only primitive with exact local topology identities; Plan 01-15 must integrate its prerequisite into UnifiedCache before CR-04 can close.
- [Phase 01]: Candidate publication and UnifiedCache global clear now reuse bounded ownership/recovery rules, but the primitive remains clear-only for Phase 3 absorption and validation stays pending orchestrator review.
- [Phase 02]: Schema-1 manifests use restricted deterministic JSON and bind every canonical field except signature.
- [Phase 02]: Only a genuinely absent raw repository record maps to None; malformed, unauthenticated, unsupported, and lifecycle-conflict records are typed failures.
- [Phase 02]: Read integrity validates one private payload snapshot before handler deserialization.
- [Phase 02]: SQLite carries raw canonical manifests only through its clear-recovery-snapshotted cache_key_params projection; empty records remain typed corruption.
- [Phase 02]: JSON and in-memory repositories preserve canonical bytes through reversible base64 metadata transport, while SQLite uses an isolated BLOB table.
- [Phase 02]: BlobStore admits only exact JsonBackend, SqliteBackend, and InMemoryBackend identities for Phase 2 manifest persistence.
- [Phase 02]: SQLite sidecar rows are removed only after the existing clear-recovery coordinator establishes terminal cache_entries authority.
- [Phase 02]: Handler payload format versions are explicit Cacheness contracts, independent of manifest schemas and dependency versions.
- [Phase 02]: HandlerRegistry resolves exact payload contracts without opening payload bytes; legacy Blosc2 is selected only by declared identity.
- [Phase 02]: Precise BlobStore subtypes retain existing broad error bases and stable direct reason codes for compatibility.
- [Phase 02]: The future cache seam classifies only explicit BlobStore failures and returns None for compatible direct absence.
- [Phase 02]: Canonical signing accepts only exact 32-byte key material or strict POSIX no-follow file attestation; reopen never creates fallback keys.
- [Phase 02]: BlobStore direct reads authenticate and validate signed fields before one guarded snapshot is SHA-256 and size verified for handler deserialization.
- [Phase 02]: Direct BlobStore operations derive authoritative state only from authenticated committed manifests.
- [Phase 02]: Metadata patches re-sign only user_metadata; clear builds mappings only after full manifest preflight.
- [Phase 02]: Exact Phase 1 fixture trees attach in-memory identities and report migration-required; Phase 7 alone owns migration execution.
- [Phase 02]: Legacy SQLite inspection uses immutable read-only mode so compatibility detection cannot create journal sidecars.
- [Phase 02]: Every direct read API preserves malformed, future-version, lifecycle, and local-backend failures rather than collapsing them into absence.
- [Phase 03]: BlobStore publication now uses immutable generation locators and exact manifest CAS as its sole authority transition.
- [Phase 03]: Reopen recovery acts only on authenticated manifest authority plus validated signed operation evidence; normal reads do not mutate evidence.
- [Phase 03]: Manifest authority mutations require the authenticated generation plus SHA-256 of the exact canonical record; repository adapters compare opaque bytes only.
- [Phase 03]: JSON conditional publication refreshes under a short OS-backed lock, while SQLite takes a writer transaction before exact-record comparison.
- [Phase 03]: LifecycleLimits is declared once in cacheness.config and is passed by identity from CacheConfig through BlobStore, LifecycleEngine, and the operation repository.
- [Phase 03]: Operation evidence uses a dedicated HMAC domain and exact-byte conditional checkpoint and retirement; grace never authorizes unauthenticated evidence.
- [Phase 03]: Stale write cleanup reclaims only the operation-bound immutable candidate after exact CAS conflict.
- [Phase 03]: Delete publishes a signed tombstone before payload reclamation and retires it with exact-record CAS.
- [Phase 03]: StoreAdmissionBarrier holds aggregate admission only while clear persists its authenticated finite target snapshot; ordinary operations are otherwise concurrent.
- [Phase 03]: Clear checkpoints exact authenticated target pages and treats changed current records as conflicts, never as authority to delete a later generation.
- [Phase 03]: Legacy clear evidence is reopen-only; new BlobStore clears are owned exclusively by LifecycleEngine.
- [Phase 03]: BlobStore ordinary operations bypass predecessor global admission and use StoreAdmissionBarrier only for current clear snapshot establishment.
- [Phase 03]: Reconciliation defaults to bounded dry-run reports and only applies authenticated, revalidated exact evidence.
- [Phase 03]: Reconciliation resumes with encrypted authenticated independent manifest and operation cursors.

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 1 planning must determine the supported legacy read window from released fixtures.
- Phase 3 planning must derive tombstone retention and orphan grace defaults from fault/crash testing.
- Phase 8 performance and coverage thresholds must be finalized from measured baselines rather than estimates.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Extensibility | General cache-policy plugin framework | Deferred to v2 | Project initialization |
| Storage | New backend families, deduplication, and cross-backend physical migration | Deferred to v2 | Project initialization |
| APIs | Native async storage/cache APIs and distributed coherence | Deferred to v2 | Project initialization |
| Security | Hostile pickle/dill deserialization | Out of scope; trusted payload boundary | Project initialization |
| Architecture | `SqlCache` redesign or merger | Out of scope | Project initialization |

## Session Continuity

Last session: 2026-08-31T04:05:06.471Z
Stopped at: Completed 03-07-PLAN.md
Resume file: None
