---
gsd_state_version: 1.0
milestone: v1.0
current_phase: 11
current_phase_name: Clean supplemental documentation and normalize validation evidence
status: ready_to_plan
stopped_at: Phase 11 context gathered
last_updated: "2026-09-19T16:40:20.145Z"
last_activity: 2026-09-17
last_activity_desc: Phase 11 added from v1.0 milestone audit debt
state_head: 6b887478bce6c145327de37318b06513bd4fb830
progress:
  total_phases: 12
  completed_phases: 10
  total_plans: 163
  completed_plans: 152
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-17)

**Core value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.
**Current focus:** Phase 11 — Clean supplemental documentation and normalize validation evidence

## Current Position

Phase: 11 (Clean supplemental documentation and normalize validation evidence) — READY TO EXECUTE
Plan: Not started
Status: Ready to discuss or plan
Last activity: 2026-09-17 — Phase 11 added from the v1.0 milestone audit

Current execution chain: **08-18 complete → 08-19 complete → 08-16 complete → independent Phase 08 verification**. No older Phase 8 ordering statement overrides this chain.

Phase 07.1 closed with 11/11 plans summarized and an independent 46/46
must-have verification pass. Built-in filesystem, memory, and S3 payload
mechanics now use one guarded `ObstoreGenerationIO`; `BlobStore` still owns the
single `AuthorityLifecycleEngine`, and `UnifiedCache` remains policy-only.

The security audit closes 47/47 plan threats after a bounded canonical-evidence
parser repair. ETag/version remain opaque signed transport corroboration;
canonical SHA-256 plus size remains the integrity decision. Live AWS,
PostgreSQL, and remote platform/package evidence remain unqualified until their
eligible gates run. Structural RSS contracts and hash-benchmark capability are
retained; controlled-Linux performance qualification is deferred to SEED-006 by
D-23, while real PostgreSQL/Amazon-S3 qualification and immutable publication are
deferred to SEED-007 by D-24.

Roadmap progress: [███████████░] 11 of 12 phases complete. Phase 9 closed the bounded
adoption surface; Phase 10 directly removed SqlCache while retaining BlobStore,
UnifiedCache, SQL lifecycle authorities, and dataframe handlers. Neither phase
reopened storage lifecycle design. The generated disk plan
counter still includes the deliberately superseded Phase 03 plan; do not reopen
that closed phase merely to repair the counter.

## Roadmap Evolution

- Phase 9 added: Adoption and Release Surface Closure. It closes the partial
  CACH-06 publication surface identified by the v1.0 milestone audit through
  current documentation, executable examples, package presentation, seed
  capture, and evidence refresh. It explicitly excludes lifecycle,
  concurrency, backend, compatibility-shim, or guarantee expansion work.

2026-09-06 — User-approved downstream alignment after Phase 3 direct qualification:

- Phase 4 edited: starting point and success criteria require reuse of the entry interface, catalog customization and narrower transactional adapters; schema choices remain open for discussion.
- Phase 5 edited: title, goal and success criteria now specify Supported Topology Qualification, not full Cartesian/availability parity; no advertised backend family was dropped.
- Phase 6 edited: carried-forward scope and criteria separate delivered local integration from remaining policy/decorator/statistics work and preserve derived-state partial outcomes.
- Phase 7 edited: goal and criteria require stopped-worker, explicit offline migration/cutover; no online writer coordination is implied.
- Phase 8 edited: criteria retain finite integrity/recovery cases, initialized-worker fixtures, separate performance distributions and honest platform/service qualification.
- PROJECT.md now describes the delivered baseline; REQUIREMENTS.md records STOR-03 through STOR-07 as complete only in the qualified local scope, leaving full BACK/CACH/MIGR/QUAL acceptance pending. CONTEXT.md distinguishes catalog attributes from derived metadata outcomes.

Phase numbering, order, dependencies, requirement ownership, and milestone scope
are unchanged. Phases 4 through 7 subsequently completed the catalog,
supported-topology, cache-policy, and explicit maintenance work. Phase 8 is the
only remaining milestone phase. Plan 01 established the deterministic exact-commit
evidence tracer; Plan 16 now closes only the local-readiness boundary while the
remote-service and publication classes stay unavailable under SEED-007. GSD's raw disk count still includes superseded 03-19 and may call Phase 3 partial;
the canonical 24/24 disposition and direct qualification ledger remain controlling.
Do not fabricate a 03-19 completion or reopen the closed gaps to repair that count.

2026-09-07 — During Phase 4 planning the user approved a pre-production
compatibility reset. Cacheness is not deployed in production, so overlapping
legacy constructors, selectors, aliases, metadata factories, and development-only
catalog layouts may be removed rather than supported through runtime adapters.
This does not weaken lifecycle safety or remove migration infrastructure: formats
remain versioned, unsupported layouts fail without mutation, and Phase 7 still
delivers explicit offline migration/rebuild tooling for future released versions.

## Performance Metrics

*Updated after each plan completion*
**Per-Plan Metrics:**

Phase 03 rows in this historical table include superseded attempts and do not
define current completion. The roadmap records Phases 1 through 7 complete and
Phase 8 is at 17/17 canonical plans complete: 08-11 and 08-12 are superseded by
D-24/SEED-007, 08-17 corrected one inherited progress assertion, 08-18 corrected
one inherited initialization assertion, 08-19 restored the frozen coverage
ratchet, and 08-16 produced the read-back-validated local-readiness evidence.
Raw file counts include superseded plans and therefore are not completion claims.

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
| Phase 03 P08 | 16min | 2 tasks | 5 files |
| Phase 03 P09 | 7h 28m | 2 tasks | 5 files |
| Phase 03 P10 | 29m | 3 tasks | 7 files |
| Phase 03 P01 | 1h 12m | 3 tasks | 8 files |
| Phase 03 P02 | 12min | 3 tasks | 7 files |
| Phase 03 P03 | 25min | 3 tasks | 8 files |
| Phase 03 P04 | 41min | 2 tasks | 8 files |
| Phase 03 P05 | 18min | 2 tasks | 8 files |
| Phase 03 P06 | 13min | 2 tasks | 8 files |
| Phase 03 P07 | 27m | 2 tasks | 13 files |
| Phase 03 P08 | 35min | 2 tasks | 11 files |
| Phase 03 P09 | 8min | 2 tasks | 12 files |
| Phase 03 P12 | 4min | 2 tasks | 2 files |
| Phase 03 P11 | 13m | 2 tasks | 4 files |
| Phase 03 P13 | 8m | 2 tasks | 5 files |
| Phase 03 P14 | 1h 29min | 3 tasks | 8 files |
| Phase 03 P15 | 31min | 3 tasks | 6 files |
| Phase 03 P17 | 1h 13m | 3 tasks | 7 files |
| Phase 03 P18 | 1h 31m | 5 tasks | 9 files |
| Phase 03 P20 | 1920 | 4 tasks | 12 files |
| Phase 04 P01 | 5m 13s | 2 tasks | 8 files |
| Phase 04-metadata-composition-and-topology-contracts P02 | 16m 10s | 2 tasks | 6 files |
| Phase 04-metadata-composition-and-topology-contracts P03 | 12m 43s | 2 tasks | 7 files |
| Phase 04 P04 | 16m 51s | 3 tasks | 12 files |
| Phase 04 P05 | 13m 52s | 2 tasks | 7 files |
| Phase 04-metadata-composition-and-topology-contracts P06 | 1575s | 2 tasks | 13 files |
| Phase 04 P07 | 22min | 2 tasks | 13 files |
| Phase 04 P08 | 5m | 2 tasks | 17 files |
| Phase 04 P10 | 13m 10s | 2 tasks | 9 files |
| Phase 04 P11 | 22m | 2 tasks | 5 files |
| Phase 04 P12 | 40m | 2 tasks | 9 files |
| Phase 04 P13 | 120m | 2 tasks | 8 files |
| Phase 06 P01 | 11 min | 2 tasks | 4 files |
| Phase 06 P02 | 13 min | 2 tasks | 7 files |
| Phase 06 P03 | 14 min | 2 tasks | 7 files |
| Phase 06 P04 | 10 min | 2 tasks | 5 files |
| Phase 06 P05 | 5min | 2 tasks | 2 files |
| Phase 06 P06 | 13min | 2 tasks | 10 files |
| Phase 06 P07 | 7min | 2 tasks | 5 files |
| Phase 06 P08 | 25min | 2 tasks | 5 files |
| Phase 06-unifiedcache-policy-composition P09 | 494s | 2 tasks | 5 files |
| Phase 06 P10 | 7 min | 2 tasks | 4 files |
| Phase 06 P11 | 12 min | 2 tasks | 7 files |
| Phase 07 P01 | 1348s | 2 tasks | 6 files |
| Phase 07 P02 | 125s | 3 tasks | 4 files |
| Phase 07 P03 | 969s | 2 tasks | 5 files |
| Phase 07-explicit-migration-and-rebuild-cutover P04 | 672s | 2 tasks | 8 files |
| Phase 07 P05 | 16m | 2 tasks | 4 files |
| Phase 07 P06 | 20min | 3 tasks | 9 files |
| Phase 07 P07 | 11min | 2 tasks | 5 files |
| Phase 07 P08 | 24m | 2 tasks | 6 files |
| Phase 07 P09 | 1178s | 3 tasks | 6 files |
| Phase 07 P10 | 12m | 2 tasks | 7 files |
| Phase 07 P11 | 27m | 2 tasks | 4 files |
| Phase 07 P12 | 23min | 2 tasks | 8 files |
| Phase 07-explicit-migration-and-rebuild-cutover P13 | 2h 57m | 2 tasks | 3 files |
| Phase 07 P14 | 19min | 2 tasks | 9 files |
| Phase 07 P15 | 35min | 2 tasks | 3 files |
| Phase 07 P16 | 18min | 2 tasks | 5 files |
| Phase 07 P17 | 10min | 2 tasks | 4 files |
| Phase 07 P18 | 20min | 2 tasks | 2 files |
| Phase 07 P19 | 20min | 2 tasks | 2 files |
| Phase 07 P20 | 12min | 2 tasks | 3 files |
| Phase 07 P21 | 18 min | 2 tasks | 4 files |
| Phase 07 P22 | 15 min | 2 tasks | 6 files |
| Phase 07 P23 | 36m | 1 tasks | 5 files |
| Phase 07-explicit-migration-and-rebuild-cutover P24 | 4m | 1 tasks | 2 files |
| Phase 07.1 P01 | 10min | 1 tasks | 1 files |
| Phase 07.1 P02 | 33min | 2 tasks | 3 files |
| Phase 07.1 P03 | 19min | 2 tasks | 4 files |
| Phase 07.1 P04 | 11m 19s | 2 tasks | 2 files |
| Phase 07.1 P05 | 16m 5s | 3 tasks | 11 files |
| Phase 07.1 P06 | 11m 14s | 2 tasks | 7 files |
| Phase 07.1 P07 | 9m 56s | 2 tasks | 8 files |
| Phase 07.1 P08 | 18min | 2 tasks | 11 files |
| Phase 07.1 P09 | 537 | 3 tasks | 12 files |
| Phase 07.1 P10 | 666s | 2 tasks | 10 files |
| Phase 07.1 P11 | 24m | 3 tasks | 13 files |
| Phase 08 P01 | 16m | 2 tasks | 3 files |
| Phase 08 P02 | 17m | 2 tasks | 3 files |
| Phase 08 P03 | 8m 33s | 2 tasks | 3 files |
| Phase 08 P04 | 12m 14s | 2 tasks | 2 files |
| Phase 08 P06 | 12m 10s | 2 tasks | 4 files |
| Phase 08 P07 | 680s | 3 tasks | 4 files |
| Phase 08 P08 | 838s | 3 tasks | 5 files |
| Phase 08 P05 | 4020 | 2 tasks | 23 files |
| Phase 08 P09 | 32min | 2 tasks | 8 files |
| Phase 08 P10 | 3301s | 3 tasks | 4 files |
| Phase 08 P13 | 12min | 1 tasks | 4 files |
| Phase 08-production-gates-and-performance-stabilization P14 | 1200 | 3 tasks | 5 files |
| Phase 08 P15 | 10m | 2 tasks | 4 files |
| Phase 08 P17 | 7m | 2 tasks | 1 files |
| Phase 08 P18 | 8min | 1 tasks | 1 files |
| Phase 08 P19 | 9m | 2 tasks | 1 files |
| Phase 08 P16 | 2h 57m | 2 tasks | 8 files |
| Phase 09 P01 | 5min | 2 tasks | 8 files |
| Phase 09 P02 | 13min | 2 tasks | 9 files |
| Phase 09 P03 | 18min | 1 tasks | 5 files |
| Phase 09 P04 | 6min | 1 tasks | 6 files |
| Phase 09 P05 | 4min | 1 tasks | 5 files |
| Phase 09 P06 | 2min | 2 tasks | 7 files |
| Phase 09 P07 | 420 | 2 tasks | 7 files |
| Phase 09 P09 | 2m | 2 tasks | 8 files |
| Phase 09 P10 | 15m | 2 tasks | 4 files |
| Phase 09 P11 | 2min | 1 tasks | 2 files |
| Phase 10 P01 | 3 min | 2 tasks | 4 files |
| Phase 10 P03 | 9 min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- `BlobStore` is the canonical payload-plus-metadata lifecycle owner; `UnifiedCache` owns cache policy only.
- `SqlCache` remains a separate subsystem. Pre-production public APIs may be replaced by one coherent BlobStore/cache surface rather than retained through compatibility adapters.
- V1 supports same-backend migration plus an explicit rebuild for incompatible or cross-backend data.
- Direct `BlobStore` integrity failures are typed exceptions; `UnifiedCache` may translate them into separately recorded misses.
- AWS S3 semantics are authoritative; compatible services are supported only where explicitly verified.
- [Phase 07]: Migration and rebuild are explicit stopped-worker maintenance operations; ordinary opens validate and never upgrade implicitly.
- [Phase 07]: Each release supports its current and immediately previous released layout; older releases advance through declared steps.
- [Phase 07]: Recovery uses bounded authenticated authority evidence and exact receipts; unexplained payloads remain invisible and are never adopted.
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

The following Phase 03 bullets are preserved as superseded file-native implementation evidence from the archived 28-cycle attempt; they are historical context, not claims about replacement-plan execution or the target architecture.

- [Phase 03 superseded]: BlobStore publication now uses immutable generation locators and exact manifest CAS as its sole authority transition.
- [Phase 03 superseded]: Reopen recovery acts only on authenticated manifest authority plus validated signed operation evidence; normal reads do not mutate evidence.
- [Phase 03 superseded]: Manifest authority mutations require the authenticated generation plus SHA-256 of the exact canonical record; repository adapters compare opaque bytes only.
- [Phase 03 superseded]: JSON conditional publication refreshes under a short OS-backed lock, while SQLite takes a writer transaction before exact-record comparison.
- [Phase 03 superseded]: LifecycleLimits is declared once in cacheness.config and is passed by identity from CacheConfig through BlobStore, LifecycleEngine, and the operation repository.
- [Phase 03 superseded]: Operation evidence uses a dedicated HMAC domain and exact-byte conditional checkpoint and retirement; grace never authorizes unauthenticated evidence.
- [Phase 03 superseded]: Stale write cleanup reclaims only the operation-bound immutable candidate after exact CAS conflict.
- [Phase 03 superseded]: Delete publishes a signed tombstone before payload reclamation and retires it with exact-record CAS.
- [Phase 03 superseded]: StoreAdmissionBarrier holds aggregate admission only while clear persists its authenticated finite target snapshot; ordinary operations are otherwise concurrent.
- [Phase 03 superseded]: Clear checkpoints exact authenticated target pages and treats changed current records as conflicts, never as authority to delete a later generation.
- [Phase 03 superseded]: Legacy clear evidence is reopen-only; new BlobStore clears are owned exclusively by LifecycleEngine.
- [Phase 03 superseded]: BlobStore ordinary operations bypass predecessor global admission and use StoreAdmissionBarrier only for current clear snapshot establishment.
- [Phase 03 superseded]: Reconciliation defaults to bounded dry-run reports and only applies authenticated, revalidated exact evidence.
- [Phase 03 superseded]: Reconciliation resumes with encrypted authenticated independent manifest and operation cursors.
- [Phase 03 superseded]: Key locks are per BlobStore instance; independent instances remain governed solely by exact manifest CAS.
- [Phase 03 superseded]: BlobStore reads authenticate M1 and M2 around each private snapshot and retry only a proven newer generation once.
- [Phase 03 superseded]: Concurrent close waiters treat CLOSED as terminal and never repeat owned resource release.
- [Phase 03 superseded]: Phase 3 release tests assert canonical lifecycle and exact-CAS authority; legacy metadata projections and clear-recovery hooks are not direct BlobStore mutation authority.

Current replacement decisions are D-23 through D-31 in `03-CONTEXT.md`: one transactional `LifecycleAuthority`, stdlib SQLite as the local persistent adapter, JSON as projection, immutable native payloads outside transactions, and physical retirement of the superseded scheduler.

- [Phase 03]: Confirmed lifecycle authority identity: .cacheness/lifecycle-authority-v1.sqlite3; SQLite application ID 0x43414348; user_version 1; generated store identity. — Blocking-human checkpoint confirmed the one-way local authority identity before Plan 03-02 creates the first database.
- [Phase 03]: SQLite authority intent precedes native payload publication; verified promotion is the sole visibility transition.
- [Phase 03]: Authority-composed BlobStore inspection is lazy and rejects established stores missing authority unchanged.
- [Phase 03]: SQLite now verifies DELETE+EXTRA, trusted-schema, the confirmed authority identity, and one absolute busy deadline before lifecycle mutations.
- [Phase 03]: Every complete authority transition changes entries, operation state, debt, projection revision, and authority revision atomically; uncertain commits reopen and classify the exact operation.
- [Phase 03]: Windows lifecycle mutation is limited to an offline-provisioned local root whose protected DACL grants ordinary mutation solely to the current token logon SID; native proof is deferred to Plan 09.
- [Phase 03]: LifecycleAuthority is the only committed-state authority; legacy manifest repositories are not initialized in authority mode.
- [Phase 03]: Promotion records exact cleanup debt atomically, while reclamation and debt retirement run outside authority transactions.
- [Phase 03]: Default BlobStore close owns and releases only the authority it constructs; injected resources stay caller-owned.
- [Phase 03]: Clear targets retain exact authenticated entry bytes and lineage, then advance only after exact deletion, absence proof, conflict, or blocked evidence.
- [Phase 03]: Reconciliation v2 is the canonical authority report; zero-argument v1 dictionaries and summaries remain pure projections from the same findings.
- [Phase 03]: JSON is a rebuildable, revision-tagged projection only; LifecycleAuthority remains the sole committed-state truth.
- [Phase 03]: Authority selection validates semantic capabilities before construction; memory is explicit same-process-only and injected authority ownership is preserved.
- [Phase 03]: Retired development controls use exact name/type classification and fail rebuild-required before authority bootstrap or mutation.
- [Phase 03]: LifecycleAuthority is the sole runtime reconciliation and recovery authority; the file-native reconciler has no compatibility path.
- [Phase 03]: BlobStore selects one LifecycleAuthority; JSON is revision-bound projection only.
- [Phase 03]: Retained coordination is process-local ordering and close admission only; payload safety stays in filesystem helpers.
- [Phase 03]: Non-Windows platform evidence remains canonical UNAVAILABLE with exit code 2 and is not Windows qualification.
- [Phase 03]: Native Windows denial proof requires a genuinely different session or service token; same-session evidence alone is insufficient.
- [Phase 03]: NumPy is a declared base runtime dependency because normal package import paths require it.
- [Phase 03]: Plan 03-12 keeps the Plan 03-09 repository runner immutable and designates a separate non-overridable native-qualification argv.
- [Phase 03]: UNAVAILABLE exit 2 remains NOT_QUALIFIED with native_evidence false; Phase 999.1 requires native PASS exit 0.
- [Phase 03]: D-32 qualification captures only fixed-command UNAVAILABLE evidence; Phase 999.1 retains native PASS/exit 0 release qualification.
- [Phase 03]: Phase 3 final acceptance requires a fresh frozen isolated all-extras/dev suite; nested uv tooling must not mutate the pytest environment.
- [Phase 03]: Darwin remains UNAVAILABLE/NOT_QUALIFIED with native_evidence false; Phase 999.1 requires native Windows PASS/exit 0.
- [Phase 03]: Projection metadata is derived compatibility state; only authenticated committed authority manifests can make facade rows live. — Prevents stale or tombstoned projections from participating in cache policy or recovery.
- [Phase 03]: Projection mutation and custom-link binding require exact generation-specific locator tokens; mismatches change no row or link. — Preserves a concurrently promoted generation while allowing idempotent same-token repair.
- [Phase 03]: Clear active ownership spans full cleanup while the snapshot admission gate lasts only through begin_clear.
- [Phase 03]: Projection and custom-link replacement require the operation's exact promoted authority lineage and captured M1 locator.
- [Phase 03]: Canonical empty state uses BlobStore authority operations; legacy cleanup requires explicit recognized composition.
- [Phase 03 superseded by ADR 0001]: Authority-path FIFO admission and the fixed 0.187-second universal-success target were implementation experiments, not supported correctness guarantees.
- [Phase 03 ADR 0001 replan]: The supported durable topology is SQLite LifecycleAuthority plus local filesystem blobs on one host with multiple processes; memory is same-process-only, and PostgreSQL/multihost or broad topology composition remains later-phase work.
- [Phase 03 ADR 0001 replan]: SQLite is the sole transactional lifecycle authority; immutable filesystem payload effects are coordinated through durable intent, exact promotion, cleanup debt, and deterministic reconciliation rather than cross-resource ACID.
- [Phase 03 ADR 0001 replan]: Supported contention may return a stable typed retryable timeout. Runtime timeout is configurable operational policy; 0.187-derived thresholds remain benchmark/regression evidence only.
- [Phase 03 ADR 0001 replan]: Do not add another lock, FIFO queue, sidecar, lifecycle truth, or timeout-stage patch to force universal success; SQLite/CAS remains the correctness boundary, and only optional demonstrably justified process-local optimization may remain.
- [Phase 03]: SQLite WAL and maintenance bootstrap before pooled sessions are published; pool checkout stays read-safe.
- [Phase 03]: Aggregate metadata queries omit mismatched projection rows and bind exact key/locator pairs instead of repairing or matching keys alone.
- [Phase 03 superseded by ADR 0001]: Named stress-schedule timeout-stage allowlists are historical evidence; Plan 03-20 replaces them with success/conflict/typed-retryable-timeout progress accounting plus independent safety assertions.
- [Phase 03]: SQLite contention is a bounded typed outcome: success, exact conflict, or retryable timeout.
- [Phase 03]: The 5.0-second authority default is caller policy; 0.187-second evidence remains benchmark-only.
- [Phase 04]: Phase 4 Ruff debt is frozen from the declared plan inventory; scope drift, changed findings, and dirty new Python files fail the gate.
- [Phase 04]: Wave 0 contracts use assertion-level import gates so absent clean APIs fail red without collection errors.
- [Phase 04]: Catalog defaults materialize only on new writes; portable predicates inspect stored presence rather than effective defaults.
- [Phase 04]: Format 2 keeps store epoch, manifest schema, SQLite user version, and payload format independently versioned; unsupported layouts fail read-only.
- [Phase 04]: BlobReceipt is the frozen public semantic result; BlobEntryInfo is not re-exported as a compatibility alias.
- [Phase 04]: StoreTopology is the sole direct BlobStore composition root; injected participants retain exact identity and default caller ownership.
- [Phase 04]: Memory topology provides only same-process ephemeral behavior, with canonical scan but no durability or index acceleration.
- [Phase 04]: Capability minima are checked before named factory construction or participant I/O.
- [Phase 04]: SQLite format-2 schema version remains independent from store format; repeat initialization is validation-only.
- [Phase 04]: Canonical signed descriptors are the only catalog state; authenticated scans use revision-bound HMAC cursors without acceleration indexes.
- [Phase 04]: Projection sinks are derived-only and cannot authorize lifecycle or canonical query completeness.
- [Phase 04]: Projection checkpoints bind source, epoch, schema, query, revision, and cursor; apply precedes checkpoint.
- [Phase 04]: Projection failure preserves the committed BlobReceipt; explicit refresh reports typed committed partial.
- [Phase 04]: SQLite projection rebuild requires explicit stopped-worker offline maintenance.
- [Phase 04]: Direct BlobStore regression fixtures construct StoreTopology explicitly instead of selecting a backend.
- [Phase 04]: Current lifecycle assertions use signed format-2 BlobManifest descriptors and BlobReceipt, not BlobEntryInfo or repository shapes.
- [Phase 04]: Mixed-scope UnifiedCache policy tests retain their outcomes while only retired metadata-selector setup changes.
- [Phase 04]: JSON, PostgreSQL, and former ORM read models are tested solely as derived ProjectionSink consumers.
- [Phase 04]: Retired metadata authority exports have explicit absence assertions for the atomic Plan 04-08 cutover.
- [Phase 04]: BlobStore is the sole payload/catalog lifecycle authority; UnifiedCache remains a narrow policy facade.
- [Phase 04]: Pre-production metadata authority compatibility surfaces are deleted rather than shimmed.
- [Phase 04]: Python 3.11/3.13 full-suite collection evidence is recorded as a release gap, without reopening retired APIs.
- [Phase 04]: StoreTopology owns exactly one RoleRegistry; BlobStore resolves only that composition root.
- [Phase 04]: Structural protocol checks and a local identity ledger replace concrete cross-role checks and duplicated close paths.
- [Phase 04]: Projection boundaries translate ordinary external exceptions into receipt-preserving derived outcomes; BaseException remains visible.
- [Phase 04]: Catalog cursors enforce closed-envelope byte and field limits before decode, JSON, HMAC, manifest loading, or authority dispatch.
- [Phase 04]: RoleRegistry is the sole registry consumer contract; retired process-global blob selectors remain absent.
- [Phase 04]: Mocked S3 registration proves only local factory construction and option forwarding; Phase 5 owns topology qualification.
- [Phase 04]: Phase 4 release evidence derives PHASE4_MATRIX only from a 41-path marker-bounded owned list; deferred pandas SQL-cache collection is diagnostic-only and non-green.
- [Phase 04]: Ruff qualification inventory may add only individually clean declared Plan 04 paths; frozen existing findings and fingerprints remain unchanged.
- [Phase 06]: Cache policy maps only declared BlobStore read failures to public outcomes and preserves the original typed cause.
- [Phase 06]: CacheStatistics is a frozen derived observer over six outcomes and performs no catalog or lifecycle work.
- [Phase 06]: Cache removal reports preserve exact-generation conflict and stale-cursor retryability without cache-side lifecycle coordination.
- [Phase 06]: Cache policy catalog entries persist authenticated namespace/prefix facts and exact expectations.
- [Phase 06]: CachePutResult holds one immutable BlobReceipt plus one bounded CacheMaintenanceResult, so post-commit policy work cannot rewrite committed storage truth.
- [Phase 06]: UnifiedCache.put runs exactly one size-maintenance step; callers explicitly resume validated opaque continuation state.
- [Phase 06]: The pre-production put-result cutover exposes cache keys through receipt.key rather than retaining a string-return compatibility layer.
- [Phase 06]: UnifiedCache owns qualified function namespaces and normalized function keys; BlobStore persists the namespace as canonical catalog data.
- [Phase 06]: The explicit cached decorator recomputes only absent and expired by default, preserving typed failure outcomes unless callers opt in.
- [Phase 06]: Function cache_clear delegates to bounded exact-generation invalidation and returns the canonical CacheRemovalReport.
- [Phase 06]: UnifiedCache lifecycle ownership follows constructor form: injected BlobStores are caller-owned while StoreTopology creates the one cache-owned store.
- [Phase 06]: Closing UnifiedCache rejects cache policy observers without delegating close to caller-owned BlobStores or coordinating topology participants.
- [Phase 06]: Removed cache singleton/factory/raw-result compatibility paths; callers use explicit UnifiedCache ownership.
- [Phase 06]: Unsupported store/version layouts remain typed offline migration-or-rebuild failures; no implicit upgrade.
- [Phase 06]: Examples use the qualified memory/memory topology so they make no remote-service claim.
- [Phase 06]: Decorator examples preserve typed failures by default and expose explicit opt-in recomputation with cache_last_lookup.
- [Phase 06]: Phase 6 fixed verification uses a root-safe fixed manifest plus narrow AST checks; missing pandas leaves CACH-07 and the full suite open without treating live PostgreSQL/S3 or Windows evidence as passed.
- [Phase 06]: Phase 06 Plan 09: Persistent signing and object-array tests use the supported sqlite-filesystem topology.
- [Phase 06]: Phase 06 Plan 09: Object-array tampering is a typed non-destructive cache outcome that never reaches ObjectHandler.
- [Phase 06]: Retained metadata-query behavior is exercised only through typed CatalogQuery pages, never a cache metadata facade.
- [Phase 06]: The Phase 1 interpolation sentinel names current catalog functions so a deleted legacy function cannot pass the gate.
- [Phase 06]: Phase 06 local verification excludes exactly three named Phase 8 live PostgreSQL/S3 modules; mocks and skips are never qualification evidence.
- [Phase 06]: The Phase 6 verifier uses a fixed Plan 09-11 inventory and AST checks that permit only structural TypeError or absence negatives.
- [Phase 06]: The locked all-extras local gate closes CACH-07 and Phase 6 local-suite evidence; BACK-05 and native Windows remain unqualified for Phase 8.
- [Phase 07]: The tracer uses a test-only current-to-current compatibility edge, so it proves the generic path without manufacturing a production format version.
- [Phase 07]: Maintenance evidence corroborates one explicit offline run, while authority activation alone selects visible store state.
- [Phase 07]: UnifiedCache preserves an unsupported-store failure as a typed lookup cause rather than adopting or changing the store.
- [Phase 07]: D-02 accepted via proceed D-02: each release supports current and immediately previous released layouts directly; older releases advance through declared steps.
- [Phase 07]: D-08 accepted via proceed D-08: rebuild excludes nothing by default; each exact exclusion requires a newly generated and reconfirmed plan.
- [Phase 07]: D-15 accepted via proceed D-15: finalize ends rollback; separately confirmed idempotent purge may remove the retained prior copy and leaves retryable cleanup debt on failure.
- [Phase 07]: D-02 is enforced by a bounded current-plus-immediately-previous ReleaseWindow with no development-layout edge.
- [Phase 07]: Migration plans are canonical bounded JSON; human reports render only validated plan state.
- [Phase 07]: Administrative inventory returns raw EntrySnapshot pages bound to store identity, revision, and key/generation continuation; manifests are authenticated only by maintenance.
- [Phase 07]: PostgreSQL inventory reports its persisted capability/schema and retains typed retryable progress causes; deterministic contract coverage is not live-service qualification.
- [Phase 07]: Maintenance evidence remains non-authoritative; exact authenticated run evidence only corroborates offline work.
- [Phase 07]: Resume accepts only an exact run ID and evidence path, then revalidates recorded outputs without latest-run discovery or candidate adoption.
- [Phase 07]: [Phase 07]: RQ-01 accepted via proceed RQ-01: SQLite authority schema 8 and PostgreSQL capability/schema 4 are the first release baseline; one lifecycle authority owns candidate/activated_offline/active/rolled_back and exact candidate/prior rows; activated_offline blocks every ordinary worker entry point until rollback or finalize; authority ACID stops at authority state while payload/evidence are immutable attributed external effects; schemas 7/3 have no compatibility promise.
- [Phase 07]: PostgreSQL transactions end at authority state; S3 candidate receipts corroborate immutable effects and never select visibility.
- [Phase 07]: S3 candidate locators are exact run-owned receipts; listings and ETags never authorize discovery or adoption.
- [Phase 07]: Rollback and finalization consume authenticated activated evidence; finalization is replay-safe only with the same exact confirmation.
- [Phase 07]: Purge confirmation binds exact retained-prior identities, locators, manifest digests, counts, and bytes.
- [Phase 07]: Purge failures remain evidence-backed cleanup debt and never change the active candidate selection.
- [Phase 07]: Rebuild scope defaults to all inspected keys; exclusions are exact regenerated and separately confirmed plan inputs.
- [Phase 07]: Registered store-local handlers own payload transformations; the migration coordinator has no native-format switch.
- [Phase 07]: Rebuild writes remain in BlobStore lifecycle ownership; accepted rebuild evidence only unlocks derived projections.
- [Phase 07]: Phase 07: Published offline migration and rebuild only through cacheness.storage; no CLI, global service, or implicit ordinary-open switch.
- [Phase 07]: Phase 07: Recorded the public-API detector false positive verbatim and retained Phase 8 ownership of live PostgreSQL and AWS S3 qualification.
- [Phase 07]: Phase 7 final verification uses a fixed literal inventory; live PostgreSQL and AWS S3 remain NOT RUN / NOT QUALIFIED Phase 8 work.
- [Phase 07]: The observed clear/delete concurrency conflict remains pre-existing evidence; no lifecycle coordination or storage race patch was added.
- [Phase 07]: Migration resume and abort recover only exact authority-attributed candidate batches; pre-checkpoint immutable payloads remain invisible, unadopted, and outside guaranteed exact cleanup.
- [Phase 07]: A pre-identity idle observation is only an early fast check; persisted PostgreSQL state must authorize worker readiness after identity load.
- [Phase 07]: BlobStore initialization and direct PostgreSQL preflight reuse require_ordinary_worker_access instead of adding local coordination or a second authority.
- [Phase 07]: Maintenance operation retries read only an exact canonical authority record; paths, listings, timing, and caller receipts never establish lifecycle ownership or completion.
- [Phase 07]: Projection suppression is internal to the maintenance canonical-put receipt path; ordinary BlobStore writes retain normal post-commit projection behavior.
- [Phase 07]: Prepared replay accepts only the authority-indexed immutable locator when signed descriptor digest and size corroborate it; mismatches fail closed.
- [Phase 07]: Rebuild retries derive deterministic operation IDs and accept completion only from the exact existing lifecycle-authority replay; bounded evidence stores projection-free BlobReceipt identities, never a second intent.
- [Phase 07]: Rebuild cleanup deletes only exact receipt-matching generations; changed ownership and operational deletion failures remain explicit rebuild cleanup debt.
- [Phase 07]: Migration compatibility requires exact source and configured destination contract matching; source-only matches are rebuild-only.
- [Phase 07]: Only authority-attributed transformed candidates resume or abort; pre-checkpoint immutable orphans remain invisible and unadopted without an exact-cleanup guarantee.
- [Phase 07]: Canonical migration plans bind source catalog and manifests by digest without serializing raw source data.
- [Phase 07]: Migration and rebuild execution re-authenticate live source state before candidate mutation and reject source_state_drift.
- [Phase 07]: Exact target descriptors remain only in existing bounded authority-owned candidate evidence; pre-checkpoint orphans stay invisible and unadopted.
- [Phase 07]: Phase 7 fixed claims use AST-validated exact pytest selectors before execution.
- [Phase 07]: Plans 07-12 through 07-19 threat rows are checked against a literal 38-ID ownership oracle.
- [Phase 07]: Phase 7 validation accepts only authority-attributed recovery; invisible pre-checkpoint orphans remain outside exact reclamation.
- [Phase 07]: Phase 7 all-mode verification is deterministic local evidence; live service, platform, performance, and obstore adoption claims remain Phase 8 or later.
- [Phase 07]: Phase 07 Plan 20: The destination handler declares the migration target format and version; source readability only establishes whether the source is readable.
- [Phase 07]: Phase 07 Plan 20: Abort records only OSError and CacheBlobBackendError as attributed cleanup debt; integrity and ownership disagreement remain fail closed.
- [Phase 07]: Cleanup debt is retired only after an exact recorded receipt and existing authority replay corroborate deletion or absence.
- [Phase 07]: A later logical-key owner is preserved; settlement addresses only the recorded immutable locator.
- [Phase 07]: Terminal ABORTED rebuild evidence requires empty debt and retirement of every recorded receipt.
- [Phase 07]: Phase 7 verification uses literal Plans 01-22 and a 50-gap/96-total threat oracle with AST-validated exact selectors.
- [Phase 07]: WR-02 name= handling is deferred as a standalone handler-registration API-contract decision, not a migration claim.
- [Phase 07]: Authenticated rebuild cleanup debt fences direct stage, verify, and accept operations; explicit resume remains the only settlement dispatcher.
- [Phase 07]: Only REBUILDING and REBUILD_VERIFYING evidence may enter exact receipt cleanup; accepted evidence with debt is invalid at the model boundary.
- [Phase 07]: T-07-21-03 now requires ordered forged-debt plus forward-fence/resume selector evidence.
- [Phase 07.1]: Phase 07.1 Plan 01: User approved exactly obstore 0.11.1; no fork, custom signer, boto3 fallback, or owner-pinning restoration is authorized.
- [Phase 07.1]: Obstore 0.11.1 is pinned behind an executable all-store SDK parity contract before production integration.
- [Phase 07.1]: D-16 keeps ExpectedBucketOwner unsupported; only test-only loopback moto endpoint overrides are qualified.
- [Phase 07.1]: Phase 07.1 Plan 03: ambiguous obstore create responses settle only from exact head plus bounded SHA-256/size evidence; AuthorityLifecycleEngine remains the visibility authority.
- [Phase 07.1]: Phase 07.1 Plan 03: LocalStore and MemoryStore share one guarded five-method participant; custom handlers retain only private suffix-preserving Paths.
- [Phase 07.1]: D-16 rejects expected_bucket_owner before S3Store construction; no signer, fork, boto3 production fallback, or parallel participant restores it.
- [Phase 07.1]: S3 publication remains one direct conditional create with multipart disabled and a 128 MiB default transfer cap.
- [Phase 07.1]: Exact head and bounded inventory observations are report-only transport evidence; AuthorityLifecycleEngine remains the only visibility authority.
- [Phase 07.1]: ETag and object version remain opaque signed observations; canonical SHA-256 plus size remains the integrity decision.
- [Phase 07.1]: Catalog and user metadata changes preserve generation and evidence through one authority-only CAS.
- [Phase 07.1]: SQLite and PostgreSQL implement the same narrow metadata CAS to remain LifecycleAuthority conformant.
- [Phase 07.1]: SQLite schema 9 and PostgreSQL schema/capability 5 are explicit cutovers; prior development layouts require stopped-worker migration or rebuild.
- [Phase 07.1]: Transport evidence remains opaque, optional authority state and is copied only with its immutable generation identity.
- [Phase 07.1]: Metadata replacement remains one authority CAS and never invokes a payload participant or claims cross-resource ACID.
- [Phase 07.1]: Transport matches are read-only opaque corroboration and cannot claim canonical SHA-256 verification or lifecycle authority.
- [Phase 07.1]: Optional structural transport observation keeps injected providers without head support selected and reports UNAVAILABLE.
- [Phase 07.1]: Only exact typed missing-object errors normalize to ABSENT; other transport failures retain typed backend causes.
- [Phase 07.1]: Phase 07.1 Plan 08: Candidate transport evidence is fresh destination-local corroboration persisted only through an exact authority candidate-verification transition; it does not select lifecycle visibility.
- [Phase 07.1]: Phase 07.1 Plan 08: Reconciliation uses only the selected participant's bounded report-only inventory cursor; listing cannot authorize cleanup or adoption.
- [Phase 07.1]: Built-in memory, filesystem, and S3 payload factories now materialize only ObstoreGenerationIO; no selector, legacy read path, boto3 escape hatch, or compatibility path remains.
- [Phase 07.1]: When a selected payload provider materializes itself as guarded handler I/O, StoreTopology's ownership ledger closes it exactly once; caller-injected providers remain caller-owned.
- [Phase 07.1]: UnifiedCache keeps CACH-03 policy-only removal: TTL, eviction, predicate, decorator, single-key, and global clear delegate exact deletion through BlobStore.
- [Phase 07.1]: Removed legacy payload modules outright; no aliases, fallback readers, runtime selector, or second lifecycle authority remains.
- [Phase 07.1]: ObstoreGenerationIO is the deliberate public payload-participant export; PostgreSQL remains conditional on its optional dependency.
- [Phase 07.1]: Mocked S3 tests use boto3 only to provision Moto buckets, while live AWS/PostgreSQL modules use native obstore credentials and remain Phase 8 collection-only evidence.
- [Phase 07.1]: Runtime S3/cloud extras contain no boto3; moto remains test-only qualification tooling.
- [Phase 07.1]: D-16 uses explicit bucket/region, stable-name/IAM/policy controls and no owner pinning or production endpoint override.
- [Phase 07.1]: Phase 07.1 evidence is fixed and fail-closed; Phase 8 gates remain explicitly unqualified.
- [Phase 08]: Deterministic PASS evidences integrity, recovery, and progress only; performance remains NOT_QUALIFIED.
- [Phase 08]: A passing deterministic gate exits 2 while external evidence classes are UNAVAILABLE, preventing a partial release pass.
- [Phase 08]: Later release tooling must revalidate both the exact Git revision and reviewed-source SHA-256 digest.
- [Phase 08]: Wheel probes freeze literal public exports and reject source-tree imports.
- [Phase 08]: Each optional group uses a fresh wheel environment; service-labelled extras remain non-live.
- [Phase 08]: TensorFlow packaging evidence is UNAVAILABLE outside reviewed stable minors, never a skip-based pass.
- [Phase 08]: Phase 08 Plan 03: Linux 3.11–3.14 is the only full core qualification matrix; Python 3.15 remains advisory.
- [Phase 08]: Phase 08 Plan 03: macOS records only 3.11 and 3.14 boundary smoke; Windows stays an UNAVAILABLE Phase 999.1 nonclaim.
- [Phase 08]: Phase 08 Plan 03: success, conflict, and typed-retryable contention remain valid ADR progress classifications, not portability or coordination failures.
- [Phase 08]: PostgreSQL DB-API boundary coverage uses deterministic transcript fixtures; live PostgreSQL remains separate qualification evidence.
- [Phase 08]: Phase 8 coverage baseline capture follows named lifecycle and cache-policy contracts; QUAL-05 remains pending for Plan 08-05.
- [Phase 08]: Phase 08 Plan 06: structural call and RSS evidence remains separate from controlled timing performance evidence.
- [Phase 08]: Phase 08 Plan 06: unsupported POSIX RSS measurement fails closed instead of claiming normalized memory evidence.
- [Phase 08]: Phase 08 Plan 07: Canonical performance evidence measures native NPZ current writes; legacy Blosc2 remains input-only compatibility and is not fabricated as a current write path.
- [Phase 08]: Phase 08 Plan 07: Only cacheness-perf-linux-x64 may verify reviewed p50/p99 envelopes; remote/macOS timing stays diagnostic and benchmark envelopes never become runtime deadlines.
- [Phase 08]: Phase 08 Plan 07: SHA-256 plus size remains canonical persisted integrity; XXH3 is comparative evidence only.
- [Phase 08]: Phase 08 Plan 08: only release_candidate live evidence can be QUALIFIED; scheduled diagnostics remain non-qualifying.
- [Phase 08]: Phase 08 Plan 08: Phase 8 qualification uses exact q8 marker-owned namespaces while preserving Phase 5 fixture compatibility.
- [Phase 08]: Phase 08 Plan 08: real-service source identity includes fixed live tests, active obstore/authority code, tools, and workflows.
- [Phase 08]: Coverage floors compare covered and total statement/branch counts plus derived rates for repository and critical scopes.
- [Phase 08]: Coverage capture excludes all protected live-service markers; mocks, skips, and unavailable services never become baseline evidence.
- [Phase 08]: The baseline is an explicit canonical JSON artifact; ordinary verification never mutates it.
- [Phase 08]: Phase 08 Plan 09 keeps core quality evidence separate from TensorFlow-compatible wheel qualification so optional dependency availability cannot downgrade core support evidence.
- [Phase 08]: Phase 08 Plan 09 qualifies retained TensorFlow support only on Python 3.11 and 3.12, matching the package probe's declared compatible range.
- [Phase 08]: Phase 08 Plan 10: exact-SHA release collection records fresh run IDs and fixed artifact names; latest-run selection is prohibited.
- [Phase 08 historical before D-24]: Plan 08-10 kept bounded UNAVAILABLE packaging/platform envelopes as explicit release-blocking nonclaims; D-24 now preserves those nonclaims while closing only local readiness.
- [Phase 08]: Phase 08 Plan 13: Controlled-runner preflight emits bounded eligibility only; its original Plan 08-11 performance handoff is superseded by D-23 and retained for SEED-006.
- [Phase 08]: Phase 08 Plan 13: Only cacheness-perf-linux-x64 plus clean detached exact SHA and canonical allow-listed digest can qualify capture.
- [Phase 08 historical before D-24]: D-23 superseded only the then-current release-blocking controlled-Linux portions of D-20/D-22; QUAL-06 remains DEFERRED/NOT_QUALIFIED under SEED-006, and D-24 subsequently narrowed current closure to local readiness.
- [Phase 08]: macOS performance evidence remains diagnostic and establishes neither Linux equivalence nor a cross-platform budget.
- [Phase 08 historical before D-24]: Plan 08-14 revised release collection/aggregation/fixed verification before exact-SHA live evidence collection; D-24 later deferred BACK-05/publication to SEED-007.
- [Phase 08]: D-23 defers only QUAL-06 controlled Linux performance to SEED-006; macOS remains diagnostic-only.
- [Phase 08]: Immutable publication requires an approval-bound prepublication digest and exact remote asset verification.
- [Phase 08]: Phase 08 Plan 15: Protected-live preflight is configuration-only; every result is service_state NOT_RUN and cannot qualify BACK-05.
- [Phase 08]: Phase 08 Plan 15: Fixed verifier inventory now binds Plan 08-15 and T-08-15-01 through T-08-15-05 to literal preflight selectors.
- [Phase 08]: D-24 closes the milestone on exact-source local readiness; Plans 08-11/08-12 are superseded and preserved for SEED-007, BACK-05 remains DEFERRED/NOT_QUALIFIED, and publication remains DEFERRED/NOT_PUBLISHED.
- [Phase 08]: The exact-snapshot clear/delete regression accepts ordinary completion or `CacheBlobLifecycleConflictError` as the two ADR-valid progress outcomes while requiring final absence, no authority entry, no cleanup debt, and bounded completion in both cases; no lifecycle fix is authorized.
- [Phase 08]: Exact-snapshot clear/delete tests accept ordinary completion or CacheBlobLifecycleConflictError while requiring identical final safety and recovery assertions.
- [Phase 08]: SQLite shared-worker qualification initializes the authority before releasing independent operations; concurrent first creation remains outside the availability guarantee, while exact application identity and bounded error-free worker completion remain required.
- [Phase 08]: Phase 08 Plan 18: SQLite shared-worker tests initialize explicitly before release; concurrent first creation remains outside the availability contract.
- [Phase 08]: Phase 08 Plan 18: Worker diagnostics must equal SQLITE_APPLICATION_ID and the initializer store identity, not merely agree with each other.
- [Phase 08]: The post-08-18 coverage deficit is a tests-only validation gap: Plan 08-19 exercises exact SQLite configuration, deadline, error-translation, identity, and schema rejection paths without lowering the baseline or changing lifecycle code.
- [Phase 08]: Phase 08 Plan 19 restores QUAL-05 only through meaningful SQLite validation and fail-closed corruption tests; no lifecycle, baseline, or concurrency semantics changed.
- [Phase 08]: Canonical JSON local-readiness evidence validates against the exact key set and cardinality, not mapping iteration order.
- [Phase 09]: Alias-free FormatHandler cutover preserves durable handler/payload identities and the existing store-local registry seam.
- [Phase 09]: Phase 09 Plan 02: Wheel qualification captures only package import output, preserving normal application output while rejecting import-time optional-capability noise.
- [Phase 09]: Phase 09 Plan 02: Persisted format identity is verified through authenticated authority manifests; public BlobEntry inspection remains metadata-only.
- [Phase 09]: Phase 09 canonical examples are literal published files exercised twice with child-only socket blocking and no residue.
- [Phase 09]: The MCAP example demonstrates stable handler and payload identities through store-local FormatHandler registration without exposing managed paths.
- [Phase 09]: Phase 09 Plan 04 deleted the first obsolete non-SqlCache example batch once canonical executable journeys covered the supported workflows.
- [Phase 09]: Phase 9 removed five obsolete non-SqlCache examples, including runnable S3 promotion; canonical local examples remain the supported executable surface and Phase 10 retains SqlCache ownership.
- [Phase 09]: Canonical public examples run only through one exact blocking CI harness on the stable local row.
- [Phase 09]: The Phase 9 examples index promotes only four qualified local journeys; SqlCache assets remain Phase 10-owned.
- [Phase 09]: Phase 09 Plan 07: README is a concise local-ready BlobStore-first gateway with exactly two explicit quick starts; detailed qualification remains linked rather than duplicated.
- [Phase 09]: Phase 09 Plan 07: SQLite plus filesystem guidance promises crash consistency through immutable generations and reconciliation, not cross-resource ACID.
- [Phase 09]: Phase 09 Plan 08: RELEASE_QUALIFICATION.md is the sole detailed evidence matrix; local readiness leaves S3/PostgreSQL, Windows, controlled Linux performance, and immutable publication explicitly nonpassing.
- [Phase 09]: Phase 09 Plan 08: SHA-256 plus size remains canonical integrity while ETag/version is opaque corroboration; format handlers use contained private staging and fail-closed parsing boundaries.
- [Phase 09]: Phase 09 Plan 09: public API reference is limited to current barrels and delegates qualification claims to their single owner.
- [Phase 09]: Phase 09 Plan 09: generic format extensions use persisted payload identities and store-local FormatHandler registration, not a global lifecycle API.
- [Phase 09]: Phase 3 historical verifier gaps are superseded by named scoped-local closure evidence; do not reopen lifecycle coordination.
- [Phase 09]: Phase 8 LOCAL_READY completes only local Wave 0/Nyquist evidence; remote, platform, performance, and publication nonclaims remain explicit.
- [Phase 09]: Narwhals remains a dormant future dataframe compatibility investigation; native handlers retain Parquet and persisted format identities.
- [Phase 09]: Migration guidance routes mutable qualification status to Release qualification; Phase 8 remains local-readiness evidence while SEED-006, SEED-007, and Phase 999.1 own deferred work.
- [Phase 10]: Phase 10 removal contracts require natural Python absence and preserve only the BlobStore/UnifiedCache public boundary.
- [Phase 10]: Use one digest-bound WheelArtifact for ZIP inspection, source-free installation, metadata checks, and retained local round trips. — A single artifact prevents checkout-only or parallel-harness evidence from masking stale shipped surface.
- [Phase 10]: Direct deletion, not a tombstone or compatibility alias, is the supported SqlCache cutover; ordinary Python import failure is intentional.
- [Phase 10]: SqlCache-only DuckDB and `sql` extras are removed while SQLAlchemy/PostgreSQL lifecycle authorities and dataframe/Parquet dependencies remain supported.
- [Phase 10]: The cutover did not alter lifecycle, concurrency, recovery, topology, or handler-persistence contracts; ADR 0001 remains the guardrail.

### Pending Todos

- Run the v1.0 milestone audit before archiving; preserve the explicit SEED-006, SEED-007, and native-Windows nonclaims.

### Blockers/Concerns

- QUAL-06 controlled-Linux thresholds remain unqualified until SEED-006; no macOS or diagnostic measurement may substitute.
- PostgreSQL/AWS S3 BACK-05 and immutable publication remain `DEFERRED`/`NOT_QUALIFIED` or `NOT_PUBLISHED` until SEED-007; native Windows remains unqualified until Phase 999.1. These are nonclaims, not Phase 8 blockers after D-24.

### Roadmap Evolution

- Phase 5 edited: Moved BACK-05 non-substitutable PostgreSQL/Amazon-S3 real-service qualification gate intact to Phase 8; Phase 5 retains the candidate implementation, deterministic contracts, frozen live suites, fail-closed runner, and truthful UNAVAILABLE evidence without a release support claim.
- Phase 07.1 inserted after Phase 7: Obstore Payload Participant Unification (URGENT)
- Phase 8 edited: edited fields: depends_on
- Phase 8 replanned: D-23 removes controlled-Linux performance from the current blocking aggregate, adds Plan 08-14, and preserves SEED-006 as the future qualification path.
- Phase 8 gap plan 08-15 adds the missing configuration-only `tools/run_phase8_qualification.py --preflight` contract before Plan 08-11; it cannot contact services, mutate resources, write evidence, or qualify BACK-05.
- Historical before gap discovery (superseded): the D-24 replan added 08-16 as the only remaining canonical plan after deferring 08-11/08-12 to SEED-007. The current chain includes later gap Plans 08-17 through 08-19.
- Historical before the second gap (superseded): Plan 08-17 was once expected to lead directly through Plan 08-18 to 08-16. The current chain is 08-18 complete → 08-19 pending → 08-16 pending.
- Historical before the coverage measurement (superseded): Plan 08-18 was once described as the final prerequisite for 08-16. The exact post-08-18 report made Plan 08-19 the final prerequisite instead.
- Phase 8 gap Plan 08-19 restores the coverage ratchet after the approved Plan 08-18 test correction; it becomes the final prerequisite before Plan 08-16 resumes and cannot edit production code or the baseline.
- Phase 8 Plan 08-19 completed with deterministic SQLite validation/error and malformed-evidence coverage; raw repository/critical floors now exceed the frozen baseline, so 08-16 is the only remaining canonical plan.
- Phase 10 added: Direct pre-production SqlCache removal approved to narrow Cacheness to BlobStore plus UnifiedCache and reduce unrelated maintenance surface.
- Phase 11 added: Clean supplemental documentation and normalize validation evidence.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Extensibility | General cache-policy plugin framework | Deferred to v2 | Project initialization |
| Storage | New backend families, deduplication, and cross-backend physical migration | Deferred to v2 | Project initialization |
| APIs | Native async storage/cache APIs and distributed coherence | Deferred to v2 | Project initialization |
| Security | Hostile pickle/dill deserialization | Out of scope; trusted payload boundary | Project initialization |
| Architecture | `SqlCache` redesign or merger | Out of scope | Project initialization |
| Remote qualification | Real PostgreSQL/Amazon-S3 exact-SHA qualification and immutable GitHub release publication | Deferred to SEED-007 / NOT_QUALIFIED / NOT_PUBLISHED | Phase 08 D-24 |

## Session Continuity

Last session: 2026-09-17T20:03:55.392Z
Stopped at: Phase 11 context gathered
Resume file: .planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md
