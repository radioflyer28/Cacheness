---
gsd_state_version: 1.0
milestone: v1.0
current_phase: 07
current_phase_name: Explicit Migration and Rebuild Cutover
status: verifying
stopped_at: Completed 07-11-PLAN.md
last_updated: "2026-09-10T05:03:33.525Z"
last_activity: 2026-09-09
last_activity_desc: Phase 07 execution started
state_head: 3a993bbe13892987ff901943cdb33e4a7a66e727
progress:
  total_phases: 8
  completed_phases: 5
  total_plans: 92
  completed_plans: 91
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-06)

**Core value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.
**Current focus:** Phase 07 — Explicit Migration and Rebuild Cutover

## Current Position

Phase: 07 (Explicit Migration and Rebuild Cutover) — EXECUTING
Plan: 11 of 11
Status: Phase complete — ready for verification
Last activity: 2026-09-09 — Phase 07 execution started

Phase 05 closed with 9/9 canonical plans summarized, 9/9 verified must-haves,
19/19 Nyquist-covered tasks, and 43/43 canonical security threats closed
(`threats_open: 0`). BACK-05 and its five real-service threats moved intact to
Phase 8; sanitized `UNAVAILABLE` evidence does not create a release support
claim.

Phase 04 closed with 14/14 plans summarized, a clean final code review,
18/18 verified must-haves, 29/29 Nyquist-covered tasks, and 64/64 plan-time
security threats resolved or explicitly accepted (`threats_open: 0`). The
canonical milestone count excludes superseded Plans 03-19 and 05-10, whose
`status: superseded` frontmatter is the machine-readable retirement record.

The user explicitly bypassed GSD execute/review/checker. The older
03-VERIFICATION.md remains historical gaps_found evidence, not the current
disposition or an instruction to re-execute closed plans. This completion is
direct primary-agent qualification, not an independent GSD verifier verdict.
Windows remains UNAVAILABLE/NOT_QUALIFIED; Phase 999.1 is unchanged.

Progress: [██████░░░░] 5 of 8 phases complete

## Roadmap Evolution

2026-09-06 — User-approved downstream alignment after Phase 3 direct qualification:

- Phase 4 edited: starting point and success criteria require reuse of the entry interface, catalog customization and narrower transactional adapters; schema choices remain open for discussion.
- Phase 5 edited: title, goal and success criteria now specify Supported Topology Qualification, not full Cartesian/availability parity; no advertised backend family was dropped.
- Phase 6 edited: carried-forward scope and criteria separate delivered local integration from remaining policy/decorator/statistics work and preserve derived-state partial outcomes.
- Phase 7 edited: goal and criteria require stopped-worker, explicit offline migration/cutover; no online writer coordination is implied.
- Phase 8 edited: criteria retain finite integrity/recovery cases, initialized-worker fixtures, separate performance distributions and honest platform/service qualification.
- PROJECT.md now describes the delivered baseline; REQUIREMENTS.md records STOR-03 through STOR-07 as complete only in the qualified local scope, leaving full BACK/CACH/MIGR/QUAL acceptance pending. CONTEXT.md distinguishes catalog attributes from derived metadata outcomes.

Phase numbering, order, dependencies, requirement ownership, and milestone scope
are unchanged. This was a documentation alignment, not another implementation or
qualification run. Phase 4 context and eight initial checker-approved plans were
created on 2026-09-07. Verification then found finite public
catalog/composition/projection gaps, and five additive checker-approved gap plans
(04-09 through 04-13) were prepared on 2026-09-08. Phases 5–8 still have no plans
or phase directories.
GSD's raw disk count still includes superseded 03-19 and may call Phase 3 partial;
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

Phase 03 rows in this historical table include superseded attempts and do not define current completion. Authoritative milestone progress is 46 of 54 currently planned canonical plans: Phase 1 has 15 complete, Phase 2 has 7 complete, Phase 3 has 24 complete, and Phase 4 has 8 planned. The last five Phase 3 plans were implemented directly in one bounded pass. Phases 5–8 remain unplanned. Plan 03-19 is excluded.

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

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- `BlobStore` is the canonical payload-plus-metadata lifecycle owner; `UnifiedCache` owns cache policy only.
- `SqlCache` remains a separate subsystem. Pre-production public APIs may be replaced by one coherent BlobStore/cache surface rather than retained through compatibility adapters.
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

### Pending Todos

- Execute Phase 4's eight checker-approved plans, beginning with Wave 0 catalog and composition contracts.
- Apply ADR 0001 and the direct implementation guide; the user approved both 03-22/24 compatibility checkpoints with “proceed.”
- Do not restart closed 03-21 through 03-25 gaps or an automated race-fix loop from historical review/verification artifacts.

### Blockers/Concerns

- Historical Phase 1 compatibility fixtures remain evidence but do not define a supported runtime read window after the pre-production reset; Phase 7 will define the source-version window for future released formats.
- Phase 3 planning must derive tombstone retention and orphan grace defaults from fault/crash testing.
- Phase 8 performance and coverage thresholds must be finalized from measured baselines rather than estimates.
- Phase 06 verification pending: default uv lacks pandas, so fixed CACH-07 SqlCache regression and the full pytest suite cannot collect; PostgreSQL/S3 BACK-05 and native-Windows evidence remain unqualified.

### Roadmap Evolution

- Phase 5 edited: Moved BACK-05 non-substitutable PostgreSQL/Amazon-S3 real-service qualification gate intact to Phase 8; Phase 5 retains the candidate implementation, deterministic contracts, frozen live suites, fail-closed runner, and truthful UNAVAILABLE evidence without a release support claim.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Extensibility | General cache-policy plugin framework | Deferred to v2 | Project initialization |
| Storage | New backend families, deduplication, and cross-backend physical migration | Deferred to v2 | Project initialization |
| APIs | Native async storage/cache APIs and distributed coherence | Deferred to v2 | Project initialization |
| Security | Hostile pickle/dill deserialization | Out of scope; trusted payload boundary | Project initialization |
| Architecture | `SqlCache` redesign or merger | Out of scope | Project initialization |

## Session Continuity

Last session: 2026-09-10T05:03:33.405Z
Stopped at: Completed 07-11-PLAN.md
Resume file: None
