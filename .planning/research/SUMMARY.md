# Project Research Summary

**Project:** Cacheness
**Domain:** Backend-neutral Python blob storage with a composed cache-policy layer
**Researched:** 2026-08-29
**Confidence:** MEDIUM

## Executive Summary

Cacheness is a Python storage and caching library whose object-storage path needs a reliability refactor, not a new cache product. The researched target is a layered design: `BlobStore` becomes the sole owner of serialization coordination, payload and metadata lifecycle, integrity, deletion, rollback, and reconciliation; `UnifiedCache` remains the compatibility-facing policy layer for keying, TTL, eviction, invalidation, statistics, and decorators; `SqlCache` stays independent. Experts solve this class of cross-backend problem with a typed, versioned entry model, immutable payload generations, and a metadata compare-and-swap (CAS) publication point rather than by pretending filesystem/S3 and SQL/JSON can share a distributed transaction.

The build should begin by freezing current API and stored-data behavior, then prove one backend-neutral lifecycle contract with deterministic fault and race tests before wiring local and remote backends. Filesystem/memory plus JSON/SQLite form the local reference path; S3/PostgreSQL follow only after generation, integrity, tombstone, and reconciliation semantics are stable. Migration/rebuild must be explicit and resumable before new-format writes become the default, while path containment, safe parsing, and required-signing fail-closed behavior are enforced throughout. The largest risks are split-brain payload/metadata commits, same-key races, metadata-only cleanup, unsafe legacy migration, and false confidence from mocks; each requires inventory-based recovery and production-service tests.

## Key Findings

### Recommended Stack

Use Python 3.11+ standard-library `Protocol`, frozen dataclasses, enums, context managers, `hashlib`, and `hmac` for the canonical records, state transitions, lifecycle ownership, and integrity boundaries. Add `filelock >=3.20` only for bounded local JSON/filesystem maintenance locks, never as a distributed-storage lock. Add Alembic `>=1.19` to SQL-bearing extras for SQLite/PostgreSQL schema revisions; it complements, but does not replace, a Cacheness migrator for JSON records and payloads.

The quality stack should be uv-locked and exercised in isolated distributions: pytest `~=9.1.1`, pytest-cov `~=7.1`, coverage `~=7.15`, Hypothesis `~=6.165`, pytest-xdist `~=3.8`, pytest-timeout `==2.4.0`, mypy `~=2.3`, Import Linter `~=2.13`, Ruff `~=0.16.4`, pytest-benchmark `~=5.2`, and pip-audit `~=2.10`. Use GitHub Actions with pinned `astral-sh/setup-uv`, blocking CPython 3.11–3.14 (3.15 scheduled until supported), a PostgreSQL service container, fast moto S3 tests plus a credentialed real-S3 acceptance tier, and base-wheel/each-extra smoke tests. Keep `boto3 >=1.37.32` for the S3 conditional/checksum behavior under test.

**Core technologies:**

- Typed stdlib models and protocols: canonical `StoredEntry`/`BlobRef`, typed errors, capability declarations, and explicit lifecycle state without adding a runtime modeling dependency.
- Immutable payload generations + metadata CAS: makes the metadata pointer the visibility commit point across non-transactional payload and metadata systems.
- `filelock`: process-level coordination for local JSON/document replacement, with timeouts and narrow scope; not a substitute for PostgreSQL/S3 CAS.
- Alembic: explicit relational schema migration with SQLite batch support and PostgreSQL transactional DDL, paired with a project-owned stored-data migration journal.
- pytest/Hypothesis/fault injection: shared backend contracts, state-machine lifecycle sequences, crash-window tests, and deterministic race coverage.
- Ruff/mypy/Import Linter: enforce Python 3.11-compatible code, typed backend boundaries, and the `UnifiedCache → BlobStore → backend` dependency direction.

Detailed stack recommendations and source links: [STACK.md](./STACK.md).

### Expected Features

**Must have (table stakes):**

- A versioned canonical manifest and explicit active/tombstone lifecycle state, with opaque backend references, independent payload digest, codec/handler version, size, revision, and signing metadata.
- Staged immutable writes, metadata-as-visibility-commit, copy-on-write overwrite, conditional same-key publication, compensating cleanup, idempotent delete/clear, and dry-run/resumable reconciliation.
- Read verification before deserialization; deterministic typed distinction between missing, corrupt, conflict, and backend failure; explicit resource ownership via `close()` and context managers.
- One payload protocol for filesystem, memory, and S3 and one metadata protocol for JSON, memory, SQLite, and PostgreSQL, backed by a parameterized full compatible pair matrix.
- Per-key coordination plus backend CAS/generation checks, read-during-overwrite consistency, documented delete/put ordering, bounded retries, and declared topology capabilities.
- `UnifiedCache` composition over `BlobStore` for TTL, eviction, invalidation, statistics, cached-`None`, and decorator clearing, while keeping `SqlCache` separate.
- Public API compatibility inventory/adapters, explicit schema/key/locator versions, dry-run migrate/verify/rebuild, resumable copy-verify-switch, path containment, safe metadata parsing/query construction, and fail-closed required signing.
- CI, minimal-install and optional-extra smoke tests, backend integration services, targeted coverage gates, security checks, and benchmark baselines.

**Should have (competitive):**

- First-class `audit()`/`reconcile()` plans with explainable, machine-readable repairs.
- A third-party backend conformance kit and capability-validated durability profiles.
- Optional generation-aware optimistic concurrency (`put-if-absent`/replace-if-version), structured lifecycle telemetry, and streaming integrity-preserving transfers for large artifacts.
- Migration progress planning and reports that make copy/verify/switch operationally safe.

**Defer (v2+):**

- General cache-policy plugin/entry-point framework, new backend families, async APIs, distributed in-memory coherence/replication, deduplication/reference counting, hostile-pickle sandboxing, and `SqlCache` redesign.
- Do not add cross-resource distributed transactions or promise exactly-once retries; use immutable generations, idempotency, conditional operations, and reconciliation instead.

Detailed feature landscape: [FEATURES.md](./FEATURES.md).

### Architecture Approach

Make `BlobStore` the only public object-lifecycle coordinator. Its public contract should expose `put/get/stat/list/touch/delete/clear/reconcile` over typed models and errors; handlers become codec/staging adapters, payload backends own opaque immutable byte references, and metadata repositories own normalized entries plus revisioned CAS/tombstones. `UnifiedCache` consumes only backend-neutral entry views and lifecycle methods, while `SqlCache` remains an unchanged boundary. A legacy adapter and explicit migration service handle old nested `actual_path` records without preserving those shapes in the new core.

**Major components:**

1. `UnifiedCache` — derives keys, evaluates TTL, selects eviction victims, invalidates, records policy statistics, and preserves public/decorator behavior; it must not touch handlers, paths, payloads, or metadata repositories directly.
2. `BlobStore` and lifecycle value objects — validate keys, coordinate codecs and backends, publish signed canonical entries, enforce generation-qualified destructive operations, and map strict errors to compatibility behavior.
3. Payload backends and metadata repositories — filesystem/memory/S3 immutable objects and JSON/memory/SQLite/PostgreSQL CAS repositories with backend-specific topology/capability declarations.
4. Reconciler — inventories active records, tombstones, missing/corrupt payloads, unreferenced generations, and staging leftovers with bounded pagination, grace periods, checkpoints, and deterministic repair/quarantine.
5. Migration service and legacy adapter — inspect, plan, dry-run, copy/verify/switch, resume, verify, and explicit rebuild; never silently rewrite on read or startup.
6. Composition/config and security services — preserve injected instances/registries, validate backend combinations, lazy-load extras, contain filesystem/S3 locators, and sign/verify the complete canonical record.

The central pattern is “immutable payload, mutable pointer”: serialize and checksum a unique generation, write it, then atomically CAS-publish the metadata pointer. Deletes first tombstone the observed revision, remove that exact generation, and retain cleanup intent for reconciliation. This gives old-or-new visibility without claiming a distributed transaction.

Detailed boundaries and data flow: [ARCHITECTURE.md](./ARCHITECTURE.md).

### Critical Pitfalls

1. **Split-brain payload and metadata transactions** — define metadata publication as the commit point; stage immutable generations, retain the old entry until CAS succeeds, and inventory/repair orphaned candidates after compensation failures.
2. **Same-key physical overwrite races** — never use fixed final or `.tmp` names; use unique generations, revision-qualified deletes, per-key local locks only as an optimization, and backend-enforced CAS for process/host concurrency.
3. **Metadata-only delete/TTL/eviction/clear** — route every removal through one `BlobStore` tombstone-and-exact-generation-delete primitive; preserve tombstones/retry state and verify payload inventories in tests.
4. **False S3/SQL atomicity and integrity assumptions** — S3 consistency and relational ACID cover only their own resources; separate ETag/version from an algorithm-qualified SHA-256 digest, keep DB transactions short, handle conditional conflicts/deadlocks explicitly, and require real service tests.
5. **Breaking or unsafe stored-data migration** — freeze released golden fixtures, version schema/key/handler/locator formats, provide dry-run resumable migration or explicit rebuild before cutover, contain legacy paths, authenticate metadata before deserialization, and fail closed when required signing is unavailable.

Other recurring risks are overpromising JSON/memory topology, expensive full-store reconciliation, registry/injection breakage, finalizer-based cleanup, and optimizing before lifecycle semantics stabilize. Detailed warnings: [PITFALLS.md](./PITFALLS.md).

## Implications for Roadmap

Based on the combined dependencies, use seven ordered phases (Phase 0 is intentional groundwork):

### Phase 0: Characterization, compatibility, and security baseline

**Rationale:** The rewrite changes ownership and stored representations; without frozen public behavior and released-format fixtures, regressions cannot be distinguished from intentional migration.

**Delivers:** API/alias/constructor/decorator/`None`/`SqlCache` characterization tests; golden legacy manifests and payloads; baseline path/parser/signing tests; current 777-test/66%-coverage baseline captured; representative performance scenarios; explicit supported topology and migration-window decisions.

**Addresses:** Public compatibility, trusted-payload boundary, migration fixtures, security and quality table stakes.

**Avoids:** Silent data loss, `eval`/path traversal regressions, accidental `SqlCache` scope expansion, and benchmark claims without a baseline.

### Phase 1: Canonical lifecycle contract and test harness

**Rationale:** Every backend depends on one semantic model; implement and test the protocol before backend-specific behavior or `UnifiedCache` rewiring.

**Delivers:** Frozen `StoredEntry`/`BlobRef`, schema/key/locator versions, typed errors and capabilities, explicit state machine, codec staging seam, generation/revision rules, checksum/HMAC scope, metadata CAS/tombstone interfaces, and parameterized fault/race/state-machine contract harness using memory doubles.

**Addresses:** Canonical manifest, atomic lifecycle, integrity, concurrency, backend protocols, and deterministic errors.

**Avoids:** Dual entry shapes, fixed temp names, backend branches in policy, and a “happy path only” interface that cannot express conflicts or recovery.

### Phase 2: Local lifecycle backends and reconciliation

**Rationale:** Filesystem/memory plus JSON/SQLite provide a controllable reference implementation before remote semantics, while exposing durability and lock-topology limits early.

**Delivers:** Unique same-directory staging + `os.replace`/optional fsync, memory generation maps, JSON reload-under-`filelock` atomic snapshots, SQLite short CAS transactions/busy handling, exact delete/clear, close/context ownership, local reconciliation, and thread/process crash tests.

**Uses:** `filelock`, stdlib integrity primitives, SQLite/Alembic schema patterns, and Hypothesis/fault injection.

**Implements:** Payload/metadata repository boundaries and reconciler.

**Avoids:** Mistaking atomic rename for directory durability, promising JSON/SQLite multi-host safety, deleting current generations during stale cleanup, and letting JSON become an unbounded production hot path.

### Phase 3: Remote lifecycle (S3 + PostgreSQL)

**Rationale:** Remote backends are the highest-risk integration and should inherit a proven protocol rather than define it through mocks.

**Delivers:** Immutable S3 generation objects, explicit checksums plus ETag/version tokens, conditional operations and bounded classified retries; PostgreSQL row CAS/short transactions/deadlock handling; remote stream/resource cleanup; paginated reconciliation; real-service integration jobs and supported capability validation.

**Addresses:** Full backend parity, multi-process/host concurrency, remote cleanup, topology profiles, and production integration evidence.

**Avoids:** Cross-store transaction claims, unconditional logical-key overwrites, ETag-as-MD5 errors, long DB transactions around uploads, and moto-only confidence.

### Phase 4: UnifiedCache composition and policy correctness

**Rationale:** Only after `BlobStore` passes lifecycle, rollback, delete, and reconciliation contracts can the facade safely relinquish its current direct storage side effects.

**Delivers:** `UnifiedCache` delegation for put/get/stat/list/touch/delete/clear; TTL as policy metadata/decision, revision-aware eviction, all invalidation through BlobStore, separate hit/miss/error statistics, cached-`None` correctness, decorator clear behavior, and architecture/spies proving no direct handler/backend access.

**Addresses:** Policy-layer table stakes and registry/injection compatibility while preserving public APIs.

**Avoids:** Two drifting lifecycle implementations, metadata-only eviction, statistics changing storage correctness, and `SqlCache` coupling.

### Phase 5: Explicit migration, rebuild, and compatibility cutover

**Rationale:** New writes must not strand existing entries. Migration depends on canonical lifecycle and reconciliation so each converted object can be copied, verified, conditionally switched, and safely resumed.

**Delivers:** Legacy read adapter, `inspect_store`/`plan_migration`, dry-run machine-readable report, versioned Alembic SQL revisions, JSON snapshot backup/replacement, journaled resumable copy-verify-switch, interruption recovery, checksum/count verification, mixed old/new read window or explicit stop-the-world constraint, and documented rebuild/rollback boundary.

**Addresses:** Stored-data compatibility, migration/rebuild, security hardening for legacy paths and parsers.

**Avoids:** Silent startup mutation, guessed handlers/locators, irreversible in-place rewrites, and dual-write state explosion.

### Phase 6: Production quality gates and performance stabilization

**Rationale:** Correctness evidence must be broad and repeatable before finalizing cost budgets or declaring backend neutrality.

**Delivers:** GitHub Actions Python 3.11–3.14 matrix, Windows/macOS smoke coverage, minimal wheel/sdist and each-extra installs, Ruff/mypy/Import Linter/pip-audit gates, PostgreSQL service and pre-release live S3 jobs, required contract/fault/concurrency/crash suites, targeted coverage thresholds, benchmark baselines and regression budgets, and operational documentation.

**Addresses:** Release quality, packaging, security, observability, scalability, and measured performance.

**Avoids:** Skipped integration tests, broad lint suppression, premature optimization, N+1 inventory operations, and unbounded reconciliation memory/cost.

### Phase Ordering Rationale

- Compatibility fixtures and security boundaries precede format changes; canonical models precede all backend work.
- Local backends are the reference for generation/CAS/reconciliation; S3/PostgreSQL then validate distributed semantics rather than inventing them.
- `UnifiedCache` follows the complete `BlobStore` lifecycle so policy cannot preserve bypasses or metadata-only cleanup.
- Migration precedes new-format cutover; release gates and benchmark blocking follow semantic stabilization.
- Tests enter each phase, with Phase 6 making service, packaging, coverage, and performance evidence mandatory. `SqlCache` remains isolated throughout.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 0:** Exact historical formats, supported migration window, Windows/macOS path replacement and symlink behavior, and public direct-`BlobStore` integrity-failure compatibility need repository evidence.
- **Phase 2:** Cross-platform `filelock` behavior, JSON writer scope, SQLite WAL/busy defaults, directory fsync, and crash tests need implementation probes.
- **Phase 3:** Chosen S3 target (AWS versus compatible service), conditional-write/checksum/versioning semantics, credentialed CI policy, PostgreSQL majors, and retry behavior require live-service validation.
- **Phase 5:** Whether v1 transfers payloads across backend pairs or limits migration to format conversion plus rebuild, and the rolling-read window, are unresolved product decisions.
- **Phase 6:** Benchmark variance, CI duration, exact coverage ratchets, and remote performance thresholds require measured baselines.

Phases with standard patterns that can usually skip `--research-phase`:

- **Phase 1:** Typed protocols, immutable generations, CAS, staged writes, and fault-injection state machines are well-established; validate against project fixtures rather than broad new research.
- **Phase 4:** Layered facade/policy composition, revision-qualified deletes, and decorator compatibility are primarily codebase integration work.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Versions and capabilities were checked against current official docs/release metadata, but exact dependency interaction with legacy shapes and live-service policy remains to probe. |
| Features | MEDIUM | Strong convergence between project evidence and official platform behavior; exact user-facing performance and migration promises remain open. |
| Architecture | HIGH | Codebase ownership boundaries, current call paths, and the layered/CAS recommendation are directly traced; cross-process locking details remain implementation-specific. |
| Pitfalls | MEDIUM | Current defects are directly evidenced and distributed-storage claims were cross-checked officially; real S3/PostgreSQL failure behavior still needs integration validation. |

**Overall confidence:** MEDIUM. The lifecycle direction and phase ordering are high-confidence; numeric thresholds, migration scope, topology promises, and live remote semantics are not yet validated.

### Gaps to Address

- Define how long old manifests/handlers/key algorithms remain readable and whether mixed-version deployments are supported; use Phase 0 fixtures and Phase 5 rollout planning.
- Decide the strict default for direct `BlobStore.get` on corruption/signature failure versus compatibility miss mapping; test both internal typed errors and public behavior.
- Choose AWS or a specific S3-compatible service as release authority and verify conditional writes, checksums, multipart/versioning, credentials, and cleanup against it rather than moto assumptions.
- Establish JSON and SQLite scale/topology envelopes, tombstone retention, orphan grace periods, and reconciliation checkpoint formats from crash/soak tests.
- Measure real CI variance before blocking on suggested coverage/performance targets; preserve the current 66% statement baseline while ratcheting new lifecycle code.
- Confirm whether cross-backend physical payload migration is required for v1 or whether explicit rebuild is acceptable for unsupported legacy handlers.

## Sources

### Primary (HIGH confidence)

- [PROJECT.md](../PROJECT.md) — milestone scope, active requirements, constraints, current baseline, and explicit out-of-scope boundaries.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — direct codebase ownership analysis, canonical contracts, data flow, and dependency-aware build order.
- [Python `os.replace`](https://docs.python.org/3.12/library/os.html#os.replace) — filesystem replacement semantics.
- [Amazon S3 consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) and [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) — per-key visibility and precondition behavior.
- [SQLite transactions](https://www.sqlite.org/lang_transaction.html), [WAL](https://www.sqlite.org/wal.html), and [busy timeout](https://sqlite.org/c3ref/busy_timeout.html) — local writer and contention semantics.
- [PostgreSQL transaction isolation](https://www.postgresql.org/docs/current/transaction-iso.html) and [explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html) — CAS/locking/retry constraints.
- [Alembic documentation](https://alembic.sqlalchemy.org/en/latest/) and [batch migrations](https://alembic.sqlalchemy.org/en/latest/batch.html) — versioned SQL schema evolution.

### Secondary (MEDIUM confidence)

- [STACK.md](./STACK.md) — current recommended versions, uv installation guidance, CI matrix, quality gates, and alternatives.
- [FEATURES.md](./FEATURES.md) — table stakes, differentiators, anti-features, dependencies, MVP, and acceptance targets.
- [PITFALLS.md](./PITFALLS.md) — critical/moderate/minor pitfalls and phase-specific warnings.
- [Boto3 `put_object`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html) — S3 checksum and conditional-write API details.
- [Python pickle warning](https://docs.python.org/3.12/library/pickle.html) — trusted-payload boundary and HMAC rationale.
- [Hypothesis stateful testing](https://hypothesis.readthedocs.io/en/latest/stateful.html), [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html), and [pytest monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html) — reusable contract and fault-testing patterns.
- [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/) and [GitHub PostgreSQL service containers](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers) — CI implementation patterns.
- [PyPA `pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) — dependency and extra declarations.
- [OpenStack Glance migration guidance](https://docs.openstack.org/glance/latest/contributor/database_migrations.html) — expand/migrate/contract ordering.

### Tertiary (LOW confidence)

- None identified. The unresolved items are implementation/product decisions or missing project measurements, not single-source recommendations.

---
*Research completed: 2026-08-29*
*Ready for roadmap: yes*
