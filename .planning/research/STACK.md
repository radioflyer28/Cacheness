# Technology Stack

**Project:** Cacheness — backend-neutral BlobStore refactor
**Researched:** 2026-08-29
**Scope:** Additions and changes for this milestone only; the existing stack is not repeated here.
**Overall confidence:** MEDIUM — recommendations are cross-checked against current official documentation and release metadata, but the GSD confidence seam classifies verified web research as MEDIUM.

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python `typing.Protocol`, frozen `dataclasses`, `Enum`, `contextlib`, `hashlib`, and `hmac` | Python 3.11+ stdlib | Define payload/metadata contracts, lifecycle records, commit states, cleanup ownership, and cryptographic integrity | Use the standard library for the new internal model. A cache entry should be a typed, versioned record rather than an unvalidated nested dictionary, but Cacheness does not need Pydantic or another runtime modeling framework. **Confidence: HIGH** from the supported-runtime constraint and direct codebase evidence. |
| `filelock` | `>=3.20` (lock current `3.32.4`) | Cross-platform inter-process lock around local JSON metadata commit/reconciliation and filesystem maintenance | The JSON backend needs a real process boundary, not only `threading.Lock`. Keep locks scoped to one metadata document or maintenance operation and always use a timeout. `filelock` is not a distributed lock and must never be used as the S3/PostgreSQL concurrency mechanism. **Confidence: MEDIUM** (current PyPI metadata and pytest-xdist's documented inter-process fixture pattern). |
| Alembic | `>=1.19` (lock current `1.19.1`) | Version and migrate SQLAlchemy-owned SQLite/PostgreSQL metadata schemas | Alembic is the SQLAlchemy project's migration tool, has programmatic command APIs, supports SQLite batch migrations, and benefits from PostgreSQL transactional DDL. Put it only in extras that install SQL metadata support. A project-owned Cacheness migration orchestrator must still coordinate payload-format and metadata-record migrations; Alembic alone is not the stored-data migration feature. **Confidence: MEDIUM** (official Alembic docs and current release metadata). |

### Database

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Alembic revision scripts plus a Cacheness schema-version table | Alembic `>=1.19`; entry schema version starts at a project-defined integer | Upgrade SQLite and PostgreSQL metadata safely and detect incompatible databases before normal reads/writes | Replace implicit `create_all()` evolution with explicit `upgrade`, `inspect`, and documented `rebuild` modes. Use `render_as_batch=True`/`batch_alter_table()` for SQLite migrations that SQLite cannot express with `ALTER`; test every revision forward from checked-in legacy fixtures on both engines. |
| Metadata compare-and-swap contract | Project-owned, no dependency | Make the metadata pointer the authoritative commit point | Add `revision`/`generation` to the canonical entry record and require `put_if_revision`, tombstone/finalize delete, and deterministic conflict errors. SQLite should use a short write transaction (for example `BEGIN IMMEDIATE` where appropriate); PostgreSQL should use an atomic `UPDATE ... WHERE revision = :expected` or row lock. Memory uses `RLock`; JSON holds `FileLock` while atomically replacing the document. This is more portable than pretending one lock implementation spans every backend. |
| Backend capability declaration | Project-owned typed enum/record | Validate supported combinations and concurrency guarantees at construction | Declare capabilities such as atomic replace, conditional create, CAS metadata, process scope, streaming, and durable flush. Reject configurations whose requested guarantee cannot be met—for example, multi-host coordination with JSON/SQLite metadata—even if the payload is on S3. Do not silently downgrade. |

### Infrastructure

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| GitHub Actions with `astral-sh/setup-uv` | Pin action by full commit SHA; pin uv version from the lock-maintenance policy | Make tests, lint, typing, packaging, and backend matrices mandatory | Official uv guidance supports Python matrices, locked sync, built-in caching, and isolated wheel/sdist smoke tests. Keep workflow permissions read-only by default and separate any publish credentials from build/test jobs. **Confidence: MEDIUM** (official uv and GitHub Actions docs). |
| CPython compatibility matrix | Blocking: `3.11`, `3.12`, `3.13`, `3.14`; scheduled non-blocking: `3.15` prerelease until supported | Verify the declared `>=3.11` range rather than only the developer's 3.13 environment | Run the complete fast suite on Linux for every blocking version. Run filesystem/path/locking and wheel-import smoke suites on current Windows and macOS. Do not claim 3.15 support until it is blocking. |
| PostgreSQL GitHub Actions service container | Pin a supported PostgreSQL major image; test at least the oldest and newest documented majors before release | Exercise real transactions, CAS, migrations, cleanup, and concurrent same-key writes | GitHub officially supports PostgreSQL service containers on Ubuntu. Use a health check and a per-job database/schema; do not add `pytest-postgresql` or Testcontainers merely to wrap a service the workflow already provides. |
| S3 test tiers | Existing moto for fast tests; opt-in/nightly real AWS S3 integration | Separate deterministic unit behavior from authoritative AWS semantics | Moto remains useful for request/response and failure-path tests. A credentialed, least-privilege AWS job should verify conditional writes, checksums, multipart behavior if supported, deletion, and reconciliation before release. MinIO/LocalStack may be used locally but must not be the sole evidence for AWS behavior and should not become mandatory dependencies. |
| Isolated distribution checks | `uv build`; isolated execution from the built wheel and sdist | Prove minimal import, extras, `py.typed`, and packaged migrations are actually shipped | Test the base wheel with only declared core dependencies, then install each extra independently (`recommended`, `dataframes`, `s3`, `postgresql`, `cloud`, `tensorflow` where feasible). Import public modules and execute one focused capability smoke test. This catches the current declared-vs-imported dependency class of defect. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | `~=9.1.1` in the locked dev group | Contract, compatibility, migration, fault-injection, and concurrency test runner | Upgrade deliberately and set pytest 9 `strict = true`, `required_plugins`, deterministic parametrization IDs, and explicit markers (`unit`, `contract`, `integration`, `postgresql`, `s3_live`, `concurrency`, `migration`, `benchmark`). Strict mode is appropriate because uv locks the dev environment. |
| pytest-cov + coverage.py | `pytest-cov~=7.1`; `coverage[toml]~=7.15` | Line and branch coverage with CI thresholds and subprocess coverage | Enable `branch = true`. For child-process crash/concurrency tests, use coverage.py's subprocess patching supported by current pytest-cov guidance. Do not keep repository-wide warning suppression to make coverage runs quiet. |
| Hypothesis | `~=6.165` | Model-based and property testing of lifecycle sequences and record parsing | Build one `RuleBasedStateMachine` against a simple in-memory reference model, then run it against each backend contract. Generate `put/get/exists/delete/clear/reconcile`, duplicate keys, invalid metadata, path-like keys, and injected failures. Use deterministic CI profiles and retain the example database as an artifact. |
| pytest-xdist | `~=3.8` | Parallel suite execution and accidental shared-state detection | Use `-n auto` only for isolated unit/contract tests. Give workers unique roots/databases using `worker_id`/`testrun_uid`; group or serialize shared PostgreSQL/S3 tests. Xdist is not the concurrency test—same-key races must be orchestrated inside a test with barriers, threads, and subprocesses. |
| pytest-timeout | `==2.4.0` | Detect deadlocked concurrency tests and dump thread stacks | Apply generous explicit timeouts to concurrency/integration tests and a session timeout in CI. Do not use timing thresholds as correctness assertions. Version `2.5.0` is yanked, so remain on `2.4.0` until a non-yanked successor is verified. |
| mypy | `~=2.3` | Enforce backend protocols, lifecycle state transitions, optional-dependency boundaries, and public annotations | Start strict checking on new storage/lifecycle/migration modules and broaden by deleting explicit legacy overrides. Do not turn on blanket `ignore_missing_imports`; isolate third-party clients behind small typed adapters. Add `py.typed` only when exported APIs pass and the marker is verified in the wheel. |
| Import Linter | `~=2.13` | Make the new architecture an executable dependency rule | Define a layers contract: public compatibility facades/`UnifiedCache` → `BlobStore` policy-free lifecycle → backend protocols/adapters. Add forbidden contracts so BlobStore cannot import cache policy, backend modules cannot import coordinators, and `SqlCache` remains independent of the object-storage path. |
| Ruff | `~=0.16.4` in the locked dev group | Single formatter/linter/security/static-style gate | Change `target-version` from `py312` to `py311`, matching the public floor. Ratchet existing findings, then enable at minimum `E`, `F`, `W`, `I`, `B`, `C4`, `UP`, `SIM`, `RUF`, `S`, and exception-hygiene rules. Make `S307` (`eval`) blocking immediately; use narrow test-only ignores for `S101`. Do not add Black, isort, Flake8, Bandit, or tryceratops separately. |
| pytest-benchmark | `~=5.2` | Track end-to-end put/get/delete/reconcile costs and final regression budgets | Benchmark representative small object, NumPy/dataframe, filesystem, memory, SQLite, and mocked-network workloads. Save JSON baselines by Python/platform and make comparisons informational during correctness work; only the final stabilized, low-variance suite should block on agreed budgets. |
| pip-audit | `~=2.10` | Known-vulnerability scan for the resolved release environments | Run on schedule and in release CI against the locked base plus each supported extra. Any ignore must name the advisory, explain reachability, name an owner, and expire. Do not treat it as a source-code security scanner. |

## Required Implementation Patterns

### One versioned entry schema

Define a canonical frozen record used by every metadata backend. At minimum it should contain `schema_version`, logical key, opaque payload reference, payload generation, serializer/format version, byte size, cryptographic digest and algorithm, optional signature, created/updated timestamps, metadata revision, and lifecycle state. Backends may have physical columns/documents, but they must round-trip the same semantic record. Compatibility adapters should normalize legacy nested shapes only at the read/migration boundary, never throughout BlobStore.

### Immutable payload generation, metadata commit point

Use this backend-neutral write protocol:

1. Serialize to a unique generation-specific staging target; never let two writers share a fixed `.tmp` or final payload name.
2. Flush and verify size plus SHA-256 before publishing. For filesystem payloads create the temporary file in the destination directory, flush and `fsync` it, then `os.replace`; successful same-filesystem replacement is atomic on POSIX. When durable-commit mode is promised, also `fsync` the containing directory on platforms that support it. For S3, upload an immutable generation key and send a supported checksum.
3. Atomically compare-and-swap the metadata record to point at that generation. This is the lifecycle commit point.
4. If the CAS loses, delete the unreferenced generation. If post-commit cleanup fails, leave a detectable orphan for reconciliation rather than corrupting the committed entry.
5. Delete via a revisioned tombstone, remove the referenced payload idempotently, then finalize the metadata deletion. Reads must never return a tombstoned or partially verified entry.

This design prevents a late metadata write from describing another writer's payload. File locks remain a local implementation detail for JSON/filesystem maintenance, not the global consistency model.

### Integrity and signing boundary

Keep XXH3 for non-adversarial cache-key/fingerprint speed where compatibility requires it, but use stdlib SHA-256 for persisted payload integrity. Bind the HMAC signature to the complete canonical record—key, schema/format versions, payload reference, digest, size, generation, and security mode. Required signing or integrity must fail construction/read closed; it may not silently disable itself. Replace `eval` with explicit numeric parsing (`json`/`ast.literal_eval` only if the accepted grammar is deliberately bounded), require path containment after resolution, and reject unknown metadata/query field names before constructing SQL/JSON paths.

### Explicit migration and rebuild modes

Ship a dry-run-capable migration API/CLI surface with `inspect`, `migrate`, `verify`, and `rebuild` modes. The migration must be resumable and idempotent, record source/target schema versions, stage new payload generations, atomically switch metadata, and retain or quarantine legacy data until verification succeeds. Use Alembic only inside the relational schema portion. Check in fixtures produced by every supported legacy representation and test interrupted migration after each lifecycle boundary.

### Typed failures and injected dependencies

Introduce typed exceptions for conflict, corrupt entry, unsupported capability, migration required, dependency unavailable, and cleanup/reconciliation failure. Inject clock, identifier/generation factory, payload backend, metadata backend, and fault hooks. An injected clock removes the need for `freezegun`; a small explicit retry policy around classified S3 conflicts removes the need for `tenacity`.

## Production Quality Gates

| Gate | Blocking policy |
|------|-----------------|
| Existing behavior | All existing public tests pass; add compatibility tests for constructor parameters, aliases, decorators (including cached `None`), return/miss semantics, registration/injection, and `SqlCache` separation. Any intentional stored-data incompatibility requires a passing migration/rebuild fixture. |
| Backend contracts | The same semantic suite passes for 3 payload backends × 4 metadata backends where combinations are valid. Unsupported distributed combinations fail at configuration, not during a write. Live PostgreSQL is required on every PR touching SQL metadata; live AWS S3 is required before release for S3 lifecycle changes. |
| Failure atomicity | Deterministic fault injection covers failure before/after payload publish, before/after metadata CAS, during old-payload deletion, during tombstone finalization, and during reconciliation. After restart/reconcile, every key is either the old committed value or the new committed value—never mixed—and every orphan is discoverable. |
| Concurrency | Barrier-controlled thread and subprocess tests perform repeated same-key put/replace/delete/read. Assert no corrupt reads, metadata/digest mismatch, leaked fixed temp files, or deadlock. Add PostgreSQL multi-connection and S3 conditional-conflict cases. Xdist-only evidence is insufficient. |
| Coverage | First refactor PR: global statement coverage may not fall below the measured 66% baseline; enable branch measurement and record its baseline. Every new lifecycle/backend/migration module must reach at least 90% statements and 85% branches. Milestone release target: at least 80% statements/75% branches globally and 95%/90% for canonical lifecycle code. Parse coverage JSON with a small repository script for per-module gates instead of adding another coverage framework. |
| Static quality | `ruff check`, `ruff format --check`, mypy on the new architecture, and Import Linter contracts all pass. Establish a checked-in Ruff baseline/scope and ratchet it down; do not hide the current 137 findings with broad `noqa` or global ignores. |
| Packaging | Build wheel and sdist; isolated base install imports `cacheness`; each extra installs and exercises its advertised feature independently; invalid optional features raise focused errors; the wheel contains `py.typed` and migration resources. Test oldest supported dependency floors in one scheduled job and locked-current dependencies on PRs. |
| Security | No `eval`; resolved filesystem paths remain under configured roots; malicious query keys never become SQL fragments; unsigned/corrupt entries fail according to explicit mode; `pip-audit` passes or has reviewed, expiring exceptions. Keep pickle/dill documented as trusted-payload-only. |
| Performance | During refactor, benchmark results are recorded but non-blocking. Before release, establish budgets from at least 20 stable CI baseline runs; block only on material end-to-end regressions (initial recommendation: >15% median and >10 ms absolute for local hot paths, reviewed per benchmark). Correctness gates always take precedence. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Entry validation | Stdlib typed dataclass + explicit parser | Pydantic/attrs | Adds runtime surface and conversion behavior without solving commit atomicity; the canonical entry schema is small and controlled. |
| Local coordination | Immutable generations + metadata CAS, with `filelock` only for JSON/local maintenance | One global lock or `portalocker` everywhere | A single local lock cannot coordinate PostgreSQL/S3 across hosts and would conceal unsupported guarantees. CAS is the portable semantic contract. |
| Database migration | Alembic for SQL schema plus a Cacheness stored-data migrator | `create_all()`, hand-edit SQL only, or Alembic alone | `create_all()` does not upgrade existing schemas; hand SQL duplicates dialect logic; Alembic does not migrate payload objects or JSON records. |
| S3 tests | Moto fast tier + small live AWS verification tier | LocalStack/MinIO as sole integration | Emulators are useful but do not prove AWS conditional/checksum behavior. Adding a heavyweight mandatory emulator would increase CI complexity while leaving an authority gap. |
| PostgreSQL tests | GitHub service container + normal SQLAlchemy/psycopg fixtures | Testcontainers or `pytest-postgresql` | The official CI service already supplies the process boundary; an additional orchestration abstraction is not needed. |
| Static tools | Ruff + mypy + Import Linter | Black/isort/Flake8/Bandit/Pyright stack | Ruff already covers formatting, imports, and selected security rules. Mypy provides the gradual typed contract gate. Import Linter covers architecture; overlapping tools add churn. |
| Time control | Injected clock | freezegun | TTL and migration timing become deterministic core design seams, without global monkeypatching. |
| Retries | Small bounded, classified S3 retry loop | tenacity/backoff | Only a narrow 409/conditional-conflict path needs retry; a framework would make lifecycle boundaries less explicit. |
| Advanced testing | Hypothesis state machines + deterministic fault injection | Chaos framework, mutation testing, or flaky reruns now | Model/fault tests directly target the rewrite risks. Mutation/chaos can be evaluated after contracts stabilize; reruns hide races and must not be a gate. |

## Installation

```bash
# Runtime additions/changes
uv add "filelock>=3.20"

# Add Alembic to the existing SQLAlchemy-bearing recommended/postgresql/cloud extras,
# and raise the existing S3 extra floor for conditional PutObject support:
# alembic>=1.19
# boto3>=1.37.32

# Dev and quality-gate additions (uv.lock records exact resolutions)
uv add --group dev \
  "pytest~=9.1.1" "pytest-cov~=7.1" "coverage[toml]~=7.15" \
  "hypothesis~=6.165" "pytest-xdist~=3.8" "pytest-timeout==2.4.0" \
  "mypy~=2.3" "import-linter~=2.13" "ruff~=0.16.4" \
  "pytest-benchmark~=5.2" "pip-audit~=2.10"
```

Library metadata should express functional lower bounds without unnecessarily narrow upper caps; `uv.lock` supplies reproducible exact versions for development and CI. Compatibility CI should exercise both the declared lower-bound set and the locked-current set.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Runtime additions | MEDIUM | Current versions and capabilities are verified against official docs/PyPI; exact interaction with legacy backend shapes requires implementation-phase probes. |
| Lifecycle primitives | HIGH | Derived from the project's explicit atomicity/concurrency requirements and direct codebase failure modes, using standard CAS/immutable-generation patterns. |
| Test/CI stack | MEDIUM | Tool releases and official CI capabilities are current; final job durations and benchmark variance must be measured in this repository. |
| Coverage/performance thresholds | MEDIUM | Thresholds are prescriptive release targets anchored to the measured 66% baseline; they should be ratcheted with observed branch coverage and CI variance. |
| Optional backend matrix | MEDIUM | PostgreSQL and AWS mechanisms are official; live credentials, supported PostgreSQL majors, and S3-compatible endpoint guarantees remain project policy decisions. |

## Sources

- [Alembic 1.19.1 documentation](https://alembic.sqlalchemy.org/en/latest/) and [SQLite batch migrations](https://alembic.sqlalchemy.org/en/latest/batch.html) — **MEDIUM**, official documentation; current as of 2026-08.
- [Alembic programmatic commands](https://alembic.sqlalchemy.org/en/latest/api/commands.html) — **MEDIUM**, official documentation.
- [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace) — **MEDIUM**, official Python documentation.
- [Boto3 S3 `put_object`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html) — **MEDIUM**, official AWS SDK documentation for checksums and conditional writes.
- [Hypothesis stateful testing](https://hypothesis.readthedocs.io/en/latest/stateful.html) — **MEDIUM**, official documentation; 6.165.x current.
- [pytest 9 configuration and strict mode](https://docs.pytest.org/en/latest/reference/reference.html#configuration-options) — **MEDIUM**, official documentation; 9.1.1 current.
- [pytest-xdist distribution](https://pytest-xdist.readthedocs.io/en/stable/distribution.html) and [worker isolation guidance](https://pytest-xdist.readthedocs.io/en/stable/how-to.html) — **MEDIUM**, official documentation.
- [pytest-timeout](https://pypi.org/project/pytest-timeout/) — **MEDIUM**, project release metadata; confirms 2.4.0 current usable release and 2.5.0 yanked.
- [Coverage.py branch coverage](https://coverage.readthedocs.io/en/latest/branch.html) and [fail-under reporting](https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html) — **MEDIUM**, official documentation; 7.15.4 current.
- [pytest-cov 7.1](https://pypi.org/project/pytest-cov/) — **MEDIUM**, official project release metadata and subprocess-coverage guidance.
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/) and [rule catalog](https://docs.astral.sh/ruff/rules/) — **MEDIUM**, official documentation; 0.16.4 current.
- [mypy strict flags](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict) — **MEDIUM**, official documentation; 2.3.1 current.
- [Import Linter](https://pypi.org/project/import-linter/) — **MEDIUM**, project release metadata; 2.13 current.
- [pytest-benchmark comparison controls](https://pytest-benchmark.readthedocs.io/en/latest/usage.html) — **MEDIUM**, official project documentation; 5.2.3 current.
- [pip-audit](https://pypi.org/project/pip-audit/) — **MEDIUM**, PyPA project release metadata; 2.10.1 current.
- [Using uv in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/) — **MEDIUM**, official uv documentation, updated 2026-08.
- [GitHub Actions PostgreSQL service containers](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers) — **MEDIUM**, official GitHub documentation.
- [PyPA `pyproject.toml` specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/) and [dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/) — **MEDIUM**, official packaging specifications.
- [Typing specification: distributing type information](https://typing.python.org/en/latest/spec/distributing.html) — **MEDIUM**, official typing specification for `py.typed`.
