# Phase 1: Compatibility and Security Baseline - Research

**Researched:** 2026-08-29
**Domain:** Python library compatibility characterization, filesystem containment, safe legacy parsing, SQL query construction, and serializer trust boundaries
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Compatibility Envelope

- **D-01:** The supported public API baseline comprises package exports, documented import paths and aliases, documented signatures, and behavior exercised by public examples and tests. Private internals and accidental quirks are not contractual. — **Reversibility:** costly — Narrowing this baseline later requires deprecation across published call sites.
- **D-02:** Characterize current `0.3.14` stored layouts plus representative earlier `0.3.x` layouts that can be identified and read safely. Historical formats that cannot be safely or reliably identified are not promised.
- **D-03:** Preserve safe public behavior through compatibility adapters and deprecation warnings. Unsafe input behavior may change immediately and must fail with a clear typed error.
- **D-04:** Historically supported top-level optional symbols remain importable even when their dependency is absent. Construction or use of the unavailable feature raises an actionable dependency error.
- **D-05:** Compatibility adapters remain for the full milestone and may be removed only through a separately announced future major release or explicit `1.0` contract change. — **Reversibility:** one-way — Shortening this window would break the published compatibility promise.
- **D-06:** Characterization tests assert intended corrected behavior and document intentional compatibility changes; known defects are not frozen as contracts.
- **D-07:** Configuration serialization preserves user-authored relative paths. Paths are resolved separately when runtime filesystem operations need an absolute location.
- **D-08:** Public exception categories/classes and stable machine-readable reason codes are contractual. Exact prose may improve as long as it remains actionable. — **Reversibility:** costly — Changing machine-readable reasons affects caller branching and monitoring.

### Path-Containment Failures

- **D-09:** An unsafe key or persisted locator raises a typed unsafe-path error for every operation, including reads, `exists`, delete, and list. A security violation is never translated into a miss.
- **D-10:** If existing metadata points outside the configured root, refuse access and leave metadata and payload untouched for later migration or reconciliation. Phase 1 must not silently delete, copy, or rewrite it.
- **D-11:** Resolve the configured storage root itself, allowing the configured root to be a symlink or mount. Reject symlinks within managed entry paths so later retargeting cannot escape containment.
- **D-12:** Filesystem backends accept only opaque backend-safe identifiers from a restricted character set. User-facing logical keys are hashed or encoded before reaching the backend. — **Reversibility:** costly — Allowing path-shaped identifiers later would broaden the security and compatibility contract.

### Legacy Arrays and Safe Parsing

- **D-13:** Existing Cacheness-defined Blosc2 raw-array headers are legacy read-only compatibility artifacts. Decode their tuple metadata with a non-executing parser and validate structure and byte consistency before reconstruction.
- **D-14:** The normal NumPy array handler uses `allow_pickle=False`. Object-dtype arrays require an explicit trusted-object handler or configuration path with warnings and integrity enforcement.
- **D-15:** If the declared array artifact exists but is malformed or unsafe, fail with a typed format or security error. Do not silently try a different sidecar format.
- **D-16:** Native libraries and handlers own serialization and container formats. Cacheness coordinates handlers and records format metadata; it must not invent another raw-array container. Blosc2 may compress already serialized pickle bytes or persist Blosc2-native data structures through its own format. — **Reversibility:** costly — Reintroducing a custom container would create another stored format and migration obligation.
- **D-17:** Legacy header checks guard framing, rank, dimensions, dtype, and agreement with actual decompressed bytes. They do not create an unrelated default payload-filesize ceiling; an application-configured size policy may still apply.

### Metadata Queries and `SqlCache` Failure Contracts

- **D-18:** Preserve documented `query_meta()` semantics: numeric keyword filters mean greater-than-or-equal, while string filters mean exact match. Nested field paths remain supported only through strict validation.
- **D-19:** Unsupported or unsafe query-field syntax rejects the entire query before database access with a typed query-validation error identifying the invalid field.
- **D-20:** `SqlCache` missing-range fetch failures are strict by default and raise a typed error rather than returning silently incomplete data. An explicit best-effort mode may return partial data only while reporting every failed range. — **Reversibility:** costly — Callers may depend on the completeness guarantee once published.
- **D-21:** Equivalent internal fallbacks, such as bulk-to-row upsert, may proceed with structured logging. Failures in caller-supplied adapters or gap detectors raise unless the caller explicitly enabled fallback.

### the agent's Discretion

No product decisions were delegated. The planner may choose exact class names, reason-code representation, safe parser implementation, restricted identifier grammar, and structured-log fields within the decisions above. Selection of future native array formats belongs to Phase 2; Phase 1 only provides a safe legacy reader and prevents new writes in the custom format.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within the phase scope. Canonical manifest design and selection of future native array formats remain part of Phase 2 as already roadmapped, not newly added scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MIGR-01 | Public imports, constructors, configuration names, registries, decorators, aliases, exceptions, and result behavior have characterization tests before ownership changes. | Build an executable public-surface matrix from `__all__`, historical aliases, signatures, documented examples, and stored fixtures before changing implementations. [VERIFIED: .planning/REQUIREMENTS.md:49-57] |
| CACH-07 | `SqlCache` remains a separate subsystem and retains representative import and behavioral regression coverage. | Preserve the independent `SqlCache`/adapter boundary, then add success, strict-fetch, best-effort, gap-detector, and internal-upsert-fallback characterization. [VERIFIED: .planning/REQUIREMENTS.md:28-36] |
| SECU-01 | Filesystem reads, writes, deletes, and listings reject traversal, absolute-path, drive, UNC, and symlink escapes from the configured root. | Put one fail-closed containment guard at every filesystem backend and persisted-locator entry point; test hostile POSIX and Windows-shaped inputs plus ancestor and leaf symlinks. [VERIFIED: .planning/REQUIREMENTS.md:38-47] |
| SECU-02 | Structured metadata uses typed safe parsers, and metadata-controlled `eval` is eliminated. | Replace legacy raw-array `eval` with a bounded tuple parser and validate header framing, dtype, shape, decompressed length, and declared-format dispatch. [VERIFIED: .planning/REQUIREMENTS.md:38-47] |
| SECU-06 | Metadata query fields are validated and safely constructed rather than interpolated into backend queries. | Validate all field paths before opening a database session; build JSON-path expressions through SQLAlchemy parameters/expressions while preserving numeric GTE and string exact-match semantics. [VERIFIED: .planning/REQUIREMENTS.md:38-47] |
| SECU-07 | Documentation states the trusted-application-payload boundary and the risks and configuration requirements of unsafe serializers. | Document that both pickle and dill can execute code, ordinary NumPy loading disables pickle, and integrity/authenticity do not make hostile executable serialization safe. [VERIFIED: .planning/REQUIREMENTS.md:38-47] |
</phase_requirements>

## Summary

Phase 1 should be planned as four ordered workstreams: (1) freeze the corrected compatibility contract with executable tests and historical fixtures, (2) introduce shared typed boundary failures and filesystem containment, (3) remove unsafe parsing/query construction and make `SqlCache` completeness explicit, and (4) publish the trusted-payload documentation and run the full regression gate. This ordering prevents hardening work from accidentally freezing defects or changing the API without evidence. [VERIFIED: .planning/ROADMAP.md:20-30] [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

The most urgent defects are concrete: filesystem methods accept arbitrary persisted paths and opaque identifiers can escape through absolute paths or internal symlinks; legacy array metadata reaches `eval`; normal NPZ reads use `allow_pickle=True`; metadata query field names are interpolated into SQL text; and `SqlCache` prints and suppresses caller-controlled failures. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:219-324] [VERIFIED: src/cacheness/handlers.py:519-604] [VERIFIED: src/cacheness/core.py:535-651] [VERIFIED: src/cacheness/sql_cache.py:486-529,600-626,710-756]

Do not introduce a canonical manifest, lifecycle generation model, migration runner, or future native array format in this phase. Existing unsafe locators are rejected without mutation, and representative historical artifacts become read-only fixtures that later phases can migrate. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

**Primary recommendation:** Make the executable compatibility baseline the first plan, then harden each exposed boundary behind shared typed exceptions and finish with strict regression/documentation gates.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Public API characterization | Public API / compatibility facade | Tests / fixtures | `src/cacheness/__init__.py` owns exports and aliases; tests make the contract executable before internals move. [VERIFIED: AGENTS.md:191-206] |
| Filesystem containment | Storage backend boundary | Cache/BlobStore persisted-locator callers | The filesystem backend owns opaque-ID-to-path mapping, while current high-level callers also dereference metadata-controlled paths directly and therefore need the same guard. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:219-324] [VERIFIED: src/cacheness/storage/blob_store.py:196-242,285-365] [VERIFIED: src/cacheness/core.py:924-1053] |
| Legacy array decoding | Handler / serialization tier | Typed error boundary | `ArrayHandler` owns the legacy custom header and NPZ format dispatch. [VERIFIED: src/cacheness/handlers.py:426-610] |
| Metadata query validation | Cache policy/API boundary | SQLAlchemy metadata backend | `UnifiedCache.query_meta()` defines caller semantics; SQLAlchemy constructs and executes the backend expression. [VERIFIED: src/cacheness/core.py:535-651] |
| `SqlCache` completeness | Separate SQL pull-through subsystem | Caller-provided adapter | `SqlCache` finds ranges and owns transactions; adapters fetch data and must not have failures silently converted to partial success. [VERIFIED: src/cacheness/sql_cache.py:96-139,486-529] |
| Serializer trust documentation | Public documentation | Handler/configuration behavior | The project promises trusted application payloads, while handlers/config decide whether executable serializers are reachable. [VERIFIED: AGENTS.md:13-22] [VERIFIED: .planning/REQUIREMENTS.md:81-88] |

## Project Constraints (from AGENTS.md)

- Preserve supported public APIs and provide an explicit documented path for stored-data migration or rebuild. [VERIFIED: AGENTS.md:13-22]
- Keep `BlobStore` responsible for storage lifecycle, `UnifiedCache` responsible for cache policy, and `SqlCache` separate. [VERIFIED: AGENTS.md:13-22]
- Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries. [VERIFIED: AGENTS.md:13-22]
- Maintain Python support quoted verbatim as `">=3.11"`; the repository development pin is `"3.13"`. [VERIFIED: pyproject.toml:1-13] [VERIFIED: .python-version:1]
- Use package-relative imports internally, lazy/guarded optional dependency imports, domain-specific exceptions, narrow catches, preserved causes, structured logging, and `pytest.raises` with specific types. [VERIFIED: AGENTS.md:132-151]
- New modules use `snake_case.py`, tests use `test_<subject>.py`, public APIs have docstrings/type hints where practical, and package exports remain centralized in `__init__.py`. [VERIFIED: AGENTS.md:105-130,153-175]
- Run `uv run ruff check src tests`; do not increase the existing lint baseline or suppress findings without a local reason. [VERIFIED: AGENTS.md:116-130]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | `">=3.11"` | `pathlib`, `re`, `enum`, checked arithmetic, exceptions, logging | Already the supported runtime; no security helper package is needed for the fixed identifier grammar or tuple parser. [VERIFIED: pyproject.toml:1-13] |
| NumPy | `"2.3.2"` | NPZ loading and validated array reconstruction | Existing handler dependency; its loader defaults to `allow_pickle=False`. [VERIFIED: uv.lock:1056-1057] [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html] |
| SQLAlchemy | `"2.0.43"` | Bound JSON-path/query expressions and `SqlCache` | Existing optional SQL layer; Core expressions bind literal values rather than requiring string interpolation. [VERIFIED: uv.lock:1812-1813] [CITED: https://docs.sqlalchemy.org/en/20/tutorial/dbapi_transactions.html] |
| pytest | `"8.4.1"` | Compatibility, security, stored-fixture, and failure-contract tests | Existing configured test runner. [VERIFIED: uv.lock:1608-1609] [VERIFIED: pyproject.toml:68-99] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Blosc2 | `"3.7.0"` | Read and decompress legacy custom raw-array artifacts | Read-only legacy path in Phase 1; do not write the custom container. [VERIFIED: uv.lock:33-34] [VERIFIED: src/cacheness/handlers.py:519-577] |
| SQLite JSON1 | SQLite `"3.47.1"` in the audited environment | Exercise JSON extraction and query-path safety | Local integration tests for `query_meta`; JSON paths must be well-formed and malformed paths error. [VERIFIED: environment probe 2026-08-29] [CITED: https://www.sqlite.org/json1.html] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Fixed manual tuple parser | `ast.literal_eval` | Do not use it here: Python documents that sufficiently small malicious input can exhaust memory, C stack, or CPU and says calling it on untrusted data is not recommended. [CITED: https://docs.python.org/3.11/library/ast.html] |
| Shared `pathlib` containment guard | Ad hoc string replacement in every operation | Current replacement does not reject absolute paths or symlinks and produces inconsistent semantics; centralize the policy. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:306-324] |
| SQLAlchemy expressions and bound path/value parameters | `text()` with interpolated field names | Official SQLAlchemy guidance is to pass untrusted literal values separately rather than stringify them into SQL. [CITED: https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] |

**Installation:** No new external package should be installed for Phase 1. [VERIFIED: pyproject.toml:1-80]

## Package Legitimacy Audit

Not applicable: the recommended implementation uses only the standard library and packages already declared/locked by the project; Phase 1 should add no external package. [VERIFIED: pyproject.toml:1-80]

## Existing Baseline and Defects to Characterize

### Public surface

- The package version is quoted verbatim as `"0.3.14"`, and `__all__` is the central public inventory. [VERIFIED: src/cacheness/__init__.py:84-84,207-290]
- The current file defines `"SQLAlchemySqlCacheAdapter"` but exports `"SQLAlchemyDataAdapter"`; `from cacheness import *` raises `AttributeError`. Historical source at commit `207f483^` exported `"SQLAlchemyDataAdapter"`, so the corrected baseline should restore that alias without preserving the defect. [VERIFIED: src/cacheness/__init__.py:74-84,286-290] [VERIFIED: git history 207f483^:src/cacheness/__init__.py:45-53,82-86]
- Optional YAML helpers are exported whenever the helper module imports, although the actual `yaml` dependency is loaded only on use. Characterize symbols as importable and move dependency failure to use, as D-04 requires. [VERIFIED: src/cacheness/__init__.py:52-61,246-251] [VERIFIED: src/cacheness/config.py:1032-1071,1105-1144]
- Existing public exception names are quoted verbatim as `"CacheError"`, `"CacheConfigurationError"`, `"CacheStorageError"`, `"CacheSerializationError"`, `"CacheHandlerError"`, `"CacheIntegrityError"`, and `"CacheMetadataError"`; extend this hierarchy rather than exposing backend exceptions. [VERIFIED: src/cacheness/error_handling.py:18-65]

### Configuration and stored fixtures

- `CacheStorageConfig` currently preserves only the exact default `"./cache"`; every other relative path is converted to a working-directory absolute path during construction, breaking serialized round trips. [VERIFIED: src/cacheness/config.py:20-39]
- The direct targeted suite has two failures, both YAML path round trips (`TestYamlConfig.test_load_yaml_config` and `TestYamlConfig.test_save_yaml_config`), matching D-07 rather than a behavior to freeze. [VERIFIED: targeted pytest run 2026-08-29]
- Git history contains an unbroken version sequence from quoted `"0.3.0"` through `"0.3.14"` even though the repository has no release tags; later source commits through `a22f4b4` retain `"0.3.14"`. The exact commits selected for compatibility fixtures are resolved in `## Open Questions (RESOLVED)` below; they cover both legacy raw-array writer variants, unsigned/signed split JSON, JSON-backed SQLite metadata, the pre-unified decorator key, and both current metadata backends without promising every intervening release. [VERIFIED: git history commits 0015b48,aeb4dd1,2afe858,67246f5,0e3d81d,041c930,4d4ea6d,a756d70,71e4ba0,6e2f993,618ad37,61ced9d,81d4d2a,76a469f,8be1f66,a22f4b4]

### Filesystem boundary

- `FilesystemBlobBackend.read_blob`, `delete_blob`, `exists`, `read_blob_stream`, and `get_size` call `Path(blob_path)` directly; `_get_blob_path` only replaces `".."`, `"/"`, and `"\\"`, so absolute and symlink escapes remain reachable. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:250-324]
- A local probe read an outside file, returned `True` for outside `exists`, wrote an absolute blob ID outside the root, and wrote through an internal symlink outside the root. [VERIFIED: local tempfile containment probe 2026-08-29]
- The existing traversal test expects `"../escape"` to be silently rewritten to `"__"`; replace this with typed rejection because D-06 does not freeze the unsafe defect. [VERIFIED: tests/test_directory_sharding.py:274-288]
- `BlobStore.get` follows nested persisted `"actual_path"`, while delete/exists inspect a different top-level location and list returns metadata without validating locators. The Phase 1 guard must cover all these paths without redesigning the manifest. [VERIFIED: src/cacheness/storage/blob_store.py:196-242,285-365]
- `UnifiedCache.get` follows persisted `"actual_path"` and broad exception handlers can remove metadata or turn failures into misses; unsafe-path errors must bypass that cleanup/miss translation. [VERIFIED: src/cacheness/core.py:924-1053]

### Parser and query boundary

- The legacy header is exactly four little-endian bytes for shape-string length, UTF-8 `str(data.shape)`, four little-endian bytes for dtype-string length, UTF-8 `str(data.dtype)`, then Blosc2-compressed raw bytes. [VERIFIED: src/cacheness/handlers.py:519-553]
- The reader currently calls `eval(shape_str)`, does not bound header lengths, and reshapes without first proving decompressed byte consistency. [VERIFIED: src/cacheness/handlers.py:555-577]
- Declared `"blosc2"` read failures are swallowed before trying an NPZ sidecar, and NPZ is loaded with `allow_pickle=True`; both behaviors conflict with the locked fail-closed decisions. [VERIFIED: src/cacheness/handlers.py:579-604]
- Cache-key parameter serialization emits exact values such as `"int:75"`, `"float:0.95"`, `"str:xgboost"`, and `"bool:True"`; current query tests pass those serialized strings, so the numeric-GTE branch is not actually exercised by the apparent numeric test. [VERIFIED: src/cacheness/serialization.py:200-206] [VERIFIED: tests/test_serialization.py:18-21] [VERIFIED: tests/test_query_meta.py:103-122]
- `query_meta()` interpolates the caller field into `JSON_EXTRACT(cache_key_params, '$.{key}')`, binds only comparison values, and catches all exceptions as `None`. Validate every key before opening the session and do not translate validation failure to a miss-like `None`. [VERIFIED: src/cacheness/core.py:535-651]

### `SqlCache` boundary

- The subsystem exposes quoted classes `"SQLCacheError"` and `"MissingDependencyError"`, checks optional dependencies in construction, and is architecturally separate from `UnifiedCache`/`BlobStore`. [VERIFIED: src/cacheness/sql_cache.py:52-96,301-359]
- Missing-range adapter exceptions are printed and suppressed before commit, so callers can receive incomplete data as if complete. Bulk-upsert and custom gap-detector failures are also printed; only bulk-to-row fallback is equivalent enough to remain automatic. [VERIFIED: src/cacheness/sql_cache.py:486-529,600-626,710-756]
- `_find_internal_gaps()` currently returns `[]` unconditionally; characterize representative existing success behavior, but do not turn this stub into a full `SqlCache` redesign in Phase 1. [VERIFIED: src/cacheness/sql_cache.py:1148-1159] [VERIFIED: .planning/REQUIREMENTS.md:81-88]

## Architecture Patterns

### System Architecture Diagram

```text
Public imports/config/examples
          |
          v
Compatibility characterization ----------> immutable 0.3.x fixtures
          |
          v
Typed boundary errors + stable reason code
          |
          +--> logical key --> hash/encode --> opaque ID validator
          |                                      |
          |                                      v
          |                              resolved storage root
          |                                      |
          |                    reject traversal/absolute/drive/UNC/symlink
          |                                      |
          |                                      v
          |                               filesystem operation
          |
          +--> declared legacy array format --> bounded header parser
          |                                      |
          |                    validate shape/dtype/decompressed byte count
          |                                      |
          |                       valid ----------+---------- invalid
          |                         |                         |
          |                         v                         v
          |                    reconstruct                typed error
          |
          +--> query_meta fields --> validate all paths --> SQLAlchemy expression
          |                                                   |
          |                                 string exact / numeric >=
          |
          +--> SqlCache missing ranges --> fetch each range
                                                  |
                                strict failure ---+--- explicit best effort
                                      |                       |
                                 typed error          partial + all failures
```

### Recommended Project Structure

Keep changes near existing ownership boundaries; add one shared internal filesystem guard module and focused test modules. The proposed new paths `src/cacheness/storage/path_security.py`, `tests/test_public_api_contract.py`, `tests/test_stored_compatibility.py`, `tests/test_filesystem_containment.py`, `tests/test_legacy_array_security.py`, `tests/test_query_meta_security.py`, and `tests/test_sql_cache_failure_contract.py` are planner recommendations, not existing files. [ASSUMED]

```text
src/cacheness/
├── __init__.py                    # stable exports and aliases
├── error_handling.py              # public typed categories/reason code
├── config.py                      # authored path vs runtime path
├── handlers.py                    # legacy read-only array decoder
├── core.py                        # query_meta validation/expressions
├── sql_cache.py                   # strict/explicit-partial contract
└── storage/
    ├── path_security.py           # proposed shared containment guard [ASSUMED]
    ├── blob_store.py              # persisted-locator enforcement
    └── backends/blob_backends.py  # opaque identifiers + all FS operations
tests/
├── fixtures/compat/               # checked-in read-only historical artifacts [ASSUMED]
└── test_<boundary>.py             # focused contract/security suites [ASSUMED]
```

### Pattern 1: Validate Before Touching State

Validate the entire caller-controlled identifier, persisted locator, or query-field set before opening/altering a file, metadata entry, or database session. The unsafe-path branch must not run cleanup, and invalid query fields must reject the whole query before any database access. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

For filesystem operations, preserve the authored configuration string and resolve the root at backend initialization to one canonical absolute root. That deliberately follows an allowed root symlink once and prevents later retargeting of the configured root alias from changing the backend's authority. Every operation must then revalidate its restricted opaque relative identifier/locator and every managed component; it must never cache a previously validated absolute entry path. `Path.resolve()` eliminates `".."` and follows symlinks, but is only the lexical/initial part of the design, not the TOCTOU guarantee. [CITED: https://docs.python.org/3.11/library/pathlib.html]

On Unix platforms exposing the required capabilities, traverse from an open canonical-root directory descriptor, open each managed directory with `O_DIRECTORY | O_NOFOLLOW`, use descriptor-relative `os.open`/`os.stat`/`os.unlink`, and publish with descriptor-relative `os.rename`; capability detection must select the explicitly documented portable fallback rather than silently claiming descriptor protection. Python documents that `dir_fd` maps to `*at` operations on POSIX, that support must be detected with `os.supports_dir_fd`, and that `O_NOFOLLOW`/similar constants may be unavailable when the C library does not define them. [CITED: https://docs.python.org/3.11/library/os.html]

On Windows, where Python 3.11 documents that `dir_fd` does not work, validate every existing component with `os.lstat`, reject any component whose `st_file_attributes` includes `FILE_ATTRIBUTE_REPARSE_POINT` (including the known `IO_REPARSE_TAG_SYMLINK` and `IO_REPARSE_TAG_MOUNT_POINT` tags), resolve and contain immediately before the kernel operation, and serialize Cacheness-managed operations through the backend lock. [CITED: https://docs.python.org/3.11/library/os.html] [CITED: https://docs.python.org/3.11/library/stat.html]

### Pattern 2: Declared Format Selects One Reader

Use recorded storage format to select exactly one decoder. A declared legacy custom Blosc2 artifact gets the bounded legacy decoder; a declared NPZ artifact gets NumPy with pickle disabled. Missing, malformed, unsafe, and unsupported declared formats remain distinguishable typed outcomes and never cause sidecar guessing. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

The fixed legacy shape grammar only needs the producer's tuple forms: `()`, `(3,)`, and `(2, 3)` are representative values produced by `str(data.shape)`; parse ASCII digits, commas, parentheses, and whitespace manually with explicit rank/header bounds. Validate nonnegative dimensions, non-object dtype, checked element-count multiplication, and `element_count * dtype.itemsize == len(decompressed)` before `frombuffer(...).reshape(...)`. [VERIFIED: src/cacheness/handlers.py:542-553] [ASSUMED]

### Pattern 3: Compatibility Matrix, Not Snapshot Guessing

Inventory each public category (exports, documented import paths/aliases, constructors/signatures, configuration names and round trips, registries, decorators, exception categories/reasons, representative results) and map it to one or more executable tests. Generate historical artifacts once from identifiable commits and store both artifact plus provenance/expected-read metadata; do not make the test suite depend on network installs or historical environments. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md] [ASSUMED]

### Pattern 4: Strict by Default, Explicit Partial Result

`SqlCache.get_data()` should roll back and raise a typed, cause-preserving fetch error containing the failed range by default. A separately requested best-effort policy may continue, but must emit structured logging for every failed range and return caller-inspectable failure reporting with the partial data. The exact result-reporting representation is an implementation choice. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

### Anti-Patterns to Avoid

- **Sanitize and continue:** Rewriting hostile identifiers creates aliases and hides attacks; reject them with a stable typed reason. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:306-324]
- **Containment only on writes:** Persisted locators are metadata-controlled inputs, so reads, exists, delete, streams, sizes, and list-derived access require the same validation. [VERIFIED: src/cacheness/storage/backends/blob_backends.py:250-304]
- **`resolve()` without symlink policy:** A one-time resolution does not by itself encode D-11's rule that root symlinks are allowed while managed-path symlinks are forbidden. [CITED: https://docs.python.org/3.11/library/pathlib.html]
- **`literal_eval` as an untrusted parser:** It prevents code execution but remains vulnerable to resource exhaustion; the accepted legacy grammar is much smaller. [CITED: https://docs.python.org/3.11/library/ast.html]
- **Broad format fallback:** A malformed declared artifact must not be mistaken for a different sidecar. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]
- **Value-only SQL binding:** Binding the filter value does not make an interpolated JSON field path safe. [VERIFIED: src/cacheness/core.py:585-600]
- **Catching `Exception` into a miss/partial success:** Security, validation, adapter, and gap-detector failures are contractual outcomes, not cache misses. [VERIFIED: src/cacheness/core.py:649-651] [VERIFIED: src/cacheness/sql_cache.py:515-529,736-743]
- **Testing the current bug as compatibility:** Characterize intended corrected behavior and document the intentional break for unsafe inputs. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Path normalization primitives | Custom cross-platform slash/drive normalization | `pathlib.Path`, `PureWindowsPath`, a small explicit identifier regex, and one project policy guard | Python already models path flavors, resolution, and relative relationships; project code should only add its locked policy. [CITED: https://docs.python.org/3.11/library/pathlib.html] |
| Array containers | Another shape/dtype/raw-byte framing format | NPZ with `allow_pickle=False` for the current ordinary path; legacy custom format read-only | New framing creates another migration obligation and violates D-16. [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html] [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md] |
| SQL quoting/escaping | Identifier escaping or string-built JSON expressions | Strict field grammar plus SQLAlchemy expressions/bind parameters | Escaping is dialect-sensitive and easy to apply incompletely. [CITED: https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] |
| Hostile pickle safety | A “safe unpickler” claim for arbitrary object graphs | Trusted-application-payload boundary, integrity/authenticity controls, and opt-in unsafe object handling | Python states pickle is not secure and malicious data can execute arbitrary code. [CITED: https://docs.python.org/3.11/library/pickle.html] |
| Migration machinery | In-place rewrites while adding compatibility fixtures | Read-only historical fixtures and refusal of unsafe legacy locators | Inventory/migration/copy-verify-switch belong to Phase 7. [VERIFIED: .planning/ROADMAP.md:90-99] |

**Key insight:** Phase 1 should hand-roll only the tiny project-specific policy surfaces: opaque identifier grammar, bounded legacy tuple grammar, stable reason enum, and strict/partial failure mode. Path operations, SQL expression building, and array containers remain owned by standard/native libraries.

## Common Pitfalls

### Pitfall 1: Validation Mutates Evidence

**What goes wrong:** A read of an outside persisted locator reaches generic corrupt-entry cleanup and deletes metadata, or a delete follows the locator outside the root. [VERIFIED: src/cacheness/core.py:962-1053]

**Why it happens:** Security validation occurs after path use or inside broad exception handling.

**How to avoid:** Validate first; raise the typed unsafe-path exception outside generic miss/corrupt translation; assert both outside payload and metadata remain unchanged.

**Warning signs:** Tests assert “miss,” `False`, metadata removal, or file cleanup for hostile paths.

### Pitfall 2: Windows-Shaped Inputs Pass on POSIX

**What goes wrong:** `C:\\...`, UNC shares, or rooted backslash paths are treated as harmless POSIX filenames and later behave differently on Windows. [CITED: https://docs.python.org/3.11/library/pathlib.html]

**How to avoid:** Validate with both POSIX and Windows path semantics independent of host OS; add drive, UNC, mixed-separator, and rooted-backslash cases.

### Pitfall 3: Symlink Check Has a Time-of-Check Gap

**What goes wrong:** A managed path component can be replaced after a path-based validation but before the filesystem call. `Path.resolve()` alone therefore cannot prove race-safe access. [CITED: https://docs.python.org/3.11/library/pathlib.html]

**How to avoid:** Use descriptor-relative, no-follow access on capable Unix platforms; on Windows use `lstat` reparse-point rejection plus immediate pre-operation resolution/containment and a backend lock. Test stale-validation and deterministic retarget seams on every platform, and separately run a POSIX adversarial swap stress test against the descriptor implementation. [CITED: https://docs.python.org/3.11/library/os.html] [CITED: https://docs.python.org/3.11/library/stat.html]

### Pitfall 4: Numeric Query Tests Exercise Strings

**What goes wrong:** The apparent numeric GTE test passes `"int:75"`, so `query_meta` performs exact string equality and numeric semantics remain untested. [VERIFIED: tests/test_query_meta.py:103-122] [VERIFIED: src/cacheness/serialization.py:200-206]

**How to avoid:** Test raw `int`/`float` callers across positive, negative, decimal, and bool values; separately test raw strings and already-serialized legacy inputs.

### Pitfall 5: Best Effort Has No Evidence

**What goes wrong:** Partial `SqlCache` data is indistinguishable from complete data, and warnings are not machine-inspectable. [VERIFIED: src/cacheness/sql_cache.py:486-529]

**How to avoid:** Strict mode is default; best effort explicitly reports every failed range in a stable result channel and structured logs. Test multiple failures, not only one.

### Pitfall 6: Object Arrays Re-enter Pickle Indirectly

**What goes wrong:** Switching to `allow_pickle=False` breaks object arrays and a fallback silently re-enables pickle. NumPy documents object-array loading as a pickle-dependent path. [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html]

**How to avoid:** Detect object dtype before write/read, reject it on the ordinary handler, and require an explicit trusted-object path with warnings and integrity requirements.

### Pitfall 7: Optional Symbols Disappear in Minimal Environments

**What goes wrong:** `__all__` and import behavior depend on installed extras, so downstream code cannot even import a historical name to branch or defer use. [VERIFIED: src/cacheness/__init__.py:52-94,246-290]

**How to avoid:** Define lightweight symbols regardless of optional packages and move dependency checks to construction/use; exercise absence in an isolated subprocess or import-blocking test.

## Code Examples

### Safe SQLAlchemy JSON-path construction

```python
# Pattern derived from SQLAlchemy Core parameter binding and SQLite JSON1.
# Field grammar and exact exception/reason names are project policy [ASSUMED].
validate_all_fields_before_session(filters)
path_parameter = bindparam("json_path")
extracted = func.json_extract(CacheEntry.cache_key_params, path_parameter)
statement = select(CacheEntry).where(extracted == bindparam("query_value"))
```

Literal values should be passed separately from SQL text; validate field syntax before converting it to a JSON path. [CITED: https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] [CITED: https://www.sqlite.org/json1.html]

### Fail-closed NPZ loading

```python
# Source: NumPy 2.3 official numpy.load documentation.
with np.load(npz_path, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
```

`NpzFile` is closable/context-manageable, and object arrays fail rather than enabling executable pickle behavior. [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html]

### Runtime root resolution separated from serialized configuration

```python
# Proposed project pattern; exact helper name is discretionary [ASSUMED].
authored_cache_dir = config.storage.cache_dir
resolved_root = Path(authored_cache_dir).resolve(strict=False)
```

Keep `authored_cache_dir` unchanged for JSON/YAML round trips; use `resolved_root` only at filesystem operations. `Path.resolve()` makes an absolute path, resolves symlinks, and removes `".."`. [CITED: https://docs.python.org/3.11/library/pathlib.html]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NumPy loads object arrays by default | `numpy.load(..., allow_pickle=False)` defaults fail closed | Default documented in current NumPy 2.3 API | Ordinary cache arrays should not opt back into pickle globally. [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html] |
| “Safe” literal parsing as a general replacement for `eval` | Fixed grammar parsers for untrusted, narrowly structured metadata | Python docs now explicitly warn about `literal_eval` resource exhaustion | Legacy shape metadata should use a bounded tuple parser. [CITED: https://docs.python.org/3.11/library/ast.html] |
| String-built SQL fragments | Expression APIs plus bound literal parameters | SQLAlchemy 2.0 Core style | Query field grammar stays project-controlled and values stay out of SQL strings. [CITED: https://docs.sqlalchemy.org/en/20/tutorial/dbapi_transactions.html] |
| ASVS 4.x category numbering | ASVS `"5.0.0"` is the current stable release | May 2025 release | Apply current file-storage, validation, and safe-deserialization controls rather than copying obsolete category tables. [CITED: https://github.com/OWASP/ASVS/releases/tag/v5.0.0] |

**Deprecated/outdated:**

- New writes of the Cacheness custom Blosc2 raw header: retain only a safe legacy reader in Phase 1. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]
- `eval`/`literal_eval` for legacy shape strings: replace with a bounded fixed grammar. [CITED: https://docs.python.org/3.11/library/ast.html]
- `print()` for library error reporting: replace with typed exceptions and structured logger calls. [VERIFIED: AGENTS.md:137-151]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Add one shared internal module at `src/cacheness/storage/path_security.py` and focused new test files. | Recommended Project Structure | Existing module placement may offer a cleaner import boundary; behavior and ownership matter more than the filename. |
| A2 | A manual fixed tuple grammar is the simplest safe parser. | Architecture Patterns | An equally bounded parser could be acceptable, but `ast.literal_eval` must not be treated as resource-safe for untrusted metadata. |
| A3 | Exact new exception names, reason-code values, best-effort reporting representation, and logging field names remain planner choices. | Architecture Patterns / Code Examples | These become contractual under D-08 once selected, so lock them in the plan before implementation. |

## Open Questions (RESOLVED)

### Resolution 1: Historical Fixture Matrix

**Planning input:** Phase 1 must check in and characterize the following eight fixture variants. Fixture names are normative planning labels; full commit hashes make provenance unambiguous despite the absence of release tags.

| Fixture ID | Source commit / version | Exact stored variant | Safe identification and required assertion |
|------------|-------------------------|----------------------|--------------------------------------------|
| `array-raw-v035-compress` | `041c930fb66c7aa23f53d1f9f524e9fafdd20e68` / `"0.3.5"` | Numeric `int32` `(2, 3)` array using the Cacheness four-byte shape length, tuple text, four-byte dtype length, dtype text, and the older `blosc2.compress` frame in `.b2nd` | Identify only when surrounding metadata declares `"storage_format": "blosc2"` and the entire bounded header/frame validates; decode with the new non-evaluating reader and compare dtype, shape, values, and decompressed byte count. [VERIFIED: git history 041c930:src/cacheness/__init__.py,handlers.py] |
| `array-raw-v037-compress2` | `a756d70c858cec13ff1c885e2316c0fe725c4949` / `"0.3.7"` | Same numeric array/header with the newer `blosc2.compress2` frame | Apply the same declared-format and full-validation gate; prove both historical frame variants decode without `eval`. A current-environment probe confirmed `blosc2.decompress2` reads frames produced by both `compress` and `compress2`. [VERIFIED: git history a756d70:src/cacheness/__init__.py,handlers.py] [VERIFIED: Blosc2 compatibility probe 2026-08-29] |
| `json-split-unsigned-v037` | `a756d70c858cec13ff1c885e2316c0fe725c4949` / `"0.3.7"` | Split-map JSON metadata plus a numeric NPZ payload, with top-level maps quoted verbatim as `"entries"`, `"access_times"`, `"creation_times"`, `"file_sizes"`, `"data_types"`, `"cache_key_params"`, `"cache_hits"`, and `"cache_misses"`; no `"entry_signature"` | Identify by the complete split-map key set, normalize one entry read-only to the current entry shape, validate its locator before opening the NPZ, and leave the JSON unchanged. [VERIFIED: git history a756d70:src/cacheness/metadata.py] |
| `json-split-signed-v038` | `71e4ba04cbcd7dfbcf74e5651f131213b7d45ab9` / `"0.3.8"` | Same split-map JSON with quoted optional field `"entry_signature"` generated from a fixed test-only HMAC key, plus numeric NPZ | Identify by the split-map key set and signature field; characterize successful read with the fixture key and a typed integrity failure with a wrong key, without changing the Phase 2 signing contract. [VERIFIED: git history 71e4ba0:src/cacheness/__init__.py,metadata.py,core.py,security.py] |
| `sqlite-metadata-json-v039` | `6e2f9933a9aa66fb6629aec0125f84a0c114c35f` / `"0.3.9"` | SQLite `"cache_entries"` row whose quoted columns are `"cache_key"`, `"description"`, `"data_type"`, `"prefix"`, `"created_at"`, `"accessed_at"`, `"file_size"`, `"file_hash"`, `"entry_signature"`, `"cache_key_params"`, and `"metadata_json"`, plus numeric NPZ | Open SQLite in read-only mode, identify with `PRAGMA table_info(cache_entries)` and presence of `"metadata_json"`, parse that column as JSON only, normalize one entry, validate locator, and prove no schema/data mutation. [VERIFIED: git history 6e2f993:src/cacheness/metadata.py:75-134] |
| `decorator-key-v0313` | `76a469f3e090a99d9f9c119c3190a97b11a5e68a` / `"0.3.13"` | Current-family nested JSON metadata plus numeric NPZ stored under the old decorator hash based on serialized `args` and `kwargs` | Identify by recomputing the documented old decorator candidate for one stable fixture function; assert same-call lookup can read it through the compatibility key adapter and emits the selected deprecation signal, without scanning arbitrary entries. [VERIFIED: git history 76a469f:src/cacheness/decorators.py:1-56] |
| `json-nested-v0314` | `a22f4b4575cb8213d9783ed388d2a70727563db1` / current source at `"0.3.14"` | Current nested JSON entry with quoted top-level keys `"entries"`, `"cache_hits"`, and `"cache_misses"`, plus numeric NPZ under the unified key | Exercise current JSON round trip, exact metadata/result behavior, safe locator validation, and current key generation. The unified key originated at `8be1f66`; the fixture itself must come from the last source commit in the audited tree, not merely the version-bump commit. [VERIFIED: src/cacheness/metadata.py:691-778] [VERIFIED: git history 8be1f66..a22f4b4:src/cacheness] |
| `sqlite-columns-v0314` | `a22f4b4575cb8213d9783ed388d2a70727563db1` / current source at `"0.3.14"` | Denormalized SQLite row whose quoted entry columns are `"cache_key"`, `"description"`, `"data_type"`, `"prefix"`, `"created_at"`, `"accessed_at"`, `"file_size"`, `"file_hash"`, `"entry_signature"`, `"object_type"`, `"storage_format"`, `"serializer"`, `"compression_codec"`, `"actual_path"`, and `"cache_key_params"`, plus numeric NPZ | Identify by `PRAGMA table_info(cache_entries)` and absence of `"metadata_json"`; exercise current read behavior and locator validation. The denormalized shape was introduced at `"0.3.10"` and remains the stored schema in the audited `"0.3.14"` source. [VERIFIED: src/cacheness/metadata.py:76-131] [VERIFIED: git history 618ad37..a22f4b4:src/cacheness/metadata.py] |

**Generation protocol:** Create a disposable detached worktree for each full commit, use an isolated environment with the repository's locked current test dependencies, run only a small writer script over fixed non-object NumPy data, and copy the resulting artifact directory into `tests/fixtures/compat/`. The writer script must never call any historical read method. Record source commit, Python/NumPy/Blosc2 versions, logical test input, file list, and SHA-256 for every file in a test-only provenance JSON. Then remove the worktree. This creates exact Cacheness layout transitions without installing an unverified historical package from a registry.

**Pre-read safety gate:** Before a fixture reaches a Cacheness reader, inspect JSON with `json.load`, inspect SQLite through a read-only URI plus `PRAGMA table_info`, inspect NPZ through `zipfile`/`numpy.load(..., allow_pickle=False)`, and inspect `.b2nd` only with the new bounded header parser and Blosc2 decompression. The fixture matrix intentionally contains no pickle, dill, object-dtype array, TensorFlow, dataframe, S3, PostgreSQL, or unknown historical artifact. Those are not added to the stored-data compatibility promise; trusted pickle/dill behavior remains ordinary in-process characterization plus documentation, and remote/backend migration remains later-phase work. [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html] [VERIFIED: .planning/ROADMAP.md:90-110]

**Read-only acceptance rule:** Every historical fixture test copies its fixture to a temporary directory, snapshots hashes/SQLite `PRAGMA data_version` and schema, performs the read, and proves the source fixture remains byte-for-byte unchanged. Any candidate that cannot pass the pre-read gate or cannot be normalized without guessing is rejected with a typed unsupported-format result and is outside the promised matrix. No runtime code infers a release number from a payload that does not carry one.

### Resolution 2: Symlink Retargeting and TOCTOU Threat Model

**Planning input:** Phase 1 must defend, on every supported platform, against (a) traversal/absolute/drive/UNC/rooted identifiers, (b) a symlink, junction, mount-point reparse entry, or outside locator already present when an operation begins, (c) retargeting/replacement between two Cacheness operations, and (d) deterministic same-process retargeting injected after initial path construction but before the final guarded filesystem call. Every outcome is either the intended operation inside the canonical root or a typed unsafe-path failure; outside data and metadata remain untouched. This applies to read, write, stream, exists, size, delete, and list-derived locator access.

**Canonical root rule:** Resolve the configured root once at backend initialization, store the canonical absolute root, and—on Unix—open its directory descriptor. A symlink used as the configured root is therefore allowed and anchored to its initialization target. Every managed entry path below that root is revalidated per operation and may contain no symlink/reparse component. Retargeting the configured root alias later does not redirect the initialized backend; a newly constructed backend resolves the then-current target. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md] [CITED: https://docs.python.org/3.11/library/pathlib.html]

**Unix mechanism:** When `os.open`, `os.stat`, `os.unlink`, `os.mkdir`, and `os.rename` expose the necessary descriptor-relative capability and `O_NOFOLLOW`/`O_DIRECTORY` exist, walk/create the shard directory from the canonical root descriptor, opening every component no-follow; operate on the final opaque basename relative to the verified parent descriptor; use a unique temporary basename in that same verified directory; publish with descriptor-relative rename; and never reopen an attacker-switchable absolute parent path. Treat capability absence as the portable fallback below rather than silently claiming descriptor protection. [CITED: https://docs.python.org/3.11/library/os.html]

**Windows and capability fallback:** Python 3.11 states that `dir_fd` is Unix-only. Under the project’s trusted-host boundary, serialize Cacheness-managed operations with the backend lock, reject Windows reparse points via `os.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT` (including symlink and mount-point tags), resolve/contain immediately before each kernel call, use unique same-directory temporary files for writes, and repeat component/containment checks before publish/delete. [CITED: https://docs.python.org/3.11/library/os.html] [CITED: https://docs.python.org/3.11/library/stat.html]

**Required proof:**

1. A cross-platform deterministic test matrix inserts an ancestor link/junction, leaf link, broken link, and outside persisted locator before each operation and expects typed rejection with no mutation.
2. A cross-platform stale-validation test validates a normal path, swaps an ancestor to a link/reparse point, then invokes a fresh operation; it proves validation results are not cached.
3. A cross-platform deterministic intra-operation test uses an internal test seam/barrier to perform the swap after relative-path construction but before the final guard/open. On descriptor-capable Unix the operation must remain bound inside or fail; on Windows/fallback the final reparse/containment check must reject it.
4. A descriptor-capable Unix stress test repeatedly swaps the pathname concurrently while reads/writes/deletes run and proves no outside sentinel is read, changed, or deleted. This validates the external-race-resistant implementation rather than relying only on monkeypatch ordering.
5. Windows CI must exercise actual directory symlinks when privileges permit and must always exercise a junction/mount-point reparse fixture; a skipped symlink-creation test without the junction/reparse equivalent is not sufficient. Python 3.11 exposes the reparse attributes/tags needed for the assertion. [CITED: https://docs.python.org/3.11/library/stat.html]

**Explicit residual risk:** On Windows and any platform without descriptor-relative no-follow operations, a separate same-user/privileged process that can mutate the managed root may win the final check-to-kernel-call race. Python 3.11 provides no portable directory-descriptor primitive on Windows. That hostile external kernel race is outside Phase 1’s trusted-host/storage-root authority boundary; deployments requiring protection from such a local adversary must deny that actor write access with OS ACLs or use a descriptor-capable Unix filesystem. This residual does not weaken SECU-01 for caller-controlled keys, persisted metadata, pre-existing links/reparse points, between-operation retargeting, or deterministic same-process races, all of which remain mandatory and tested. [CITED: https://docs.python.org/3.11/library/os.html]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | all verification | ✓ | `"0.12.6"` | — [VERIFIED: environment probe 2026-08-29] |
| CPython via `uv` | supported implementation/tests | ✓ | `"3.13.3"` | System `python3` is `"3.12.1"`; Python 3.11 still needs later matrix verification. [VERIFIED: environment probe 2026-08-29] |
| NumPy | array security tests | ✓ | `"2.3.2"` | — [VERIFIED: environment probe 2026-08-29] |
| Blosc2 | legacy custom-array fixtures | ✓ | `"3.7.0"` | Skip only tests marked optional if fixture decoding is explicitly unavailable. [VERIFIED: uv.lock:33-34] |
| SQLAlchemy | `query_meta` and `SqlCache` tests | ✓ | `"2.0.43"` | Optional-dependency absence is separately tested. [VERIFIED: environment probe 2026-08-29] |
| SQLite JSON1 | local query integration | ✓ | SQLite `"3.47.1"` | — [VERIFIED: environment probe 2026-08-29] |
| pytest | validation | ✓ | `"8.4.1"` | — [VERIFIED: environment probe 2026-08-29] |

**Missing dependencies with no fallback:** Python 3.11 and additional supported-version interpreters were not available in this single environment; cross-version CI belongs to Phase 8, but Phase 1 code must avoid 3.12/3.13-only syntax. [VERIFIED: pyproject.toml:1-13] [VERIFIED: .planning/ROADMAP.md:101-110]

**Missing dependencies with fallback:** No Phase 1 implementation blocker was found. Optional-dependency absence behavior should be tested through isolated imports/mocking rather than uninstalling the working environment. [ASSUMED]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest `"8.4.1"` [VERIFIED: uv.lock:1608-1609] |
| Config file | `pyproject.toml`; quoted discovery values are `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `python_classes = ["Test*"]`, `python_functions = ["test_*"]`. [VERIFIED: pyproject.toml:82-99] |
| Quick run command | `uv run pytest -q -o log_cli=false <phase-test-file> -x` |
| Full suite command | `uv run pytest -q -o log_cli=false` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| MIGR-01 | Every supported export/alias/import, constructor/signature, config round trip, registry, decorator, exception/reason, result, and selected historical artifact is executable | characterization + fixture integration | `uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_stored_compatibility.py -x` | ❌ Wave 0 [ASSUMED] |
| CACH-07 | Independent imports and pull-through success remain; adapter fetch/gap failures are strict unless explicit best effort; internal bulk fallback remains functional | unit + SQLite integration | `uv run pytest -q -o log_cli=false tests/test_sql_cache.py tests/test_sql_cache_failure_contract.py -x` | existing success file + ❌ Wave 0 [VERIFIED: tests/test_sql_cache.py:1-374] [ASSUMED] |
| SECU-01 | Every filesystem operation rejects traversal, POSIX absolute, drive, UNC, mixed separators, root/leaf/ancestor symlink escapes; root symlink is allowed; rejected persisted locators mutate nothing | unit + filesystem integration | `uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py -x` | ❌ Wave 0 [ASSUMED] |
| SECU-02 | Legacy header parser accepts valid scalar/1-D/n-D artifacts, rejects malformed lengths/grammar/rank/dim/dtype/byte mismatch, removes `eval`, disables NPZ pickle, and never sidecar-falls back | unit + stored fixture | `uv run pytest -q -o log_cli=false tests/test_legacy_array_security.py -x` | ❌ Wave 0 [ASSUMED] |
| SECU-06 | All query fields prevalidate before DB; hostile/nested-invalid fields reject atomically; string exact and raw int/float GTE semantics work; values and paths are bound safely | unit + SQLite integration | `uv run pytest -q -o log_cli=false tests/test_query_meta.py tests/test_query_meta_security.py -x` | existing behavior file + ❌ Wave 0 [VERIFIED: tests/test_query_meta.py:1-772] [ASSUMED] |
| SECU-07 | README/security docs state trusted payload boundary, pickle+dill code-execution risk, object-array opt-in, integrity limitation, and safe defaults | documentation assertion + review | `uv run pytest -q -o log_cli=false tests/test_security_documentation.py -x` | ❌ Wave 0 [ASSUMED] |

### Required Test Dimensions

- **No-mutation assertions:** snapshot metadata plus outside file bytes before hostile read/exists/delete/list and compare after the typed exception. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]
- **Cross-platform path corpus on every host:** POSIX absolute/traversal plus Windows drive, UNC, rooted backslash, and mixed separators; do not rely on CI host flavor to generate the corpus. [CITED: https://docs.python.org/3.11/library/pathlib.html]
- **Retarget race tiers:** run static link/reparse rejection, between-operation retarget, and deterministic intra-operation barrier tests on every platform; add the concurrent pathname-swap stress test on descriptor-capable Unix and a real junction/reparse test on Windows. [CITED: https://docs.python.org/3.11/library/os.html] [CITED: https://docs.python.org/3.11/library/stat.html]
- **Call-order spies:** prove invalid query fields are rejected before session creation/database execution. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md]
- **Negative array corpus:** truncated length words, oversized declared headers, invalid UTF-8, non-tuple grammar, negative/excessive dimensions, object dtype, decompression failure, byte mismatch, malformed declared artifact with valid sidecar present. [VERIFIED: src/cacheness/handlers.py:519-604] [ASSUMED]
- **Import matrix:** full environment, blocked YAML import, blocked SQLAlchemy/pandas, and `from cacheness import *`; every guaranteed name exists before dependency use. [VERIFIED: src/cacheness/__init__.py:52-94,207-290]
- **Strict/partial matrix:** one and multiple missing ranges, first/middle/last failure, fetch returns empty, custom gap detector throws, bulk upsert throws then row fallback succeeds/fails. [VERIFIED: src/cacheness/sql_cache.py:486-529,600-626,710-756]

### Sampling Rate

- **Per task commit:** the mapped quick command for the boundary being changed.
- **Per wave merge:** `uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_stored_compatibility.py tests/test_filesystem_containment.py tests/test_legacy_array_security.py tests/test_query_meta.py tests/test_query_meta_security.py tests/test_sql_cache.py tests/test_sql_cache_failure_contract.py tests/test_security_documentation.py` [ASSUMED]
- **Phase gate:** full suite green, then `uv run ruff check src tests`; no remaining metadata-controlled `eval`, `allow_pickle=True` on the ordinary array path, interpolated JSON path, or `print()` in `SqlCache` failure paths. [VERIFIED: AGENTS.md:116-151]

### Wave 0 Gaps

- [ ] Create the public API/fixture characterization modules before implementation. [ASSUMED]
- [ ] Create a reusable hostile path corpus and outside-root/symlink fixtures. [ASSUMED]
- [ ] Create valid and corrupt legacy raw-array fixtures without evaluating fixture metadata. [ASSUMED]
- [ ] Add session/engine spies for pre-database query rejection. [ASSUMED]
- [ ] Add deterministic failing adapters/gap detectors and a log capture fixture for `SqlCache`. [ASSUMED]
- [ ] Add documentation contract assertions or an explicit manual review checklist if prose tests prove too brittle. [ASSUMED]

## Security Domain

Security enforcement is enabled with quoted configuration values `"security_enforcement": true`, `"security_asvs_level": 1`, and `"security_block_on": "high"`. [VERIFIED: .planning/config.json:20-49]

### Applicable ASVS 5.0 Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V1.5 Safe Deserialization | yes | Do not deserialize hostile pickle/dill; use a bounded legacy metadata parser; ordinary NumPy path disables pickle. [CITED: https://github.com/OWASP/ASVS/tree/v5.0.0] [CITED: https://docs.python.org/3.11/library/pickle.html] |
| V2 Validation and Business Logic | yes | Strict field/identifier grammar, whole-request rejection, typed errors, and numeric/string semantic tests. [CITED: https://github.com/OWASP/ASVS/tree/v5.0.0] |
| V5.3 File Handling | yes | Restrict filenames/paths, resolve the configured root, reject traversal/absolute/drive/UNC and managed-path symlinks, and preserve rejected evidence. [CITED: https://github.com/OWASP/ASVS/tree/v5.0.0] |
| Authentication / Session Management | no | This is an in-process storage library with no authentication or session surface in Phase 1. [VERIFIED: AGENTS.md:3-22]
| Cryptography | no change in Phase 1 | Existing signing/integrity contract is hardened in Phase 2; Phase 1 documents that integrity does not make executable serialization safe. [VERIFIED: .planning/ROADMAP.md:32-41]

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Traversal/absolute/drive/UNC key escape | Tampering / Information Disclosure | Restricted opaque identifiers, backend hashing/encoding, runtime root containment, typed fail-closed rejection. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md] |
| Persisted locator outside root | Tampering / Information Disclosure | Validate before access; do not delete or rewrite metadata/payload. [VERIFIED: .planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md] |
| Managed-path symlink escape | Tampering / Elevation of Privilege | Resolve allowed root, forbid symlink components beneath it, verify containment immediately before operation. [CITED: https://docs.python.org/3.11/library/pathlib.html] |
| Metadata-controlled `eval` | Elevation of Privilege / Denial of Service | Bounded fixed grammar plus framing and byte-consistency checks. [VERIFIED: src/cacheness/handlers.py:562-577] |
| Pickle/object-array execution | Elevation of Privilege | Trusted-payload boundary, default `allow_pickle=False`, explicit unsafe opt-in only. [CITED: https://docs.python.org/3.11/library/pickle.html] [CITED: https://numpy.org/doc/2.3/reference/generated/numpy.load.html] |
| Query field injection | Tampering / Information Disclosure | Whole-field-set validation before DB and SQLAlchemy-bound expressions. [VERIFIED: src/cacheness/core.py:585-600] [CITED: https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] |
| Silent partial pull-through result | Repudiation / Integrity | Strict default typed failure; explicit best effort reports every failed range. [VERIFIED: src/cacheness/sql_cache.py:513-529] |

## Sources

### Primary (HIGH confidence)

- Repository source and tests opened in this session: `src/cacheness/__init__.py`, `config.py`, `core.py`, `error_handling.py`, `handlers.py`, `serialization.py`, `sql_cache.py`, `storage/blob_store.py`, `storage/backends/blob_backends.py`, `tests/test_query_meta.py`, `tests/test_sql_cache.py`, `tests/test_config_validation.py`, and `tests/test_directory_sharding.py`.
- Project contract: `AGENTS.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md`.
- Local runtime probes for path containment, imports, dependency versions, and targeted pytest baseline on 2026-08-29.

### Secondary (MEDIUM confidence)

- [Python 3.11 pathlib documentation](https://docs.python.org/3.11/library/pathlib.html) — resolution, symlinks, lexical relative checks, Windows path flavors.
- [Python 3.11 os documentation](https://docs.python.org/3.11/library/os.html) — descriptor-relative operations, `supports_dir_fd`, no-follow flags, and the explicit lack of Windows `dir_fd` support.
- [Python 3.11 stat documentation](https://docs.python.org/3.11/library/stat.html) — `lstat`, symlink mode checks, Windows reparse-point attributes, and reparse tags.
- [Python 3.11 AST documentation](https://docs.python.org/3.11/library/ast.html) — `literal_eval` non-execution and resource-exhaustion warnings.
- [Python 3.11 pickle documentation](https://docs.python.org/3.11/library/pickle.html) — arbitrary-code risk and trusted-data boundary.
- [NumPy 2.3 `numpy.load`](https://numpy.org/doc/2.3/reference/generated/numpy.load.html) — default `allow_pickle=False`, object arrays, closable NPZ files.
- [SQLite JSON1](https://www.sqlite.org/json1.html) — JSON path grammar and scalar extraction.
- [SQLAlchemy 2.0 Core/DBAPI tutorial](https://docs.sqlalchemy.org/en/20/tutorial/dbapi_transactions.html) and [SQL expression FAQ](https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html) — bound parameters and untrusted input.
- [OWASP ASVS 5.0.0](https://github.com/OWASP/ASVS/releases/tag/v5.0.0) — current stable application security verification categories.

### Tertiary (LOW confidence)

- None. All unresolved implementation choices are marked `[ASSUMED]` and listed in the Assumptions Log.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new dependencies; current versions were read from the lockfile and environment, and behavior came from official documentation.
- Architecture: HIGH — boundaries and defects were traced directly through current source, tests, local probes, and locked decisions.
- Pitfalls: HIGH — reproduced code paths plus a resolved cross-platform retarget/TOCTOU threat model grounded in Python 3.11 platform capabilities.
- Validation architecture: HIGH for required behaviors, fixture matrix, race-test tiers, and current test infrastructure; MEDIUM only for proposed new test filenames.

**Research date:** 2026-08-29
**Valid until:** 2026-09-28 for repository findings; re-check official dependency/API documentation if implementation occurs later.
