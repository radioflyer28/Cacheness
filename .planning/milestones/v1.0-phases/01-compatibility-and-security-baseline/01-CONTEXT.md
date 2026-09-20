# Phase 1: Compatibility and Security Baseline - Context

**Gathered:** 2026-08-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Freeze the supported Cacheness public behavior and stored-format baseline, preserve `SqlCache` as an independently tested subsystem, and close the currently exposed filesystem-containment, metadata-parser, metadata-query, and serializer-trust gaps before lifecycle ownership changes. This phase characterizes and hardens existing boundaries; it does not introduce the canonical manifest, rework lifecycle ownership, or migrate stored data.

</domain>

<decisions>
## Implementation Decisions

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Contract

- `.planning/PROJECT.md` — Core value, architecture boundary, compatibility promise, trusted-payload model, and milestone constraints.
- `.planning/REQUIREMENTS.md` — Phase 1 requirements `MIGR-01`, `CACH-07`, `SECU-01`, `SECU-02`, `SECU-06`, and `SECU-07`.
- `.planning/ROADMAP.md` — Phase boundary, dependency order, goal, and success criteria.

### Codebase Evidence

- `.planning/codebase/ARCHITECTURE.md` — Existing public surfaces, lifecycle boundaries, storage layers, and integration points.
- `.planning/codebase/CONCERNS.md` — Reproduced path traversal, `eval`, `allow_pickle`, query construction, compatibility, and `SqlCache` failure concerns.
- `.planning/codebase/TESTING.md` — Existing characterization patterns, measured baseline, and missing security/compatibility coverage.

No external specification or ADR was referenced during the discussion.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/error_handling.py`: Existing `CacheError` hierarchy can host stable security, query-validation, optional-dependency, and partial-fetch categories without exposing raw backend exceptions.
- `src/cacheness/__init__.py`: Central `__all__`, compatibility aliases, and conditional optional exports provide the inventory source for public API characterization.
- `tests/test_error_handling.py`, `tests/test_config_validation.py`, and `tests/test_sql_cache.py`: Existing pytest style and fixtures can be extended into executable compatibility and failure-contract tests.
- `tests/test_metadata_backend_registry.py` and `tests/test_blob_backend_registry.py`: Registry isolation patterns are reusable for import/export and public registration characterization.

### Established Patterns

- Public convenience imports and backward-compatible aliases are centralized in `src/cacheness/__init__.py`, but optional exports currently vary with the installed environment.
- Domain exceptions usually preserve causes with `raise ... from e`, while older storage and `SqlCache` paths still swallow broad exceptions, print warnings, or translate failures into misses.
- Tests use temporary directories and real local filesystem/SQLite operations, with mocks at optional-service boundaries.
- Configuration currently resolves paths while loading, which caused the two observed YAML relative-path round-trip failures.

### Integration Points

- `src/cacheness/storage/backends/blob_backends.py`: `FilesystemBlobBackend._get_blob_path()` currently rewrites traversal tokens and direct read/delete/exists methods accept arbitrary persisted paths; containment must be enforced at this backend boundary.
- `src/cacheness/storage/blob_store.py` and `src/cacheness/core.py`: Persisted `actual_path` values are resolved and accessed directly; unsafe legacy locators must surface typed failures without mutation.
- `src/cacheness/handlers.py`: `ArrayHandler._read_blosc2_array()` evaluates a Cacheness-defined shape string, and the NPZ path enables pickle globally.
- `src/cacheness/core.py`: `query_meta()` interpolates JSON field paths into SQL while binding only values; preserve documented comparison behavior while validating/building field expressions safely.
- `src/cacheness/sql_cache.py`: Missing-range fetch, bulk-upsert fallback, and custom gap-detector failures currently use `print()` and can silently change completeness or behavior.
- `pyproject.toml`, `src/cacheness/__init__.py`, and `docs/`: Package metadata, exports, examples, and documentation together define the characterization inventory for version `0.3.14`.

</code_context>

<specifics>
## Specific Ideas

- The original handler model is intentional: NumPy, Blosc2, PyArrow, pickle, and similar integrations perform their own serialization/container operations; Cacheness orchestrates those handlers.
- Blosc2 has two distinct valid roles: compress already serialized pickle bytes, and store Blosc2-compatible data structures through its native array format and compression support.
- The current custom array header was not required by unsupported data structures. It resulted from compressing `array.tobytes()` directly and separately recording lost shape/dtype information, and it must not become the future format.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within the phase scope. Canonical manifest design and selection of future native array formats remain part of Phase 2 as already roadmapped, not newly added scope.

</deferred>

---

*Phase: 01-compatibility-and-security-baseline*
*Context gathered: 2026-08-29*
