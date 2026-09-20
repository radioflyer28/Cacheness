<!-- refreshed: 2026-09-17 -->
# Codebase Concerns

**Analysis date:** 2026-09-17

This is a current risk map. It preserves the post-refactor boundary: `BlobStore`
owns lifecycle, while `UnifiedCache` owns only policy.

## High-priority boundaries

### Topology-specific rather than universal guarantees

- **Risk:** treating a common API as evidence of cross-resource ACID, lock-free
  global availability, or instantaneous orphan-free cleanup.
- **Why it matters:** immutable payload effects and authority transitions are
  different resources. Adding a queue, lock, bootstrap protocol, or projection
  gate to close every timing window recreates the coordination problem the
  architecture removed.
- **Required response:** read ADR 0001, state the topology and valid typed
  outcomes, then stop rather than broaden lifecycle coordination.
- **Evidence:** `docs/adr/0001-topology-specific-storage-guarantees.md`,
  `src/cacheness/storage/blob_store.py`,
  `src/cacheness/storage/reconciliation.py`.

### Remote qualification remains deferred

- **Risk:** local/mocked evidence is described as proof of live PostgreSQL or
  Amazon-S3 release behavior.
- **Impact:** remote authentication, service semantics, and operational failure
  modes can differ from local deterministic contracts.
- **Required response:** retain non-live tests and explicit capability claims;
  acquire real-service evidence in the dedicated future qualification work.
- **Evidence:** `tests/integration/test_remote_topology.py`,
  `tests/qualification/test_live_evidence.py`, `.planning/PROJECT.md`.

### Handler boundary is security-sensitive

- **Risk:** a custom handler bypasses contained staging/snapshot paths or
  treats an unverified descriptor as input.
- **Impact:** path escape, accidental authority bypass, unbounded-memory reads,
  or a corrupted artifact reaching a deserializer.
- **Required response:** use `store.handlers.register_handler(...)`, preserve
  suffix/containment validation, and rely on manifest digest/size checks before
  reconstruction.
- **Evidence:** `src/cacheness/storage/guarded_handler_io.py`,
  `src/cacheness/storage/path_security.py`, `src/cacheness/handlers.py`.

## Maintainability risks

### Large lifecycle authority implementations

- **Risk:** SQLite and PostgreSQL authority modules encode many explicit
  maintenance and migration transitions.
- **Impact:** a localized correctness change can alter a boundary shared by
  publication, cleanup debt, recovery, and offline maintenance.
- **Required response:** add focused contract tests and keep new behavior in
  the existing authority contract; do not introduce parallel ownership.
- **Evidence:** `src/cacheness/storage/sqlite_lifecycle_authority.py`,
  `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`,
  `tests/contracts/test_lifecycle_authority.py`.

### Optional dependency truthfulness

- **Risk:** an extra, guarded import, and package metadata drift apart.
- **Impact:** users can install a documented surface that cannot import or an
  artifact can accidentally lose a retained integration.
- **Required response:** update `pyproject.toml` and `uv.lock` together, run
  `uv lock --check`, and retain fresh isolated-wheel metadata probes, including
  the source-free check that rejects retired modules, metadata, and handler
  identities.
- **Evidence:** `pyproject.toml`, `tools/run_phase8_packaging.py`,
  `tests/packaging/test_wheel_matrix.py`.

### Global compatibility and registry state

- **Risk:** legacy re-export modules and process-level registries create
  ambiguous ownership or test leakage.
- **Impact:** a custom handler/backend can appear registered in an unrelated
  store or callers can depend on an implementation path rather than the public
  surface.
- **Required response:** prefer store-local handler registration, restore
  registry state in tests, and keep compatibility modules thin.
- **Evidence:** `src/cacheness/handlers.py`,
  `src/cacheness/storage/backends/__init__.py`,
  `tests/test_handler_registration.py`.

## Product and qualification gaps

- **Minimal install coverage:** base import and each optional integration need
  continued artifact-level verification; NumPy is intentionally a base
  dependency while the other handler integrations remain optional.
- **Live remote evidence:** real PostgreSQL/Amazon-S3 and immutable publication
  qualification are still future work, not local-release claims.
- **Performance evidence:** the benchmark harness and diagnostics exist, but
  controlled-Linux budgets remain future work.

## Historical lesson retained for future phases

The earlier sequence of race fixes exposed overlapping lifecycle mechanisms.
The durable remedy is not an ever-stronger filesystem/database/process-local
coordination stack: it is one declared authority per topology plus bounded,
observable reconciliation for payload effects. Future work must explicitly
state where a guarantee stops before changing lifecycle code.

---

*Current concerns map refreshed for the post-cut product boundary on 2026-09-19.*
