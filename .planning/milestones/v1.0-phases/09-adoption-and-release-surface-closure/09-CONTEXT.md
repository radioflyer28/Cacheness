# Phase 9: Adoption and Release Surface Closure - Context

**Gathered:** 2026-09-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 9 makes the completed BlobStore-first architecture accurately usable from
the repository's primary documentation, installation guidance, examples,
package presentation, and public format-handler surface. It closes the partial
`CACH-06` publication requirement for the locally qualified development version.

This phase does not reopen storage lifecycle, concurrency, backend composition,
or topology guarantees. It does not add compatibility shims, another lifecycle
authority, or stronger claims than ADR 0001 permits. Real PostgreSQL/Amazon-S3
qualification, immutable publication, controlled-Linux performance, and native
Windows qualification remain explicit future work.

</domain>

<decisions>
## Implementation Decisions

### Product story and documentation structure
- **D-01:** The README is a concise gateway rather than a comprehensive manual. It briefly establishes the BlobStore-first architecture and provides two quick starts: direct storage and caching through `UnifiedCache`.
- **D-02:** Linked documentation is organized first around user tasks—store objects, cache function results, add a file format, and operate or migrate a store—with focused component references for `BlobStore`, `UnifiedCache`, handlers, and related APIs.
- **D-03:** Documents whose primary purpose is the removed pre-cutover API are deleted from supported documentation rather than retained in a historical docs archive. Git history and planning artifacts preserve historical evidence.
- **D-04:** The README contains a short, prominent qualification-status box. One linked guarantees page owns the detailed topology, platform, payload-bound, and performance claim matrix.
- **D-05:** The supported product story contains `BlobStore` as the storage foundation and `UnifiedCache` as cache policy over that foundation. Phase 9 does not promote `SqlCache`; its separately approved removal belongs to Phase 10.

### Installation and release posture
- **D-06:** Until an immutable release is published, the primary installation path is a checked-out repository managed with `uv`. Local wheel build/install instructions are secondary.
- **D-07:** Describe the current state as a **local-ready development version**: the checked-out revision is qualified for local use, has not been published, and may change before the first supported release. Do not call it an alpha release candidate or imply a frozen v1.0 API.
- **D-08:** PostgreSQL and S3 appear only in guarantees/reference material and remain visibly `NOT_QUALIFIED`. They do not appear in quick starts, normal recommendations, or promoted examples.
- **D-09:** Documentation starts with the minimal installation and introduces capability-specific extras only in the task guide that needs them. Do not recommend an install-everything or vague broad bundle as the default.

### Executable examples
- **D-10:** Maintain four canonical example journeys: in-memory `BlobStore`; durable filesystem-plus-SQLite `BlobStore` with catalog metadata; `UnifiedCache` plus decorator use; and store-local MCAP-style custom format registration.
- **D-11:** Audit the existing example inventory aggressively. Delete obsolete, redundant, or unsupported scripts unless they demonstrate a distinct current workflow and can be made executable. Do not create an unsupported-example archive.
- **D-12:** CI executes the exact published example files unchanged. Do not maintain test-only copies of documentation snippets or a second equivalent example implementation.
- **D-13:** Canonical examples are self-verifying and disposable: they use isolated temporary/private storage, assert expected results, print a short success indication, and leave no files behind.

### Format-handler public surface
- **D-14:** Phase 9 includes one practical MCAP-style tutorial plus the minimum extension contract: store-local registration, stable data/payload identity, version declaration, safe suffixes, contained path I/O, and successful round trips. Reusable conformance tooling and broader long-term extension policy remain future developer-kit work.
- **D-15:** Rename the public `CacheHandler` protocol directly to `FormatHandler` throughout implementation, built-in subclasses, public imports, annotations, errors/messages, tests, documentation, and examples. Do not retain a compatibility alias. — **Reversibility:** costly — Undoing this cutover would touch the same public imports, subclass declarations, tests, and documentation again; the project is intentionally making the correction before a supported release.
- **D-16:** Preserve existing persisted `data_type`, `payload_format`, and payload-format-version identities. Values such as `array`, `object`, `pandas_dataframe`, `npz`, and `parquet` already describe the data/native format and do not encode the misleading Python protocol name. The rename must not create a migration or rebuild requirement.
- **D-17:** Keep the already accepted per-store registration shape, `store.handlers.register_handler(...)`; the rename clarifies the protocol without creating a new registry API.

### the agent's Discretion
- Choose exact task-guide filenames, navigation order, and concise prose while preserving the README/task/reference hierarchy above.
- Decide which obsolete documents and examples contain unique current material worth folding into a canonical guide before deletion.
- Choose the bounded test harness that executes exact canonical example files in isolated temporary environments.
- Choose internal transitional edit order for the alias-free `FormatHandler` cutover, provided no commit or completed plan leaves persisted handler identities changed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and audit findings
- `.planning/PROJECT.md` — BlobStore-first product identity, cache-over-store architecture, pre-production cutover permission, and explicit nonclaims.
- `.planning/ROADMAP.md` § Phase 9 — fixed goal, success criteria, and prohibition on lifecycle/concurrency expansion.
- `.planning/REQUIREMENTS.md` — partial `CACH-06` publication requirement and the current deferred qualification claims.
- `.planning/v1.0-v1.0-MILESTONE-AUDIT.md` — authoritative adoption gaps, broken documentation flows, stale evidence findings, and bounded closure recommendations.
- `docs/adr/0001-topology-specific-storage-guarantees.md` — mandatory topology-specific guarantee vocabulary and stop conditions against new coordination machinery.

### Completed architecture being documented
- `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md` — local-readiness meaning, installation matrix decisions, optional dependency policy, and preserved remote/performance nonclaims.
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-CONTEXT.md` — final obstore participant boundary, path-based handler seam, transport-evidence limits, and payload bounds.
- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — explicit offline migration/rebuild contract and handler-owned payload identities.
- `.codex/skills/spike-findings-cacheness/SKILL.md` — verified obstore/handler integration summary and non-negotiable authority separation.

### Current public and documentation surfaces to reconcile
- `README.md` — current primary onboarding path; presently teaches removed APIs and the old cache-first identity.
- `docs/BLOB_STORE.md` — current direct-storage guide containing obsolete construction and backend semantics.
- `docs/API_REFERENCE.md` — mixed current and removed public surfaces that must be replaced or narrowed.
- `docs/PLUGIN_DEVELOPMENT.md` — stale global handler-registration guidance to replace with the store-local `FormatHandler` tutorial.
- `docs/RELEASE_QUALIFICATION.md` — existing qualification vocabulary to consolidate into the detailed guarantees page.
- `src/cacheness/__init__.py` — canonical top-level imports and package identity.
- `src/cacheness/storage/__init__.py` — direct-storage exports, including the current `CacheHandler` export that becomes `FormatHandler`.
- `src/cacheness/interfaces.py` — canonical handler protocol and related type/error terminology.
- `src/cacheness/handlers.py` — built-in format handlers and per-store registry implementation.
- `pyproject.toml` — package description and capability-specific dependency extras.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/test_phase6_examples.py`: existing executable cache examples and patterns that can inform the canonical example harness.
- `tests/packaging/test_wheel_matrix.py`: current isolated package/import checks for the base wheel and optional extras.
- `tests/test_handler_registration.py` and `tests/test_guarded_handler_io.py`: store/handler registration, path containment, suffix, and guarded-I/O behavior to reuse for the `FormatHandler` cutover and tutorial.
- `src/cacheness/storage/blob_store.py`: current direct storage, catalog metadata, and store-local handler integration used by the direct-storage examples.
- `src/cacheness/core.py` and `src/cacheness/decorators.py`: current `UnifiedCache` and decorator surfaces used by the cache example.

### Established Patterns
- Public package barrels deliberately enumerate supported symbols through `__all__`; a protocol rename must update both top-level and storage barrels plus import-contract tests.
- Persisted manifests bind stable handler and payload-format identities. Python class names are not the persisted format contract and must remain separate.
- Exact documented examples should run against temporary stores and current public construction rather than test-only helpers or legacy global factories.
- Qualification claims are evidence-class-specific: local deterministic/package evidence does not imply live remote, Windows, controlled-Linux, or publication qualification.

### Integration Points
- Rewrite the README and task-oriented documentation against actual signatures and exports in `src/cacheness/__init__.py`, `src/cacheness/storage/__init__.py`, `src/cacheness/core.py`, and `src/cacheness/storage/blob_store.py`.
- Replace `CacheHandler` in `src/cacheness/interfaces.py`, `src/cacheness/handlers.py`, storage barrels, annotations, error terminology, tests, docs, and examples while leaving manifest identities unchanged.
- Reduce `examples/` to the four canonical executable journeys and attach their exact files to bounded CI/pytest execution.
- Align `pyproject.toml` description and install guidance with the BlobStore-first product identity and task-specific extras.
- Refresh Phase 3 and Phase 8 verification/validation metadata from already-existing evidence; do not modify storage production code to make old reports look current.

</code_context>

<specifics>
## Specific Ideas

- The README should let a new user understand the product and complete either a direct-storage or cache-policy round trip without navigating a large manual.
- The MCAP-style example is the representative third-party format because it demonstrates a native custom suffix and path-based serialization without teaching the handler about managed obstore locators.
- “Format handler” is the canonical term because the protocol owns data detection and native serialization format, not cache policy or blob lifecycle.
- Local-ready wording must be candid: useful from a checkout now, not an already published or remotely qualified release.

</specifics>

<deferred>
## Deferred Ideas

- Add a separate Phase 10 before milestone completion to remove `SqlCache` directly and without a compatibility alias. That phase owns removal of public exports, implementation, dedicated tests, examples, documentation, and dependency surface that becomes unused. Phase 9 should not spend effort promoting or comprehensively rewriting `SqlCache` documentation.
- A future handler developer-kit phase owns reusable conformance tooling, multiple third-party format examples, and a broader long-term handler compatibility policy beyond Phase 9's minimum tutorial contract.
- Real PostgreSQL/Amazon-S3 qualification and immutable publication remain `SEED-007`; controlled-Linux performance remains `SEED-006`; native Windows qualification remains Phase 999.1.

</deferred>

---

*Phase: 09-Adoption and Release Surface Closure*
*Context gathered: 2026-09-16*
