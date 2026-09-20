# Phase 11: Clean Supplemental Documentation and Normalize Validation Evidence - Context

**Gathered:** 2026-09-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the bounded documentation and validation-evidence debt identified by the
v1.0 milestone audit, then refresh that audit with a current milestone verdict.
Consolidate or delete stale supplemental guides, normalize legacy phase
validation records to the current Nyquist schema, and replace Phase 3's obsolete
draft validation narrative with a compact canonical evidence record.

Phase 11 also directly removes the dormant, automatically disabled TensorFlow
handler and its complete package, configuration, test, CI, qualification, and
documentation surface. This is a pre-production product-surface cleanup, not a
new format or replacement feature.

This phase must not redesign storage lifecycle, concurrency, recovery, backend
composition, handler persistence, or topology guarantees. A genuine Phase 3
lifecycle regression is a stop condition requiring separately authorized work
under ADR 0001; it is not permission for another race-fix loop.

</domain>

<decisions>
## Implementation Decisions

### Supplemental documentation
- **D-01:** Apply selective consolidation: move still-useful, verified material into canonical task/reference guides, repair only genuinely distinct guidance, and delete redundant or misleading supplemental documents.
- **D-02:** Consolidate unique verified platform material into `docs/RELEASE_QUALIFICATION.md`, then delete `docs/CROSS_PLATFORM_GUIDE.md` and `docs/WINDOWS_COMPATIBILITY.md`. Native Windows remains explicitly `UNAVAILABLE` / `NOT_QUALIFIED`.
- **D-03:** Evaluate `docs/PANDAS_COMPATIBILITY.md` and `docs/CUSTOM_METADATA.md` under the same selective-consolidation rule. Their content survives only where it adds verified value beyond current canonical format/catalog guidance.
- **D-04:** Prefer links to the four canonical executable examples. Any retained additional runnable snippet must execute directly or have an exact import/signature contract test.

### TensorFlow removal
- **D-05:** Remove dormant native TensorFlow support completely: handler implementation and exports, lazy-import machinery, configuration flags, optional dependency groups and lock entries, dedicated tests, packaging profiles, CI/qualification jobs, documentation, and current feature claims. Do not retain aliases, tombstones, dormant code, or a re-enable recipe. — **Reversibility:** costly — restoring support would require deliberately reintroducing the dependency, format handler, public/configuration surface, qualification matrix, and persisted-format support.
- **D-06:** Delete TensorFlow-specific documentation, including both the user guide and implementation-status document, rather than consolidating it into supported format documentation.
- **D-07:** Mark `.planning/seeds/SEED-005-remove-native-tensorflow-support.md` fulfilled and retain it as historical rationale with a Phase 11 resolution link so it cannot be promoted again.
- **D-08:** TensorFlow removal is a bounded package/handler cutover. It does not authorize changes to `BlobStore`, `UnifiedCache`, lifecycle authorities, obstore participation, or other retained handlers.

### Validation normalization
- **D-09:** Normalize Phases 1, 5, 6, 7, 8, and 9 using their existing evidence plus one bounded current non-live regression gate. Run phase-specific checks only where evidence is missing, stale, or contradicted.
- **D-10:** Update each existing `*-VALIDATION.md` in place to the current canonical schema (`status: validated`, accurate Nyquist fields, green or explicitly superseded task rows, commands, scope, and nonclaims). Do not create wrapper reports or a competing milestone-level source of validation truth.
- **D-11:** When an old row names a removed test, command, or feature, preserve a traceable supersession mapping to the current replacement evidence or explicit removal decision. Do not silently rewrite history or leave obsolete pending rows controlling discovery.
- **D-12:** Scoped validation is valid: real PostgreSQL/Amazon-S3 evidence, controlled-Linux performance, native Windows, and immutable publication remain visibly `NOT_QUALIFIED`, `NOT_PUBLISHED`, or deferred without blocking validation of the approved local/deterministic phase scope.

### Phase 3 evidence boundary
- **D-13:** Replace `03-VALIDATION.md` with a compact canonical validation record covering the approved SQLite/local-filesystem and single-process-memory scopes, ADR guarantee classes, qualified commit/results, controlling evidence chain, current regression confirmation, explicit nonclaims, and stop conditions. Preserve obsolete detail through Git history and dated evidence rather than embedding it as a live appendix.
- **D-14:** Phase 3's controlling historical evidence remains `03-25-SUMMARY.md` and `docs/phase3-direct-implementation-2026-09-06.md`, corroborated by later Phase 07.1/8 evidence and one finite current contract regression. Do not attempt to reproduce every original environment, timing, or detached-checkout detail.
- **D-15:** The current Phase 3 gate covers named lifecycle integrity, deterministic recovery, cache-over-store composition, and the frozen non-live repository suite once. Documented typed contention outcomes are valid; timing noise or universal contender success is not a reason to add coordination.
- **D-16:** If that gate exposes a genuine integrity or recovery defect, classify it against ADR 0001 and stop Phase 11. This phase may correct obsolete evidence references, but it may not repair production lifecycle code or launch an automatic review/fix cycle.
- **D-17:** State the original provenance plainly: Phase 3 used user-approved direct primary-agent implementation and exact-commit qualification rather than an independent verifier. Phase 11 confirms continued behavior but must not retroactively relabel that history.

### Completion and audit closure
- **D-18:** Final acceptance is a layered bounded gate: focused TensorFlow-removal/package contracts, exact documentation/reference scans, executable-example ownership, canonical validation discovery, finite Phase 3 contracts, scoped Ruff, fresh-wheel membership/metadata/import/round-trip inspection, and one frozen non-live regression suite.
- **D-19:** Refresh `.planning/v1.0-v1.0-MILESTONE-AUDIT.md` after the gates pass. Mark supplemental-documentation and validation-format debt resolved, record current Nyquist and package evidence, preserve all explicit deferrals, and issue the current milestone-completion verdict.
- **D-20:** The completion record must not convert unavailable external evidence into a pass or reopen `BACK-05`, `QUAL-06`, native Windows, or immutable publication.

### the agent's Discretion
- Choose the exact canonical destination or deletion outcome for unique pandas and custom-metadata material after checking it against current code and task guides.
- Choose the smallest exact test selectors and supersession mappings that satisfy the bounded validation gate without recreating obsolete suites.
- Choose the internal edit order for TensorFlow removal, manifest/lock convergence, documentation cleanup, validation normalization, and final audit refresh.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and milestone findings
- `.planning/PROJECT.md` — current BlobStore-first product boundary, pre-production cutover permission, and explicit qualification limits.
- `.planning/ROADMAP.md` § Phase 11 — phase placement after Phase 10 and milestone-cleanup intent.
- `.planning/REQUIREMENTS.md` — authoritative requirement status, approved local scope, and deferred evidence classes.
- `.planning/v1.0-v1.0-MILESTONE-AUDIT.md` — exact supplemental-documentation, validation-status, integration, and Nyquist findings this phase closes.
- `docs/adr/0001-topology-specific-storage-guarantees.md` — mandatory single-authority guardrail, guarantee vocabulary, and stop conditions for any Phase 3 regression finding.

### Prior decisions and Phase 3 evidence
- `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md` — local-readiness boundary, retained nonclaims, package matrix, and bounded performance posture.
- `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md` — canonical documentation hierarchy, executable example ownership, and local-ready product story.
- `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md` — direct pre-production removal pattern and historical-document preservation rule.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-25-SUMMARY.md` — direct Phase 3 closure and exact qualified commit summary.
- `docs/phase3-direct-implementation-2026-09-06.md` — complete Phase 3 implementation/qualification ledger and provenance.

### TensorFlow removal surface
- `.planning/seeds/SEED-005-remove-native-tensorflow-support.md` — original removal rationale to close as fulfilled.
- `src/cacheness/handlers.py` — dormant `TensorFlowTensorHandler`, lazy import code, registry exclusions, and persisted identity references.
- `src/cacheness/config.py` — TensorFlow handler configuration flag.
- `src/cacheness/storage/handlers/__init__.py` — compatibility export surface to remove.
- `pyproject.toml` — TensorFlow optional dependency/profile declarations.
- `.github/workflows/quality.yml` — TensorFlow-specific qualification jobs.
- `tests/test_tensorflow_handler.py` and `tests/packaging/test_wheel_matrix.py` — dedicated handler and installed-profile contracts that must be removed or inverted.

### Documentation and validation targets
- `docs/RELEASE_QUALIFICATION.md` — sole detailed owner for platform evidence and nonclaims after consolidation.
- `docs/PANDAS_COMPATIBILITY.md`, `docs/CUSTOM_METADATA.md`, `docs/CROSS_PLATFORM_GUIDE.md`, `docs/WINDOWS_COMPATIBILITY.md`, `docs/TENSORFLOW_TENSOR_GUIDE.md`, and `docs/TENSORFLOW_HANDLER_STATUS.md` — supplemental documents to consolidate, replace, or delete under the decisions above.
- `.planning/phases/01-compatibility-and-security-baseline/01-VALIDATION.md`, `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VALIDATION.md`, `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-VALIDATION.md`, `.planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md`, `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md`, `.planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md`, and `.planning/phases/09-adoption-and-release-surface-closure/09-VALIDATION.md` — legacy/draft records to normalize in place.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/test_phase9_documentation.py`: explicit current-document inventory and reference checks suitable for enforcing consolidation/deletion outcomes.
- `tests/packaging/test_wheel_matrix.py`: fresh source-free wheel membership, dependency metadata, import, and local round-trip proof; extend/invert it for TensorFlow absence.
- `tests/qualification/` and `tools/verify_phase8_contracts.py`: current local qualification and evidence-schema contracts without pretending to supply live remote evidence.
- Current authority, reconciliation, integrity, cache composition, and non-live repository tests provide the finite Phase 3 confirmation boundary.
- `examples/memory_blob_store.py`, `examples/durable_catalog_store.py`, `examples/unified_cache.py`, and `examples/custom_mcap_format.py`: canonical executable documentation owners.

### Established Patterns
- Pre-production removals are direct: remove implementation, exports, dependencies, tests, documentation, and package claims together without aliases or tombstones.
- Current-facing documents are updated or deleted; dated audits, ledgers, summaries, and Git history preserve historical truth.
- Validation claims are scope-specific. `validated` local/deterministic evidence can coexist with explicit remote/platform/performance nonqualification.
- A documentation contract should scan an explicit current inventory and link to executable canonical examples instead of duplicating large unowned snippets.
- ADR 0001 separates integrity, recovery, progress, and performance; a typed contention outcome is not a lifecycle defect.

### Integration Points
- TensorFlow removal crosses `handlers.py`, `config.py`, storage compatibility exports, dependency groups/lockfile, CI workflows, qualification/packaging tooling, tests, and documentation.
- Supplemental-guide consolidation updates the docs index and any tests that enumerate current documents or qualification wording.
- Validation normalization must satisfy the active GSD/Nyquist discovery schema while retaining exact evidence provenance and deferral states.
- The final milestone audit refresh consumes the completed Phase 11 verification and updated validation discovery; it is last, after code/package/docs gates pass.

</code_context>

<specifics>
## Specific Ideas

- TensorFlow was found to be dormant rather than removed: its handler and dependency surface remained while automatic registration was commented out. Phase 11 finishes that cutover instead of documenting a disabled feature.
- `docs/RELEASE_QUALIFICATION.md` becomes the single current owner for Linux/macOS/Windows evidence boundaries; portable wheel tags must not be presented as native platform qualification.
- Phase 3's normalized record should be short enough that a future agent cannot mistake obsolete pending plans for current work authorization.
- The final audit should distinguish “debt resolved” from “deferred evidence completed”; BACK-05, QUAL-06, Windows, and publication stay deferred.

</specifics>

<deferred>
## Deferred Ideas

- Real PostgreSQL/Amazon-S3 qualification and immutable publication remain `SEED-007` / `BACK-05` work.
- Controlled-Linux performance qualification remains `SEED-006` / `QUAL-06` work.
- Native Windows lifecycle qualification remains Phase 999.1.
- Narwhals/dataframe expansion, XXH3 canonical-digest reconsideration, and broader format-handler developer tooling remain their existing future seeds/backlog work.

</deferred>

---

*Phase: 11-clean-supplemental-documentation-and-normalize-validation-ev*
*Context gathered: 2026-09-17*
