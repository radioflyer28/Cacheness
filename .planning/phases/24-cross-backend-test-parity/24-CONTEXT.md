# Phase 24: Cross-Backend Test Parity - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Ensure encryption test coverage is equal across all metadata backends (JSON, SQLite, PostgreSQL). Parametrize existing integration tests so they run against all 3 backends. No encryption test should remain hardcoded to a single backend. Remove the outdated "encryption+SQLite incompatibility" comment.

</domain>

<decisions>
## Implementation Decisions

### SQLite Compatibility
- **D-01:** The "encryption+SQLite known incompatibility" referenced in `test_concurrent_security.py` (L63) and elsewhere was the **exact bug Phase 23 fixed** (metadata fields being discarded due to missing schema columns). It is now resolved.
- **D-02:** Before parametrizing all tests, write a **quick smoke test** (encrypted put/get on SQLite) to verify Phase 23's fix works end-to-end. Then parametrize confidently.
- **D-03:** Update the outdated comment in `test_concurrent_security.py` to reflect that the incompatibility is resolved.

### Parametrization Strategy
- **D-04:** Use **decorator-based `@pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])`**. This matches the existing pattern in `test_namespace_integration.py` and is the most visible/simple approach.
- **D-05:** PostgreSQL parametrized tests must use `@pytest.mark.xdist_group("docker")` and skip when `CACHENESS_TEST_POSTGRES_URL` is not set, following the established pattern in `test_pg_schema_versioning.py`.

### Test Scope
- **D-06:** Only parametrize the **12 integration tests** (TestBlobStoreEncryption: 5, TestUnifiedCacheEncryption: 5, TestEncryptionKeyRotation: 2). The 6 pure encryption primitive tests in TestEncryptionModule are backend-agnostic and stay as-is.
- **D-07:** Each parametrized test must create a cache with the appropriate backend configuration and handle PG skip logic.

### File Organization
- **D-08:** Parametrized cross-backend encryption tests go into **`test_backend_parity.py`**, grouping them with other cross-backend parity concerns. The original `test_encryption_at_rest.py` tests remain for JSON-only baseline.

### Agent's Discretion
- Helper fixture design for creating encrypted caches with different backends
- Whether to extract shared setup into a conftest fixture or keep it file-local
- How to handle PG cleanup in parametrized tests (transaction rollback vs explicit cleanup)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Encryption Tests
- `tests/test_encryption_at_rest.py` — All 18 encryption tests (6 primitive + 12 integration), currently JSON-only
- `tests/test_concurrent_security.py` L55-80 — `encrypted_cache` fixture with outdated incompatibility comment

### Cross-Backend Test Patterns
- `tests/test_backend_parity.py` — Target file for parametrized encryption tests
- `tests/test_namespace_integration.py` L380, L410 — Decorator-based `@pytest.mark.parametrize("backend", ...)` pattern
- `tests/test_custom_metadata_backends.py` L73 — Fixture-based params pattern (NOT using this, but reference)

### PG Skip/Docker Pattern
- `tests/test_pg_schema_versioning.py` L37-39 — `requires_postgres` skip marker + `@pytest.mark.xdist_group("docker")`
- `tests/conftest.py` — `postgres_available()` session-scoped fixture

### Phase 23 Context (Prerequisite)
- `.planning/phases/23-encryption-schema-storage/23-CONTEXT.md` — Schema decisions, migration approach
- `src/cacheness/metadata/sqlite_backend.py` — SQLite encryption field handling (Phase 23 fix)
- `src/cacheness/storage/backends/postgresql_backend.py` — PG encryption field handling (Phase 23 fix)

### Encryption Infrastructure
- `src/cacheness/encryption.py` — encrypt_blob(), decrypt_blob(), derive_encryption_key()
- `src/cacheness/storage/blob_store.py` — _write_blob encryption (L490-520), get decryption (L615-628)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`_make_encrypted_cache()` helper** in `test_encryption_at_rest.py`: Creates JSON-backed encrypted cache. Needs to be generalized to accept backend parameter.
- **`_make_encrypted_blobstore()` helper**: Creates encrypted BlobStore with JSON metadata. Same generalization needed.
- **`postgres_available` fixture** in conftest.py: Session-scoped PG availability check — reuse for skip logic.
- **`requires_postgres` marker** in test_pg_schema_versioning.py: Skip decorator — reuse or import pattern.

### Established Patterns
- **Decorator parametrize**: `@pytest.mark.parametrize("backend", ["json", "sqlite"])` on test functions, backend selection inside test body.
- **PG conditional skip**: `if backend == "postgresql" and not postgres_available: pytest.skip("...")`
- **xdist grouping**: `@pytest.mark.xdist_group("docker")` on classes/tests that need PG.

### Integration Points
- `test_backend_parity.py` — destination for parametrized tests
- `test_encryption_at_rest.py` — source of test logic to parametrize
- `test_concurrent_security.py` L63 — outdated comment to update

</code_context>

<specifics>
## Specific Ideas

- The smoke test (D-02) should be a simple put/get with encryption enabled on SQLite, verifying that `encryption_algorithm` and `encryption_iv` survive the roundtrip. If it fails, Phase 23 has a bug.
- The parametrized tests should create their own tmp_path-based caches, not share state between backends.
- For PG tests, use `@pytest.mark.xdist_group("docker")` at the class level if grouping multiple PG-related parametrized tests.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 24-cross-backend-test-parity*
*Context gathered: 2026-04-06*
