# Requirements: Cacheness v0.12.0 Reliability Remediation

**Defined:** 2026-06-12
**Core Value:** Improve reliability, security, and maintainability of Cacheness without changing its public API semantics.

## v0.12.0 Requirements

### Wave 1 Reliability

- [ ] **REL-01**: User can call `clear_all()` or `clear_all_namespaces()` and have both metadata and namespace blob files removed while reserved metadata/security files remain intact.
- [ ] **REL-02**: User can rely on write-intent cleanup resolving relative blob paths against the cache directory, never the process working directory.
- [ ] **REL-03**: User can restart after a crash window between metadata commit and intent cleanup without valid committed blobs being deleted.
- [ ] **REL-04**: User can use storage mode with stale write intents and get conservative cleanup that preserves committed durable entries and removes only uncommitted orphans.
- [ ] **REL-05**: User can use the JSON backend and receive an error when data-critical metadata saves fail instead of getting a false success.
- [ ] **REL-06**: User can open a cache with corrupt JSON metadata and have the corrupt file preserved as a timestamped backup before the backend starts fresh.

### Cache Key Stability

- [ ] **KEY-01**: User can generate persistent cache keys for large tuples and other fallback paths that stay stable across processes and different `PYTHONHASHSEED` values.
- [ ] **KEY-02**: User can rely on property-based regression tests that stress cache-key determinism, collision resistance, control-parameter stripping, argument order normalization, and cross-process stability.

### TTL and Eviction

- [x] **TTL-01**: User can set per-entry TTL and have stored `expires_at` honored by reads and cleanup across JSON, SQLite, and PostgreSQL semantics.
- [ ] **TTL-02**: User can rely on init-time expired-entry cleanup deleting both metadata and blob files through the same path as public cleanup.
- [ ] **TTL-03**: User can overwrite or update metadata without unexpectedly resetting access counters, provenance timestamps, expiry semantics, or signatures.
- [ ] **TTL-04**: User can rely on size/eviction cleanup deleting remote blobs through the configured blob backend instead of leaking S3 or memory-backed objects.

### Multi-Process and Backend Parity

- [ ] **PAR-01**: User can write blobs from concurrent or repeated writers without deterministic temp-file collisions corrupting the final blob.
- [ ] **PAR-02**: User can store and filter custom user metadata with SQLite and PostgreSQL backends the same way JSON already supports it.
- [ ] **PAR-03**: User can retry or fail a same-key overwrite without losing the previous committed value in cache mode or storage mode.
- [ ] **PAR-04**: User can run integrity checks and namespace cleanup over all backend-visible blobs, including custom handler extensions and inline fallback files.

### Security and Storage Mode

- [ ] **SEC-01**: User can configure a minimum accepted signature version and reject downgraded signatures while receiving clear documentation for unsigned-entry risks.
- [ ] **SEC-02**: User can read encrypted blobs through the blob backend abstraction without avoidable plaintext temp files on disk.
- [ ] **SEC-03**: User can rotate keys with a two-phase process that leaves the original key and entries usable if rotation is interrupted.
- [ ] **SEC-04**: User can rely on UnifiedCache and BlobStore signing the same canonical field set, with a documented compatibility path for existing signed entries.
- [ ] **STRG-01**: User can enable storage mode and be protected from accidental cache-eviction APIs through an explicit raise-or-warning policy.
- [ ] **STRG-02**: User can understand and configure the storage-mode durability contract, including whether fsync is performed for JSON saves, blob writes, and intent files.
- [ ] **STRG-03**: User can rely on write intents being recorded before blob writes where needed so crash recovery covers the full uncommitted-blob window.

### Small Fixes and Release Polish

- [ ] **POL-01**: User metadata dictionaries passed to `BlobStore.put()` remain unchanged by Cacheness.
- [ ] **POL-02**: User blob keys that require sanitization cannot silently collide with distinct original keys.
- [ ] **POL-03**: User hot-path `get()` calls avoid redundant metadata reads when checking expiry.
- [ ] **POL-04**: User-visible package version metadata matches the shipped changelog/version line.
- [ ] **POL-05**: User-provided absolute blob IDs are rejected before path construction.
- [ ] **POL-06**: User SQLite connections apply PRAGMA behavior in the correct lifecycle location.
- [ ] **POL-07**: User S3 namespace deletion reports per-object failures instead of hiding them.
- [ ] **POL-08**: User can import `UnifiedCache` from the package root without a recurring foot-gun.

## Future Requirements

### Deferred Features

- **FUT-01**: User can compose local and remote caches through a tiered pull-through cache after v0.12.0 reliability prerequisites land.
- **FUT-02**: User can opt into JSON backend write batching/debouncing after save-failure semantics are hardened.
- **FUT-03**: User can move to a breaking-change default where signing keys imply signed entries are required.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Tiered pull-through cache implementation | Pending todo explicitly lists v0.12.0 reliability/parity work as prerequisites. |
| Flipping `allow_unsigned_entries` default | Breaking change; keep as a future major-version seed. |
| JSON backend write batching/debouncing | Performance policy decision; do after TASK-3 so batching cannot reintroduce silent persistence loss. |
| Async/await cache API | Larger feature unrelated to the remediation waves. |
| New eviction policies | Larger feature; current milestone fixes correctness of existing cleanup/eviction paths. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REL-01 | Phase 28 | Pending |
| REL-02 | Phase 28 | Pending |
| REL-03 | Phase 28 | Pending |
| REL-04 | Phase 28 | Pending |
| REL-05 | Phase 28 | Pending |
| REL-06 | Phase 28 | Pending |
| KEY-01 | Phase 28 | Pending |
| KEY-02 | Phase 28 | Pending |
| TTL-01 | Phase 29 | Complete |
| TTL-02 | Phase 29 | Pending |
| TTL-03 | Phase 29 | Pending |
| TTL-04 | Phase 29 | Pending |
| PAR-01 | Phase 30 | Pending |
| PAR-02 | Phase 30 | Pending |
| PAR-03 | Phase 30 | Pending |
| PAR-04 | Phase 30 | Pending |
| SEC-01 | Phase 31 | Pending |
| SEC-02 | Phase 31 | Pending |
| SEC-03 | Phase 31 | Pending |
| SEC-04 | Phase 31 | Pending |
| STRG-01 | Phase 31 | Pending |
| STRG-02 | Phase 31 | Pending |
| STRG-03 | Phase 31 | Pending |
| POL-01 | Phase 32 | Pending |
| POL-02 | Phase 32 | Pending |
| POL-03 | Phase 32 | Pending |
| POL-04 | Phase 32 | Pending |
| POL-05 | Phase 32 | Pending |
| POL-06 | Phase 32 | Pending |
| POL-07 | Phase 32 | Pending |
| POL-08 | Phase 32 | Pending |

**Coverage:**
- v0.12.0 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0

---
*Requirements defined: 2026-06-12*
*Last updated: 2026-06-12 after v0.12.0 milestone initialization*
