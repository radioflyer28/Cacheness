# Requirements: Cacheness v0.10.0

**Defined:** 2026-04-03
**Core Value:** Reliable, type-aware disk caching with pluggable backends

## v0.10.0 Requirements

Requirements for Security & Architecture milestone. Each maps to roadmap phases.

### Security

- [ ] **SEC-01**: Per-namespace key derivation via HKDF — derive namespace-specific signing keys from master key + namespace ID. Existing entries with shared key must still verify (migration path).
- [ ] **SEC-02**: Configurable key fallback behavior — replace silent in-memory fallback with user-controlled policy (`raise` / `warn` / `fallback`). Default: `warn` (preserves current behavior with explicit configuration).
- [ ] **SEC-03**: Blob content encryption at rest — AES-GCM encryption of cached blob data on disk. Disabled by default, signing remains enabled. Configuration exposed via `CacheMetadataConfig` with `encryption_enabled`, `encryption_key_file`, etc.
- [ ] **SEC-04**: Manual key rotation API — `rotate_key()` re-derives namespace keys and re-signs existing entries. Includes test scenarios: delete old key file, restart cache, verify old-key entries handled gracefully.

### Architecture

- [ ] **ARCH-01**: Further core.py decomposition — extract eviction, namespace management, and/or initialization logic into mixins or delegates. Target: ~2500 → ~1500 lines. All existing tests must pass unchanged.

### Testing

- [ ] **TEST-01**: Thread safety under concurrent access — concurrent `put()`/`get()` from multiple threads against `UnifiedCache`. Verify no data races, no corruption, no deadlocks across JSON and SQLite backends.
- [ ] **TEST-02**: Cross-platform atomic write verification — verify `shutil.move()` correctness on Windows, especially same-volume temp file scenarios. Verify no corruption under concurrent writes.

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Key Management

- **KEY-01**: Automated/scheduled key rotation — config-driven periodic key rotation with background task. Needs scheduler infrastructure that doesn't exist yet.

### Developer Experience

- **DX-01**: CLI tool for cache inspection — `cacheness inspect/list/stats/cleanup/verify` commands.
- **DX-02**: Advanced eviction policies — LRU, LFU, size-based, composite eviction.

### Async

- **ASYNC-01**: Async/await support — `AsyncUnifiedCache` with async backends (asyncpg, aioboto3, aiosqlite).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Async/await support | Large feature addition, separate milestone |
| Advanced eviction policies | Large feature addition, separate milestone |
| CLI tool | Separate DX milestone |
| TensorFlow handler fixes | Low priority, platform issues (Windows hangs) |
| JSON backend O(n²) writes | Documented limitation, "use SQLite" is the mitigation |
| Automated key rotation | Needs background task infrastructure, defer until demand |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEC-01 | Phase 16 | Pending |
| SEC-02 | Phase 15 | Pending |
| SEC-03 | Phase 18 | Pending |
| SEC-04 | Phase 17 | Pending |
| ARCH-01 | Phase 19 | Pending |
| TEST-01 | Phase 20 | Pending |
| TEST-02 | Phase 20 | Pending |

**Coverage:**
- v0.10.0 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-03*
*Last updated: 2026-04-03*
