## Plan: Storage Mode via Hybrid Approach

**TL;DR:** Phase 1 adds a `storage_mode` config flag to `UnifiedCache` by fixing 4 bugs that currently prevent disabling TTL and eviction, then aliasing `hash_key` in the storage API. Phase 2 (deferred, documented as roadmap) refactors `UnifiedCache` to layer on a promoted `BlobStore`. All selected features (xxhash keys, integrity, signing, custom metadata, content-addressable) flow from the existing `UnifiedCache` implementation with minimal changes.

**Steps — Phase 1**

1. **Add `storage_mode` to config** — In `src/cacheness/config.py`, add `storage_mode: bool = False` to the top-level `CacheConfig` dataclass. When `True`, override `default_ttl_seconds` to `None` and `max_cache_size_mb` to `None` in `__post_init__`, and set `cleanup_on_init` to `False`.

2. **Fix `default_ttl_seconds=None` validation** — In `CacheMetadataConfig.__post_init__()` at config.py L96, guard the `<= 0` check with `if self.default_ttl_seconds is not None:`. Also update `validate_config()` if it has redundant validation paths.

3. **Fix `_cleanup_expired()` for `None` TTL** — In core.py ~L829, add an early return `if ttl_seconds is None: return` before calling `self.metadata_backend.cleanup_expired(ttl_seconds)`. This means "no TTL = no cleanup."

4. **Fix `_enforce_size_limit()` for `None` max size** — In core.py ~L1826, add an early return `if self.config.storage.max_cache_size_mb is None: return`. This disables eviction entirely.

5. **Disable stats in storage mode** — In `get()` at core.py, wrap `increment_hits()`/`increment_misses()` calls with `if not self.config.storage_mode:`. Alternatively, gate on the existing `enable_cache_stats` flag that currently isn't checked in `get()`.

6. **Disable auto-delete on errors in storage mode** — In the `except` blocks of `get()` at core.py ~L1110-L1125, add `if not self.config.storage_mode:` guard around the `self.delete(cache_key)` calls. Storage users should not lose data due to transient deserialization issues.

7. **Add `hash_key` alias** — Add a `hash_key` property/parameter alias in the public-facing methods:
   - `put()` — accept `hash_key` kwarg as alias for `cache_key` return value (documentation + return dict key)
   - `get()` — accept `hash_key` as alias for `cache_key` parameter
   - `get_metadata()`, `delete()`, `exists()` — same aliasing pattern
   - Implement via a small helper that normalizes `hash_key` → `cache_key` internally

8. **Content-addressable mode** — Port `BlobStore._compute_content_hash()` logic (blob_store.py L406-L414) into `UnifiedCache` as an alternative key generation strategy. Add `content_addressable: bool = False` to config. When enabled, `_create_cache_key()` should hash the data content (SHA-256) instead of the function parameters. This enables deduplication — storing the same data twice returns the same key.

9. **Update docs** — Update docs/BLOB_STORE.md comparison table to note that `UnifiedCache` now supports storage mode. Add a "Storage Mode" section to docs/CONFIGURATION.md explaining how to use `storage_mode=True` and what it disables.

10. **Tests** — Add test cases for:
    - `storage_mode=True` config initializes without errors
    - `None` TTL: entries never expire, no cleanup on init
    - `None` max size: entries are never evicted
    - `hash_key` alias works in `put()`/`get()`/`delete()`/`exists()`
    - Content-addressable: same data → same key, different data → different key
    - No auto-delete on deserialization errors in storage mode
    - Stats not tracked in storage mode

**Steps — Phase 2 (Roadmap, not implemented now)**

- Reconcile the two `MetadataBackend` ABCs into one
- Move xxhash key generation, integrity, signing, custom metadata into `BlobStore`
- Refactor `UnifiedCache` to compose `BlobStore` + cache concerns (TTL, eviction, stats)
- `storage_mode` becomes "use BlobStore directly without cache layer"
- Consider extracting storage as a separate package (`cacheness-store` or similar)

**Verification**

- `uv run pytest tests/ -x -q` — baseline must stay at 787 passed
- New tests for storage mode pass
- Manual test: create `UnifiedCache(storage_mode=True)`, store data, retrieve after restart — no TTL expiry, no eviction
- Quality check: `.\scripts\quality-check.ps1`

**Decisions**

- Chose Approach C (hybrid) over pure BlobStore promotion — lower risk, immediate value
- `hash_key` is an alias only — `cache_key` remains the internal canonical name
- Decorator support deferred — not selected for Phase 1 scope
- Content-addressable uses SHA-256 (matching BlobStore) not xxhash — content hashing benefits from cryptographic properties
