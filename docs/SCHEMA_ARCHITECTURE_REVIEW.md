# Cache Entries Schema Architecture Review

**Issue:** CACHE-q25  
**Status:** Design review (no code changes)  
**Date:** 2026-02-20

## Executive Summary

The current `cache_entries` table is a single monolithic table with 16 columns mixing four distinct concerns: **blob identity**, **serialization details**, **cache lifecycle**, and **user metadata**. This review evaluates whether splitting those concerns improves the system, and assesses two key enhancements: **content-addressable blob deduplication** and **inline small-blob storage**.

**Recommendation:** Keep the single-table design for now. The complexity cost of splitting outweighs the benefits at current scale. Instead, pursue **targeted column improvements** and **inline blob storage** as incremental enhancements.

---

## 1. Current Schema

### 1.1 Column Inventory

| # | Column | Type (SQLite/PG) | Purpose | Concern |
|---|--------|-------------------|---------|---------|
| 1 | `cache_key` | VARCHAR(16) PK | Entry identity | **Identity** |
| 2 | `description` | VARCHAR(500) | User label | User metadata |
| 3 | `data_type` | VARCHAR(20) NOT NULL | Handler type (`"dataframe"`, `"array"`, `"bytes"`, `"object"`) | Serialization |
| 4 | `created_at` | DATETIME NOT NULL | Entry creation time | Lifecycle |
| 5 | `accessed_at` | DATETIME NOT NULL | Last access time (eviction key) | Lifecycle |
| 6 | `file_size` | INTEGER NOT NULL | Blob size in bytes | Blob identity |
| 7 | `file_hash` | VARCHAR(16) | XXH3_64 integrity hash | Blob identity |
| 8 | `entry_signature` | VARCHAR(100) | HMAC-SHA256 signature | Security |
| 9 | `s3_etag` | VARCHAR(100) | S3 ETag (remote backend) | Blob backend |
| 10 | `object_type` | VARCHAR(100) | Python type string | Serialization |
| 11 | `storage_format` | VARCHAR(20) | Handler format (`"parquet"`, `"blosc2"`, `"pickle"`) | Serialization |
| 12 | `serializer` | VARCHAR(20) | Serializer used (`"pickle"`, `"dill"`) | Serialization |
| 13 | `compression_codec` | VARCHAR(20) | Compression codec | Serialization |
| 14 | `actual_path` | VARCHAR(500) | Relative blob file path | Blob identity |
| 15 | `cache_key_params` | TEXT / JSONB | Serialized function kwargs | User metadata |
| 16 | `metadata_dict` | TEXT / JSONB | User-facing queryable metadata | User metadata |

### 1.2 Indexes (SQLite)

| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_list_entries` | `created_at DESC` | List entries in reverse chronological order |
| `idx_cleanup` | `created_at` | Eviction by age |
| `idx_size_mgmt` | `file_size, created_at` | Size-based eviction |
| `idx_data_type` | `data_type` | Filter by handler type |
| `idx_metadata_notnull` | `created_at DESC WHERE metadata_dict IS NOT NULL` | Partial index for metadata queries |

### 1.3 PostgreSQL Differences

- `cache_key_params` and `metadata_dict` use `JSONB` (vs TEXT in SQLite)
- GIN index on `metadata_dict` with `jsonb_path_ops` for JSON queries
- `PgCacheStats` has `total_entries`, `total_size_bytes`, `last_cleanup_at` (SQLite stats table lacks these)

### 1.4 JSON Backend Differences

- Flat dict structure: `{description, data_type, created_at, accessed_at, file_size, metadata: {...}}`
- No column-level access — entire file parsed on every operation
- `metadata` sub-dict holds what SQLite puts in dedicated columns (actual_path, file_hash, etc.)
- Only viable for <200 entries per existing guidance

---

## 2. Concern Analysis

### 2.1 Four Concerns in One Table

The 16 columns serve four distinct roles:

1. **Blob identity** (4 columns): `cache_key`, `file_size`, `file_hash`, `actual_path`
2. **Serialization details** (4 columns): `data_type`, `object_type`, `storage_format`, `serializer`, `compression_codec`
3. **Cache lifecycle** (2 columns): `created_at`, `accessed_at`
4. **User/external metadata** (4 columns): `description`, `cache_key_params`, `metadata_dict`, `s3_etag`
5. **Security** (1 column): `entry_signature`

### 2.2 Current Pain Points

1. **Sparse columns**: `s3_etag`, `entry_signature`, `cache_key_params`, `serializer` are NULL for most entries. On SQLite this wastes minimal space (NULL = 0 bytes in row), but adds cognitive overhead.

2. **Serialization columns are blob concerns, not cache concerns**: `storage_format`, `serializer`, `compression_codec` describe how data was written, not how the cache manages it. These could belong to a "blob metadata" table.

3. **`file_hash` exists but is not used as a dedup key**: The hash is computed and stored for integrity verification, but two entries with identical content create two separate blob files.

4. **No `access_count`**: Eviction is LRU-only (by `accessed_at`). LFU or hybrid policies need an access counter. The `CacheStats` table has global `cache_hits`/`cache_misses` but no per-entry counts.

5. **No inline blob storage**: Every entry, even a 10-byte payload, creates a separate file.

6. **No TTL column**: TTL is enforced at read time from config, not stored per-entry. This prevents mixed-TTL caches and makes it impossible to query "what's expired?" without the config.

---

## 3. Evaluation: Two-Table Design (Blobs + Cache)

### 3.1 Proposed Split

**Table 1 — Blobs (content-addressable):**
```sql
CREATE TABLE blobs (
    blob_hash     TEXT PRIMARY KEY,    -- XXH3_64 content hash
    file_size     INTEGER NOT NULL,
    storage_format TEXT,
    serializer    TEXT,
    compression_codec TEXT,
    actual_path   TEXT,
    inline_data   BLOB,               -- NULL if stored on disk
    is_inline     INTEGER DEFAULT 0,
    s3_etag       TEXT,
    created_at    DATETIME NOT NULL
);
```

**Table 2 — Cache index (lifecycle + policy):**
```sql
CREATE TABLE cache_index (
    cache_key     TEXT PRIMARY KEY,
    blob_hash     TEXT NOT NULL,       -- FK → blobs.blob_hash
    description   TEXT DEFAULT '',
    data_type     TEXT NOT NULL,
    object_type   TEXT,
    created_at    DATETIME NOT NULL,
    accessed_at   DATETIME NOT NULL,
    access_count  INTEGER DEFAULT 0,
    ttl_seconds   INTEGER,            -- per-entry TTL
    expires_at    DATETIME,           -- computed: created_at + ttl
    entry_signature TEXT,
    cache_key_params TEXT,
    metadata_dict TEXT,
    FOREIGN KEY(blob_hash) REFERENCES blobs(blob_hash)
);
```

### 3.2 Benefits

| Benefit | Impact | Confidence |
|---------|--------|------------|
| **Content dedup** | Identical data stored once. Saves disk for repeated puts of same data. | Medium — depends on actual usage patterns |
| **Cleaner separation** | Blob concerns vs cache concerns in different tables | High — conceptually cleaner |
| **Inline blob support** | `inline_data` column naturally belongs in blob table | High |
| **Future composability** | Blob table reusable across multiple cache indexes | Low — speculative |

### 3.3 Costs

| Cost | Impact | Severity |
|------|--------|----------|
| **JOIN on every read** | `get()` needs cache_index → blobs JOIN. ~0.1ms overhead per op on SQLite, more on PG. | Medium |
| **Cascading deletes** | Evicting an entry must check if other entries reference the same blob before deleting the file. Requires `SELECT COUNT(*) FROM cache_index WHERE blob_hash = ?` on every delete. | High |
| **Migration complexity** | 3 backends × namespace tables. Need to split existing rows into two tables, backfill blob_hash, handle FK constraints. | High |
| **Transaction complexity** | `put()` needs INSERT into both tables atomically. `cleanup_by_size()` needs to check blob refcount before deciding what to delete. | Medium |
| **Loss of INSERT OR REPLACE simplicity** | Current SQLite `put_entry` is a single `INSERT OR REPLACE`. Two-table design needs: check if blob exists, insert if not, upsert cache index. | Medium |
| **Code churn** | Every backend method touches both tables. `get_entry`, `put_entry`, `remove_entry`, `update_entry_metadata`, `list_entries`, `iter_entry_summaries`, `cleanup_by_size`, `clear_all` all need updates. 3 backends. | Very High |

### 3.4 Verdict: Not recommended at this time

**Cost/benefit ratio is poor.** The two-table design adds significant complexity (JOIN overhead, cascading deletes, migration across 3 backends, refcount logic) for benefits that are modest at current scale. Content dedup is valuable only when users frequently store identical data, which is not a reported use case.

**When to reconsider:** If Cacheness adds a "content-addressable mode" as a first-class feature, or if blob storage becomes a shared resource across multiple cache instances.

---

## 4. Evaluation: Content-Addressable Dedup (Single Table)

Even without a two-table split, we could add dedup within the current schema:

### 4.1 Approach

- Compute `file_hash` on write (already done)
- Before writing blob to disk, check if a blob with this hash already exists
- If so, reuse the `actual_path` (skip blob backend write)
- Track reference count or use existence checks on delete

### 4.2 Analysis

| Pro | Con |
|-----|-----|
| Saves disk for duplicate content | All the refcount complexity of two-table approach without the cleaner separation |
| Simple concept | Hash collisions (XXH3_64 = 64-bit) need handling at scale |
| No schema change needed | `file_hash` currently populated only when signing is enabled — would need to become always-on |
| — | Cleaning up blobs on delete requires scanning all entries for shared hash |

### 4.3 Verdict: Defer

Content-addressable dedup is a significant behavioral change (blob lifecycle becomes reference-counted) for a benefit that hasn't been requested. The hash is already stored for integrity — dedup could be layered on later with minimal schema impact.

---

## 5. Evaluation: Inline Small Blob Storage

**This is the highest-value improvement** identified in this review.

### 5.1 Current Problem

A 10-byte pickled integer creates:
- One file on disk (4KB minimum allocation on NTFS/ext4)
- One directory entry
- open() + write() + close() syscalls
- blob backend path resolution
- file_hash computation

For small payloads, this overhead is >99% of the work.

### 5.2 Proposed Approach

Add two columns to `cache_entries`:

```sql
ALTER TABLE cache_entries ADD COLUMN blob_data BLOB;
ALTER TABLE cache_entries ADD COLUMN is_inline INTEGER DEFAULT 0;
```

**Write path:**
1. Handler produces bytes (already happens via `handler.put()`)
2. Check `len(result_bytes) <= max_inline_size`
3. If below threshold: store in `blob_data` column, set `is_inline = 1`, skip blob backend write
4. If above threshold: current behavior (write file, store `actual_path`)

**Read path:**
1. Check `is_inline` flag on entry
2. If inline: pass `blob_data` bytes to `handler.get()` via a temp file or BytesIO adapter
3. If not inline: current behavior (read from `actual_path`)

### 5.3 Threshold Recommendations

| Backend | Recommended Threshold | Rationale |
|---------|----------------------|-----------|
| **SQLite** | 4,000 bytes | SQLite page size default is 4096. BLOBs ≤ page_size - overhead stay in the B-tree page. Larger BLOBs spill to overflow pages. Our custom `page_size=32768` allows up to ~32KB but that risks DB bloat. Conservative 4KB sweet spot. |
| **PostgreSQL** | 2,000 bytes | PG TOAST threshold is ~2KB. Values below this are stored inline in the heap tuple (fastest). Above this, PG auto-TOASTs (compresses + stores out-of-line). Let PG handle its own threshold. |
| **JSON** | 0 (disabled) | Base64 encoding adds 33% size. Entire file parsed on every op. Not worth it. |

### 5.4 Scale Considerations

| Entry Count | Avg Blob Size | Total Inline Data | SQLite DB Size Impact | Recommendation |
|-------------|---------------|--------------------|-----------------------|----------------|
| 1,000 | 1 KB | ~1 MB | Negligible (+1MB) | Inline ✅ |
| 10,000 | 1 KB | ~10 MB | Moderate (+10MB) | Inline ✅ |
| 100,000 | 1 KB | ~100 MB | Significant (+100MB) | Caution ⚠️ |
| 1,000,000 | 1 KB | ~1 GB | Problematic (+1GB) | Disable ❌ |

**At 100K+ inline blobs,** SQLite DB operations (VACUUM, backup, WAL checkpoints) may become noticeably slower. Consider:

- Automatic threshold reduction when entry count exceeds a configurable limit
- OR document that inline is best for <50K entries
- OR add a `max_inline_count` config that auto-disables inlining after N entries

### 5.5 Handler Interaction

**Challenge:** Handlers currently write files directly. The inline path needs to intercept the bytes *before* they hit the filesystem.

**Option A — Post-write ingest:** Let handler write to disk as normal, then if file ≤ threshold, read bytes back into `blob_data` and delete the file. Simple but wastes I/O.

**Option B — Handler produces bytes directly:** Modify handler interface to optionally return bytes instead of writing to disk. Major interface change.

**Option C — RAM-backed staging (recommended):** Handler writes to a temp path. Before `write_blob_from_path()`, check file size. If ≤ threshold, read into `blob_data` and delete temp file. If > threshold, proceed normally. Minimal interface change.

### 5.6 Verdict: Recommended (Phase 1 improvement)

Inline blob storage is the single highest-impact improvement for small-payload use cases. It requires:
- Schema migration (2 nullable columns)  
- Small write-path change (size check after handler.put)
- Small read-path change (check is_inline before blob read)
- Config additions (`max_inline_size`, disabled by default)

See **CACHE-gf0** for the dedicated implementation issue.

---

## 6. Recommended Incremental Improvements

Instead of a disruptive two-table split, pursue these targeted changes:

### Phase 1: Quick Wins (Low Risk)

| Change | Effort | Impact | Issue |
|--------|--------|--------|-------|
| Add `access_count INTEGER DEFAULT 0` column | Small | Enables LFU/hybrid eviction | New |
| Add `ttl_seconds INTEGER` column | Small | Per-entry TTL, "what's expired?" queries | New |
| Add `expires_at DATETIME` computed column | Small | Index for TTL-based cleanup | New |
| Document column semantics in code | Small | Reduces cognitive overhead | This issue |

### Phase 2: Inline Blob Storage (Medium Risk)

| Change | Effort | Impact | Issue |
|--------|--------|--------|-------|
| Add `blob_data BLOB` + `is_inline INTEGER` columns | Medium | Eliminates file overhead for small payloads | CACHE-gf0 |
| Config: `max_inline_size` (default 0 = disabled) | Small | User control | CACHE-gf0 |
| Schema migration for all 3 backends | Medium | Backward-compatible (nullable columns) | CACHE-gf0 |

### Phase 3: Future (Higher Risk, Deferred)

| Change | Effort | Impact | Issue |
|--------|--------|--------|-------|
| Content-addressable dedup | High | Saves disk for duplicate content | Defer |
| Two-table split | Very High | Cleaner separation, composability | Defer |
| Shared blob pool across cache instances | Very High | Multi-cache dedup | Defer |

---

## 7. Recommended Column Additions (Summary)

These columns would be added to `CacheEntryMixin` (SQLite), `PgCacheEntryMixin` (PG), and the JSON entry structure:

```python
# Phase 1
access_count = Column(Integer, default=0, nullable=False)
ttl_seconds = Column(Integer, nullable=True)        # NULL = use config default
expires_at = Column(DateTime(timezone=True), nullable=True)  # computed: created_at + ttl

# Phase 2 (CACHE-gf0)
blob_data = Column(LargeBinary, nullable=True)       # inline blob content
is_inline = Column(Integer, default=0, nullable=False)
```

### Migration Path

1. All new columns are nullable (except `access_count` with DEFAULT 0 and `is_inline` with DEFAULT 0)
2. `ALTER TABLE ... ADD COLUMN` works on SQLite and PG without rewrite
3. Schema version bump: v2 → v3
4. Existing entries: `access_count = 0`, `ttl_seconds = NULL`, `expires_at = NULL`, `blob_data = NULL`, `is_inline = 0`
5. Backward compatible: old code ignores new columns; new code handles NULL gracefully

### New Indexes

```sql
-- Phase 1: TTL-based cleanup
CREATE INDEX idx_expires_at ON cache_entries (expires_at) WHERE expires_at IS NOT NULL;

-- Phase 1: access_count for LFU eviction
CREATE INDEX idx_access_count ON cache_entries (access_count, accessed_at);
```

---

## 8. Open Questions

1. **`access_count` vs external stats:** Should per-entry access counting replace or supplement the global `CacheStats` table?

2. **`expires_at` computation:** Should it be computed on write (`created_at + ttl_seconds`) and stored, or computed on read? Stored is faster for queries, but stale if TTL config changes.

3. **Phase 1 priority:** Are `access_count` and `ttl_seconds` worth implementing before inline blobs, or should Phase 2 (CACHE-gf0) come first?

4. **JSON backend:** Should JSON backend get the new columns at all, or permanently skip them (already recommended for <200 entries only)?

5. **`data_type` + `object_type` redundancy:** `data_type` is the handler name (e.g., `"dataframe"`), `object_type` is the Python type (e.g., `"pandas.core.frame.DataFrame"`). Both are useful but could be consolidated.

---

## 9. Conclusion

The current monolithic `cache_entries` table is **adequate for current needs** despite mixing concerns. The two-table split (blobs + cache index) is architecturally cleaner but introduces disproportionate complexity for the current feature set.

**Recommended path forward:**
1. Keep single-table design
2. Add `access_count`, `ttl_seconds`, `expires_at` columns (Phase 1)
3. Add inline blob storage via CACHE-gf0 (Phase 2)
4. Revisit two-table split only if content-addressable storage becomes a first-class feature
