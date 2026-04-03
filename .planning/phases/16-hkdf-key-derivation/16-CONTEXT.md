# Phase 16: HKDF Key Derivation - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Per-namespace cryptographic key isolation via HKDF-SHA256. Each namespace derives its own signing key from the master key + namespace ID, so compromising one derived key doesn't reveal the master or other namespaces' keys. This replaces the current shared-key model where all namespaces use the same 32-byte HMAC key. No new external dependencies — HKDF implemented using stdlib only.

</domain>

<decisions>
## Implementation Decisions

### HKDF Implementation
- **D-01:** Implement HKDF-SHA256 using only Python stdlib (`hmac` + `hashlib`), per RFC 5869. No `cryptography` dependency. The algorithm is ~15 lines (extract + expand steps).
- **D-02:** Derived keys are 32 bytes (same length as master key). HMAC-SHA256 natively produces 32 bytes, so single-block HKDF-Expand suffices.

### Migration & Backward Compatibility
- **D-03:** Use signature version bump: v2 entries = master key (pre-HKDF), v3 entries = derived key (HKDF). On verify, the signature version determines which key to use — no dual-verify needed.
- **D-04:** Provide an optional re-sign utility so users can migrate v2 entries to v3. This utility should be deferred to Phase 17 (Key Rotation API), which already handles re-signing.

### Opt-in/Opt-out Design
- **D-05:** Add `use_hkdf_derivation: bool = True` to `SecurityConfig`. Enabled by default — new caches get per-namespace key isolation out of the box.
- **D-06:** Disabling HKDF (`use_hkdf_derivation=False`) after v3 entries exist logs an info/warning. v3 entries still verify correctly (version in signature identifies the key), new entries revert to master key (v2).

### Per-namespace Key Scope
- **D-07:** Derived keys apply to both entry signatures AND namespace signatures. Full cryptographic isolation per namespace.
- **D-08:** Key derivation happens inside `CacheEntrySigner` — it receives `namespace_id` + `use_hkdf_derivation` flag, and derives internally. `create_cache_signer()` passes both through. This makes HKDF transparent to callers (UnifiedCache in cache mode, BlobStore in storage mode).

### Agent's Discretion
- HKDF `info` parameter content (e.g., `b"cacheness-ns-key-v1"` or similar)
- Whether to add a `derive_key(namespace_id)` method vs inline derivation
- Test file structure (extend existing or new `test_hkdf_derivation.py`)
- Exact log message wording for HKDF disable warning

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Security Architecture
- `src/cacheness/security.py` — `CacheEntrySigner.__init__()`, `sign_entry()`, `verify_entry()`, `sign_namespace()`, `verify_namespace()`, `_create_signature_payload()`, signature versioning (`CURRENT_SIGNATURE_VERSION = 2`)
- `src/cacheness/config.py` — `SecurityConfig` dataclass (~L389-438), `key_fallback_policy` field from Phase 15
- `src/cacheness/core.py` — `_init_entry_signer()` (~L329-360), `_sign_current_namespace()`, `_sign_entry_if_enabled()`
- `src/cacheness/storage/blob_store.py` — `_init_signer()`, entry signing in `put()`, verification in `get()`
- `src/cacheness/error_handling.py` — `CacheSecurityError`

### Phase 15 Context (predecessor)
- `.planning/phases/15-configurable-key-fallback/15-CONTEXT.md` — `key_fallback_policy` decisions, `create_cache_signer()` factory pattern

### Existing Tests
- `tests/test_namespace_signing.py` — namespace signing/verification patterns, cross-namespace isolation tests
- `tests/test_key_rotation.py` — key file deletion, re-generation, fallback behavior
- `tests/test_key_fallback_policy.py` — 3-mode policy tests from Phase 15
- `tests/test_cache_signing.py` — in-memory key tests

### Standard
- RFC 5869: HMAC-based Extract-and-Expand Key Derivation Function (HKDF)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CacheEntrySigner` — the class to extend with HKDF derivation logic
- `create_cache_signer()` factory — the entry point for both cache mode and storage mode
- `CURRENT_SIGNATURE_VERSION = 2` — increment to 3 for HKDF-derived signatures
- Signature parsing (`parse_versioned_signature()`) — already handles version extraction from `vN:hex` format
- `_create_signature_payload()` — creates HMAC payload from entry fields

### Established Patterns
- Signature format: `v{version}:{hex}` for entries, `ns1:{hex}` for namespaces
- Config via `@dataclass` fields on `SecurityConfig`
- `create_cache_signer()` factory passes all config to `CacheEntrySigner.__init__()`
- `# intentionally broad` annotation for catch clauses

### Integration Points
- `SecurityConfig` → new `use_hkdf_derivation: bool = True` field
- `CacheEntrySigner.__init__()` → accept `namespace_id` + `use_hkdf_derivation`, derive key if enabled
- `create_cache_signer()` → pass `namespace_id` + `use_hkdf_derivation` through
- `_init_entry_signer()` in core.py → pass `namespace_id` from `self.namespace`
- `BlobStore._init_signer()` → pass `namespace_id`
- `sign_entry()` / `verify_entry()` → version 3 uses derived key
- `sign_namespace()` / `verify_namespace()` → use derived key (namespace signature format may need version bump too: `ns1` → `ns2`)

</code_context>

<specifics>
## Specific Ideas

- The HKDF extract step uses the master key as IKM (input keying material) and an optional salt. The expand step uses namespace_id as the `info` parameter to produce the derived key.
- `CacheEntrySigner` stores both `self.master_key` (original 32 bytes) and `self.derived_key` (HKDF output). Signing uses `derived_key` when HKDF is enabled, `master_key` otherwise.
- For verification of old v2 signatures, the signer needs access to `master_key` regardless of HKDF setting.
- Enabled by default means existing users upgrading will see v3 signatures on new entries. Old v2 entries remain valid.

</specifics>

<deferred>
## Deferred Ideas

- **Re-sign utility** for migrating v2→v3 entries — belongs in Phase 17 (Key Rotation API)
- **Key derivation for encryption keys** — if Phase 18 (Encryption at Rest) also needs per-namespace keys, HKDF infrastructure from this phase can be reused

</deferred>

---

*Phase: 16-hkdf-key-derivation*
*Context gathered: 2026-04-03*
