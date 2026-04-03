# Phase 16: HKDF Key Derivation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 16-hkdf-key-derivation
**Areas discussed:** HKDF implementation strategy, Migration & backward compat, Opt-in/opt-out design, Per-namespace key scope

---

## HKDF Implementation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Stdlib HMAC-based HKDF | Implement HKDF-SHA256 using only hmac+hashlib (RFC 5869 is ~15 lines). Zero new deps. | ✓ |
| Add cryptography dependency | Use cryptography.hazmat.primitives.kdf.HKDF. Battle-tested but adds ~3MB compiled dependency. | |
| Simple HMAC derivation | derive = HMAC(master_key, namespace_id). Not formally HKDF but sufficient. Simplest. | |

**User's choice:** Stdlib HMAC-based HKDF
**Notes:** No new dependencies. RFC 5869 implementation is straightforward.

### Derived Key Length

| Option | Description | Selected |
|--------|-------------|----------|
| 32 bytes (same as master) | HKDF-SHA256 naturally produces 32 bytes. | ✓ |
| 64 bytes | Could use longer for extra margin, but HMAC-SHA256 only uses 32 bytes anyway. | |

**User's choice:** 32 bytes (same as master)

---

## Migration & Backward Compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-verify fallback | On verify failure with derived key, retry with master key. No re-signing needed. | |
| Eager migration scan | On enable, scan all entries and re-sign with derived keys. Clean but requires full scan + write. | |
| Signature version bump | v2 = master key, v3 = derived key. Version in signature determines which key to use. | ✓ |

**User's choice:** Signature version bump
**Notes:** Version bump provides clean separation — v2 entries verified with master key, v3 with derived key.

### Re-signing Old Entries

| Option | Description | Selected |
|--------|-------------|----------|
| Leave old entries as v2 | Old v2 entries remain valid forever. No automated re-signing. | |
| Optional re-sign utility | Provide method to re-sign v2→v3a (defer to Phase 17 rotation API). | ✓ |

**User's choice:** Optional re-sign utility (deferred to Phase 17)

---

## Opt-in/Opt-out Design

| Option | Description | Selected |
|--------|-------------|----------|
| Disabled by default (opt-in) | Existing users see no change. Must explicitly enable. Safest migration. | |
| Enabled by default (opt-out) | All new caches get HKDF. Existing caches keep working (v2 sigs verified with master key). | ✓ |

**User's choice:** Enabled by default (opt-out)

### Config Field

| Option | Description | Selected |
|--------|-------------|----------|
| use_hkdf_derivation: bool = True | Boolean on SecurityConfig. True = derive per-namespace keys. | ✓ |
| key_derivation: str = 'hkdf' | Enum field. Extensible for future schemes. | |

**User's choice:** use_hkdf_derivation: bool = True

### Toggle Mid-use

| Option | Description | Selected |
|--------|-------------|----------|
| Graceful | v2/v3 still verify, new entries use master key. | |
| Warn on disable | Log info/warning that HKDF was disabled. Otherwise same as graceful. | ✓ |

**User's choice:** Warn on disable

---

## Per-namespace Key Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both entry + namespace signing | Derived keys for entry and namespace signatures. Full cryptographic isolation. | ✓ |
| Entry signing only | Only entry signatures use derived keys. Namespace registry uses master key. | |

**User's choice:** Both entry + namespace signing

### Derivation Location

| Option | Description | Selected |
|--------|-------------|----------|
| Derive inside CacheEntrySigner | CacheEntrySigner derives internally. Transparent to callers (both cache and storage mode). | ✓ |
| Callers derive and pass key | UnifiedCache/BlobStore derive the key, pass it to CacheEntrySigner. More explicit but duplicated. | |

**User's choice:** Derive inside CacheEntrySigner
**Notes:** User asked about cache mode vs storage mode — both create signers via create_cache_signer(), so derivation inside the signer ensures both modes get HKDF automatically.

---

## Agent's Discretion

- HKDF `info` parameter content
- derive_key method structure
- Test file organization
- Log message wording

## Deferred Ideas

- Re-sign utility for v2→v3 migration — Phase 17
- Key derivation for encryption keys — Phase 18 can reuse HKDF infrastructure
