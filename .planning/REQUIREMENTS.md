# Requirements: v0.11.0 Cross-Backend Hardening

**Milestone:** v0.11.0
**Created:** 2026-04-06
**Status:** Active

## Requirements

### Encryption Backend Support

- [ ] **ENC-01**: Encryption metadata (`encryption_algorithm`, `encryption_iv`) preserved across SQLite and PostgreSQL backends via dedicated schema columns
- [ ] **ENC-02**: Encrypted blob roundtrip (put→get) works correctly with all 3 metadata backends (JSON, SQLite, PostgreSQL)
- [x] **ENC-03**: Encryption tests parametrized across all backends with full parity

### Inline Blob Encryption

- [ ] **INLINE-01**: Direct inline path (`_try_direct_inline`) encrypts data before storing in metadata
- [ ] **INLINE-02**: Inline read path (`_read_inline_blob`) decrypts ciphertext before passing to handler
- [ ] **INLINE-03**: Key rotation handles inline entries (decrypt with old key, re-encrypt with new key)

### Hardening

- [ ] **HARD-01**: Config validation fails loudly for known-bad configuration combinations at init time
- [ ] **HARD-02**: Schema migration tested on existing databases containing data

## Future Requirements

(None deferred from this milestone)

## Out of Scope

- Per-entry encryption keys — complexity without benefit over per-namespace HKDF
- Transparent re-encryption on backend switch — fragile and slow
- Encrypting metadata field values — blob content encryption is sufficient
- Multiple encryption algorithm support — AES-256-GCM is the standard
- Automated key rotation — needs background task infrastructure

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENC-01 | Phase 23 | Not started |
| ENC-02 | Phase 23 | Not started |
| ENC-03 | Phase 24 | Complete |
| INLINE-01 | Phase 25 | Not started |
| INLINE-02 | Phase 25 | Not started |
| INLINE-03 | Phase 25 | Not started |
| HARD-01 | Phase 26 | Not started |
| HARD-02 | Phase 26 | Not started |
