# Phase 24: Cross-Backend Test Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 24-cross-backend-test-parity
**Areas discussed:** SQLite compatibility, Parametrization strategy, Test scope filtering, File organization

---

## SQLite Compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Assume fixed, parametrize confidently | Phase 23 added dedicated columns and field handling. Update the outdated comment. | |
| Verify first, then parametrize | Write a quick smoke test (encrypted put/get on SQLite) before parametrizing all tests | ✓ |
| Keep SQLite skip-marked | Parametrize but mark SQLite encryption tests as xfail until separately verified | |

**User's choice:** Verify first, then parametrize
**Notes:** User wants a smoke test to confirm Phase 23 fix before committing to full parametrization.

---

## Parametrization Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Decorator-based parametrize | @pytest.mark.parametrize('backend', [...]) on each test function. Existing pattern in test_namespace_integration.py. | ✓ |
| Fixture-based params | Shared fixture with params=['json', 'sqlite', 'postgresql']. Existing pattern in test_custom_metadata_backends.py. | |
| Class-per-backend | Separate test classes per backend with own skip logic. | |

**User's choice:** Decorator-based parametrize
**Notes:** Most visible and simple approach, matches existing codebase conventions.

---

## Test Scope Filtering

| Option | Description | Selected |
|--------|-------------|----------|
| Integration tests only | Parametrize the 12 integration tests (BlobStore + UnifiedCache + KeyRotation). Leave 6 primitives as-is. | ✓ |
| All 18 tests | Parametrize everything including backend-agnostic primitives. | |
| Curated subset | Parametrize a representative subset rather than all 12. | |

**User's choice:** Integration tests only
**Notes:** 6 primitive tests (encrypt_blob/decrypt_blob) don't touch backends, no value in parametrizing them.

---

## File Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in test_encryption_at_rest.py | Add parametrize decorators to existing file. | |
| New cross-backend file | New test_encryption_cross_backend.py for parametrized versions. | |
| Merge into test_backend_parity.py | Parametrized tests alongside other parity tests. Groups cross-backend concerns. | ✓ |

**User's choice:** Merge into test_backend_parity.py
**Notes:** Groups cross-backend parity concerns in one place.

---

## Agent's Discretion

- Helper fixture design for creating encrypted caches with different backends
- PG cleanup approach in parametrized tests
- Whether to extract shared setup into conftest

## Deferred Ideas

None — discussion stayed within phase scope.
