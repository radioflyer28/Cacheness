<!-- refreshed: 2026-09-17 -->
# Coding Conventions

**Analysis date:** 2026-09-17

## Naming and module design

- Use lowercase `snake_case.py` modules, `snake_case` functions/variables, and
  PascalCase classes/exceptions. Private implementation state starts with `_`.
- Name tests `test_<subject>.py` under `tests/`; use descriptive test names that
  state the observable contract.
- Public exports are deliberate package-barrel decisions. Keep canonical
  implementation imports internal and avoid expanding compatibility paths.
- Model grouped configuration with dataclasses and use focused abstract
  contracts for handlers, lifecycle authorities, and payload participants.

## Python style and imports

- Use four-space indentation, PEP 8 spacing, focused functions, and module or
  public-API docstrings where context is not obvious.
- Group imports as standard library, third party, then local package imports.
  Guard optional imports at the feature boundary.
- Ruff targets Python 3.11 with an 88-character line length. The repository has
  historical baseline findings, so new/edited Python must pass scoped
  `ruff check` without adding suppressions lacking a local reason.
- Prefer f-strings and clear structured logging context. Avoid broad exception
  catches at new boundaries.

## Errors, logging, and security

- Raise the domain hierarchy in `error_handling.py` or handler-specific errors
  in `interfaces.py`. Translate a narrow operational exception with
  `raise ... from exc` when the public boundary needs a domain error.
- Use `debug` for detail, `info` for successful lifecycle events, `warning` for
  explicit fallbacks/partial outcomes, and `error` for domain failures.
- Treat application payload bytes as trusted, but treat persisted metadata,
  paths, descriptors, and transport evidence as untrusted inputs. Preserve
  containment, safe parsing, signing, and fail-closed integrity checks.

## Storage and handler conventions

- `BlobStore` is the only lifecycle coordinator. A handler writes/reads a
  private staged or snapshot artifact; it does not pick visible generations or
  create managed paths.
- Register a custom format through `store.handlers.register_handler(...)` and
  keep its serialization/deserialization contract independent of cache policy.
- `UnifiedCache` layers keys, TTL, outcomes, invalidation, and maintenance over
  one store. Do not add a second metadata authority to policy code.
- Read ADR 0001 before modifying lifecycle, topology, concurrency, recovery,
  or timeouts. A new lock, queue, startup protocol, projection gate, or
  cross-resource transaction proposal is an architectural stop condition.

## Tests and packaging

- Prefer real temporary local stores and small fake collaborators. Mock only
  unavailable optional services, time, permissions, or SDK boundaries.
- Restore global registry state in tests and close stores/backends constructed
  by fixtures. Keep live service tests behind explicit markers.
- For package/dependency changes, run `uv lock --check` and the existing
  isolated-wheel probe. Artifact evidence must inspect the built wheel and its
  installed metadata, not only the source checkout.
- Preserve dated audit/planning records when current maps are refreshed. Update
  current claims surgically rather than erasing historical evidence.

---

*Current conventions map refreshed for the post-cut product boundary on 2026-09-17.*
