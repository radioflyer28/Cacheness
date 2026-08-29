# Coding Conventions

**Analysis Date:** 2026-08-29

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for library modules, for example `src/cacheness/error_handling.py` and `src/cacheness/file_hashing.py`.
- Name tests `test_<subject>.py` under `tests/`, such as `tests/test_config_validation.py` and `tests/test_s3_blob_backend.py`.
- Keep package API re-exports in `__init__.py` files, especially `src/cacheness/__init__.py` and `src/cacheness/storage/backends/__init__.py`.

**Functions:**
- Use lowercase `snake_case` for public and private functions and methods, with a leading underscore for implementation helpers such as `_normalize_function_args` in `src/cacheness/core.py` and `_hash_single_file` in `src/cacheness/file_hashing.py`.
- Use verbs for operations (`create_metadata_backend`, `validate_config`, `register_handler`) and `is_`/`has_`/`can_` predicates (`is_custom_metadata_available`, `can_handle`).
- Preserve `__dunder__` names for protocol methods and use `@property` for read-only identifiers such as handler `data_type`.

**Variables:**
- Use descriptive lowercase `snake_case` names (`cache_dir`, `metadata_backend`, `error_context`).
- Use uppercase names for module constants and capability flags (`SQLALCHEMY_AVAILABLE`, `PSYCOPG_AVAILABLE`, `BLOSC2_AVAILABLE`, `DEFAULT_TTL`-style sentinels).
- Prefix intentionally private module state with `_`, for example `_default_registry` in `src/cacheness/__init__.py` and `_metadata_backend_registry` in `src/cacheness/storage/backends/__init__.py`.

**Types:**
- Use PascalCase for classes and exception types (`UnifiedCache`, `CacheConfig`, `CacheStorageError`, `PostgresBackend`).
- Model grouped configuration as `@dataclass` classes in `src/cacheness/config.py`.
- Use abstract base classes and focused interfaces in `src/cacheness/interfaces.py` and `src/cacheness/storage/backends/base.py`; concrete implementations inherit the relevant interface.
- Type hints mix Python 3.11+ built-in generics (`list`, `dict`) with `typing.Optional`, `List`, `Dict`, `Tuple`, and `Union`. Match the surrounding module when extending it and type public boundaries where practical.

## Code Style

**Formatting:**
- Use four-space indentation and conventional PEP 8 spacing. The configured target line length is 88 in `pyproject.toml` (`[tool.ruff]`), although existing source and tests contain longer lines and trailing whitespace.
- Start modules with a descriptive module docstring; multi-line public APIs generally use Google-style `Args`, `Returns`, and `Raises` sections. Examples include `src/cacheness/core.py`, `src/cacheness/error_handling.py`, and `src/cacheness/interfaces.py`.
- Use section banner comments (`# =============================================================================`) in larger modules and test files to separate registries, fixtures, and behavior groups; follow the pattern in `src/cacheness/handlers.py` and `tests/test_blob_backend_registry.py`.
- Use f-strings for structured messages, paths, and log records. Preserve comments explaining optional dependency behavior, compatibility aliases, and platform-specific workarounds.

**Linting:**
- Run `uv run ruff check src tests`; Ruff is declared at `>=0.12.8` in the `dev` dependency group.
- `pyproject.toml` sets Ruff `target-version = "py312"` and ignores `B008` and `C901`. The intended lint groups are documented in comments (`E`, `W`, `F`, `I`, `B`, `C4`, `UP`), but the `lint.select` setting is commented out, so do not assume import sorting or all optional rule groups are enforced.
- The current source/test tree produces 123 findings under the active Ruff defaults, including unused imports, unused locals, late imports, and a small number of bare-except/lambda-style issues. New code should avoid adding to this baseline and should not use `# noqa` without a local reason.

## Import Organization

**Order:**
1. Standard-library imports (`logging`, `pathlib`, `typing`, `dataclasses`, and similar).
2. Third-party imports (`pytest`, `numpy`, `sqlalchemy`, `xxhash`, and optional libraries).
3. Local package imports (`from .config ...`, `from cacheness ...`).

The grouping is visible in `src/cacheness/core.py` and most tests, but it is not mechanically enforced and some files place `pytest` or `Path` in a different order. Keep imports grouped and remove unused imports when touching a module.

**Path Aliases:**
- No configured import path aliases were detected. Use package-relative imports inside `src/cacheness` and `cacheness.<module>` imports in tests, as shown in `tests/test_core.py` and `tests/test_error_handling.py`.
- Import optional dependencies lazily or behind `try/except ImportError` when the feature is optional. Examples include `src/cacheness/__init__.py`, `src/cacheness/handlers.py`, and `src/cacheness/storage/backends/s3_backend.py`.

## Error Handling

**Patterns:**
- Raise the domain-specific hierarchy from `src/cacheness/error_handling.py` (`CacheError` and its configuration, storage, serialization, handler, integrity, and metadata subclasses) for cross-cutting cache failures.
- Handler-specific failures use `CacheHandlerError` and its `CacheWriteError`, `CacheReadError`, `CacheFormatError`, and `CacheValidationError` subclasses in `src/cacheness/interfaces.py`.
- Preserve the original cause with `raise ... from e` when translating `OSError`, import, serialization, or backend failures. `with_error_handling` in `src/cacheness/error_handling.py` adds function/argument context and either reraises or returns a configured fallback.
- Use `pytest.raises` with a specific exception and, where stable, `match=` in tests; see `tests/test_directory_sharding.py`, `tests/test_handler_registration.py`, and `tests/test_error_handling.py`.
- Handle optional features explicitly: capability detection is represented by flags such as `SQLALCHEMY_AVAILABLE`, and unavailable optional paths should be skipped or produce a clear install-oriented error.

## Logging

**Framework:** Python standard-library `logging`, with `logger = logging.getLogger(__name__)` in library modules such as `src/cacheness/config.py`, `src/cacheness/core.py`, and `src/cacheness/metadata.py`.

**Patterns:**
- Use `debug` for configuration and operation details, `info` for backend selection/lifecycle and successful cache operations, `warning` for fallbacks or suppressed failures, and `error` for domain failures.
- Include operation context in messages or with `extra=...`; `src/cacheness/error_handling.py` demonstrates both structured context and duration logging.
- Tests that assert logs should use `caplog.at_level(...)` and inspect `caplog.text`, as in `tests/test_error_handling.py` and `tests/test_interfaces.py`.
- Preserve the existing user-facing log style, including backend/lifecycle messages and the existing emoji status prefixes in `src/cacheness/core.py`, when modifying adjacent operations.

## Comments

**When to Comment:**
- Add module/class/function docstrings for public APIs and explain non-obvious serialization, backend, concurrency, or compatibility decisions.
- Use short inline comments for algorithm steps and resource cleanup; larger phase/feature sections in `src/cacheness/config.py` and `tests/test_config_validation.py` use banner comments.
- Document why imports are lazy, why a fallback is selected, or why a test is skipped. Avoid comments that merely restate a simple line.

**JSDoc/TSDoc:**
- Not applicable. This is a Python project; docstrings are the API documentation mechanism.
- Google-style docstrings are common for library interfaces, with examples in `src/cacheness/interfaces.py` and `src/cacheness/error_handling.py`. Tests generally use one-line docstrings on classes and test methods.

## Function Design

**Size:**
- Keep new functions focused around one cache, serialization, backend, or validation responsibility. Existing large modules (`src/cacheness/sql_cache.py`, `src/cacheness/metadata.py`, `src/cacheness/handlers.py`, and `src/cacheness/core.py`) contain long orchestration methods, so extract helpers rather than growing those methods further.

**Parameters:**
- Type public parameters when stable; use `Optional[...]` for optional configuration and `**kwargs` for backend-specific options or cache-key parameters, matching `src/cacheness/core.py` and `src/cacheness/storage/backends/blob_backends.py`.
- Pass grouped behavior through configuration objects (`CacheConfig` and its sub-configurations in `src/cacheness/config.py`) instead of adding unrelated flags to every handler method.

**Return Values:**
- Return concrete values that callers can inspect: handlers return metadata dictionaries, backends return storage paths/bytes/bools, and validators return lists of errors or raise in strict mode.
- Use `None` for an absent optional object and explicit booleans for predicates. Keep metadata keys stable because tests and signing code inspect them directly (for example, `tests/test_handlers.py` and `tests/test_cache_integrity.py`).

## Module Design

**Exports:**
- Expose the supported convenience API through `__all__` in `src/cacheness/__init__.py`; optional exports are added only when their dependencies/imports are available.
- Keep registry functions and backend classes together with their registry implementation (`src/cacheness/storage/backends/__init__.py` and `src/cacheness/storage/backends/blob_backends.py`).

**Barrel Files:**
- Package `__init__.py` files act as deliberate barrels for public convenience imports: `src/cacheness/__init__.py` and `src/cacheness/storage/__init__.py` re-export core classes, handlers, metadata backends, and storage APIs.
- Prefer direct module imports for internal implementation dependencies to avoid expanding the public surface or creating circular imports; optional imports in `src/cacheness/core.py` are kept inside methods for this reason.

---

*Convention analysis: 2026-08-29*
