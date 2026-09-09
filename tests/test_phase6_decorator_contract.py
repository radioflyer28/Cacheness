"""Phase 6 contracts for explicit UnifiedCache function caching."""

from __future__ import annotations

import ast
import gc
import inspect

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the supported same-process topology used by decorator contracts."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache(tmp_path) -> UnifiedCache:
    """Create and initialize an explicitly owned cache policy instance."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    return cache


def test_none_result_is_a_presence_hit_through_an_explicit_cache(tmp_path) -> None:
    """A stored ``None`` is a hit and never triggers another function call."""

    cache = _cache(tmp_path)
    calls = 0
    try:
        @cached(cache=cache)
        def returns_none(value: str) -> None:
            nonlocal calls
            calls += 1
            return None

        assert returns_none("value") is None
        assert returns_none("value") is None
        assert calls == 1
        assert cache.statistics().absent == 1
        assert cache.statistics().hit == 1
    finally:
        cache.close()


def test_explicit_cache_keeps_qualified_functions_isolated(tmp_path) -> None:
    """Same bound arguments cannot cross-hit between qualified functions."""

    cache = _cache(tmp_path)
    first_calls = 0
    second_calls = 0
    try:
        @cached(cache=cache)
        def first(value: int) -> str:
            nonlocal first_calls
            first_calls += 1
            return f"first:{value}"

        @cached(cache=cache)
        def second(value: int) -> str:
            nonlocal second_calls
            second_calls += 1
            return f"second:{value}"

        assert first(7) == "first:7"
        assert second(7) == "second:7"
        assert first(7) == "first:7"
        assert second(7) == "second:7"
        assert first_calls == second_calls == 1
        assert cache.function_namespace(first.__wrapped__) != cache.function_namespace(
            second.__wrapped__
        )
    finally:
        cache.close()


def test_explicit_cache_normalizes_equivalent_function_call_forms(tmp_path) -> None:
    """Bound defaults and argument spellings produce one policy key."""

    cache = _cache(tmp_path)
    calls = 0
    try:
        @cached(cache=cache)
        def render(value: int, scale: int = 2, suffix: str = "!") -> str:
            nonlocal calls
            calls += 1
            return f"{value * scale}{suffix}"

        assert render(4) == "8!"
        assert render(4, 2, "!") == "8!"
        assert render(value=4, scale=2, suffix="!") == "8!"
        assert render(4, suffix="!", scale=2) == "8!"
        assert calls == 1
        assert render(4, 3) == "12!"
        assert calls == 2
    finally:
        cache.close()


def test_explicit_decorator_module_has_no_implicit_lifecycle_owner(tmp_path) -> None:
    """Decorator construction only closes over a caller-supplied cache instance."""

    source = inspect.getsource(__import__("cacheness.decorators", fromlist=["cached"]))
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "atexit" not in imported_modules
    assert "weakref" not in imported_modules
    assert "UnifiedCache" not in called_names

    cache = _cache(tmp_path)
    close_calls = 0
    original_close = cache.close

    def observed_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    cache.close = observed_close  # type: ignore[method-assign]
    try:
        @cached(cache=cache)
        def value() -> int:
            return 1

        del value
        gc.collect()
        assert close_calls == 0
    finally:
        cache.close()
