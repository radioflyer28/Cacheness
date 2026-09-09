"""Explicit function-cache policy facades.

Decorators consume one caller-owned :class:`UnifiedCache`; they never create,
discover, close, or otherwise coordinate storage lifecycle resources.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from .cache_policy import (
    CacheLookupResult,
    CacheOutcome,
    CachePutResult,
    CacheRemovalReport,
)
from .core import UnifiedCache


FunctionT = TypeVar("FunctionT", bound=Callable[..., Any])
_DEFAULT_RECOMPUTE_OUTCOMES = frozenset({CacheOutcome.ABSENT, CacheOutcome.EXPIRED})
_NON_HIT_OUTCOMES = frozenset(CacheOutcome) - {CacheOutcome.HIT}


def _validate_recompute_policy(
    recompute_on: frozenset[CacheOutcome],
) -> frozenset[CacheOutcome]:
    """Validate the immutable outcome set before a function is wrapped."""

    if not isinstance(recompute_on, frozenset):
        raise TypeError("recompute_on must be a frozenset of CacheOutcome values")
    if not all(isinstance(outcome, CacheOutcome) for outcome in recompute_on):
        raise TypeError("recompute_on must contain only CacheOutcome values")
    if not recompute_on <= _NON_HIT_OUTCOMES:
        raise ValueError("recompute_on cannot include CacheOutcome.HIT")
    return recompute_on


def _raise_lookup_cause(result: CacheLookupResult) -> None:
    """Preserve a non-recomputed typed lookup failure for the caller."""

    if result.cause is not None:
        raise result.cause
    raise RuntimeError(
        f"Cache lookup outcome {result.outcome.value!r} cannot be recomputed "
        "without an explicit policy"
    )


def cached(
    *,
    cache: UnifiedCache,
    recompute_on: frozenset[CacheOutcome] = _DEFAULT_RECOMPUTE_OUTCOMES,
) -> Callable[[FunctionT], FunctionT]:
    """Cache one function through a caller-owned :class:`UnifiedCache`.

    ``ABSENT`` and ``EXPIRED`` calls recompute by default. Other typed lookup
    outcomes preserve their original cause unless the caller explicitly includes
    them in ``recompute_on``. The latest observed lookup remains available as
    ``cache_last_lookup`` on the wrapped function even when that explicit policy
    permits recomputation.
    """

    if not isinstance(cache, UnifiedCache):
        raise TypeError("cache must be an explicit UnifiedCache instance")
    policy = _validate_recompute_policy(recompute_on)

    def decorate(func: FunctionT) -> FunctionT:
        if not callable(func):
            raise TypeError("cached can wrap only callable functions")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            namespace, cache_key = cache._function_cache_key(func, args, kwargs)
            result = cache.lookup_call(func, args, kwargs, cache_key=cache_key)
            wrapper.cache_last_lookup = result
            if result.outcome is CacheOutcome.HIT:
                return result.value
            if result.outcome not in policy:
                _raise_lookup_cause(result)

            value = func(*args, **kwargs)
            wrapper.cache_last_put_result = cache.put_call(
                func,
                args,
                kwargs,
                value,
                namespace=namespace,
                cache_key=cache_key,
            )
            return value

        def cache_clear(**kwargs: Any) -> CacheRemovalReport:
            """Return one truthful, bounded removal report for this function."""

            return cache.invalidate_function(func, **kwargs)

        wrapper.cache_clear = cache_clear
        wrapper.cache_last_lookup = cast(CacheLookupResult | None, None)
        wrapper.cache_last_put_result = cast(CachePutResult | None, None)
        return cast(FunctionT, wrapper)

    return decorate


__all__ = ["cached"]
