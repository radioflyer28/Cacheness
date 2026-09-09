"""Cache deterministic transport results with an explicit policy facade.

This deliberately does not contact an HTTP service. A caller injects the
transport and cache, then owns closing the cache after the demonstration.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

from cacheness import (
    CacheConfig,
    CacheLookupResult,
    CacheOutcome,
    CachePolicyConfig,
    StoreTopology,
    UnifiedCache,
    cached,
)
from cacheness.config import CacheStorageConfig
from cacheness.error_handling import CacheBlobBackendError
from cacheness.storage import BackendRef


class DeterministicTransport:
    """A supplied transport whose optional response is intentionally ``None``."""

    def __init__(self) -> None:
        self.calls = 0

    def fetch_optional_profile(self, user_id: str) -> None:
        """Return one known absent profile without opening a socket."""

        self.calls += 1
        assert user_id == "ada"
        return None


def memory_topology() -> StoreTopology:
    """Build the explicitly supported same-process topology."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Show misses, cached ``None``, failure policy, and bounded clearing."""

    with TemporaryDirectory(prefix="cacheness-api-example-") as directory:
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=directory),
            policy=CachePolicyConfig(
                max_authoritative_bytes=1_024,
                catalog_page_size=8,
                maintenance_work_cap=8,
            ),
        )
        cache = UnifiedCache(config, store=memory_topology())
        transport = DeterministicTransport()
        try:
            cache.initialize()

            @cached(cache=cache)
            def optional_profile(user_id: str) -> None:
                return transport.fetch_optional_profile(user_id)

            assert optional_profile("ada") is None
            first_lookup = optional_profile.cache_last_lookup
            assert first_lookup is not None
            assert first_lookup.outcome is CacheOutcome.ABSENT
            assert optional_profile("ada") is None
            assert optional_profile.cache_last_lookup.outcome is CacheOutcome.HIT
            assert transport.calls == 1
            print(f"DEFAULT_RECOMPUTE_OUTCOME={first_lookup.outcome.value}")
            print("DEFAULT_RECOMPUTE_SET=absent,expired")
            print(
                "STORED_NONE_HIT="
                f"outcome:{optional_profile.cache_last_lookup.outcome.value},"
                f"transport_calls:{transport.calls}"
            )

            # This local injection proves default policy preserves typed failures.
            backend_error = CacheBlobBackendError("demonstration authority outage")
            forced_lookup = CacheLookupResult(
                CacheOutcome.BACKEND_ERROR,
                cause=backend_error,
            )
            original_lookup_call = cache.lookup_call
            cache.lookup_call = lambda *_args, **_kwargs: forced_lookup  # type: ignore[method-assign]
            default_failure_calls = 0
            try:
                @cached(cache=cache)
                def default_failure() -> str:
                    nonlocal default_failure_calls
                    default_failure_calls += 1
                    return "must not run"

                try:
                    default_failure()
                except CacheBlobBackendError as error:
                    assert error is backend_error
                else:  # pragma: no cover - documents the required policy.
                    raise AssertionError("default failure policy unexpectedly recomputed")
                assert default_failure_calls == 0
                print("DEFAULT_FAILURE_OUTCOME=backend_error")
                print("DEFAULT_FAILURE_RECOMPUTED=False")

                @cached(
                    cache=cache,
                    recompute_on=frozenset({CacheOutcome.BACKEND_ERROR}),
                )
                def opted_in_failure() -> str:
                    return "caller-chosen-fallback"

                assert opted_in_failure() == "caller-chosen-fallback"
                assert opted_in_failure.cache_last_lookup is forced_lookup
                assert opted_in_failure.cache_last_lookup.cause is backend_error
                print(
                    "OPT_IN_FAILURE="
                    f"outcome:{opted_in_failure.cache_last_lookup.outcome.value},"
                    f"cause:{type(backend_error).__name__}"
                )
            finally:
                cache.lookup_call = original_lookup_call  # type: ignore[method-assign]

            report = optional_profile.cache_clear()
            print(
                "FUNCTION_CACHE_CLEAR="
                f"attempted:{report.attempted},removed:{report.removed},"
                f"complete:{report.complete},retryable:{report.retryable}"
            )
            print(f"STATISTICS_HITS={cache.statistics().hit}")
        finally:
            cache.close()

    print("CANONICAL_FUNCTION_CACHE_EXAMPLE_OK")


if __name__ == "__main__":
    main()
