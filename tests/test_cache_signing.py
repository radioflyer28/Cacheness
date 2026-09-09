"""Signing regressions through explicit cache policy composition."""

from __future__ import annotations

import logging

import pytest

from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheConfig, CacheStorageConfig, SecurityConfig
from cacheness.core import UnifiedCache
from cacheness.storage.composition import BackendRef, StoreTopology


def _sqlite_filesystem_topology(root):
    """Build the persistent local profile required by signing regressions."""

    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root / "payloads"}),
        authority=BackendRef(name="sqlite", options={"root": root / "authority"}),
    )


def _signed_cache(root, *, delete_invalid_signatures=True, use_in_memory_key=False):
    """Create one initialized, explicitly owned cache for a signing test."""

    cache = UnifiedCache(
        CacheConfig(
            storage=CacheStorageConfig(cache_dir=root / "cache"),
            security=SecurityConfig(
                enable_entry_signing=True,
                delete_invalid_signatures=delete_invalid_signatures,
                allow_unsigned_entries=True,
                use_in_memory_key=use_in_memory_key,
            ),
        ),
        store=_sqlite_filesystem_topology(root),
    )
    cache.initialize()
    return cache


@pytest.fixture
def temp_cache_signing_enabled(tmp_path):
    """Create an explicitly composed cache that preserves corrupt evidence."""

    cache = _signed_cache(tmp_path / "delete-enabled")
    try:
        yield cache
    finally:
        cache.close()


@pytest.fixture
def temp_cache_delete_disabled(tmp_path):
    """Create an explicitly composed cache that retains invalid evidence."""

    cache = _signed_cache(
        tmp_path / "delete-disabled", delete_invalid_signatures=False
    )
    try:
        yield cache
    finally:
        cache.close()


class TestDeleteInvalidSignatures:
    """Test the supported nested signing configuration and typed results."""

    def test_config_option_defaults_and_flat_alias_rejection(self):
        """Security options live only on SecurityConfig after the cutover."""

        assert CacheConfig().security.delete_invalid_signatures
        assert CacheConfig(
            security=SecurityConfig(delete_invalid_signatures=True)
        ).security.delete_invalid_signatures
        assert not CacheConfig(
            security=SecurityConfig(delete_invalid_signatures=False)
        ).security.delete_invalid_signatures

        with pytest.raises(TypeError):
            CacheConfig(delete_invalid_signatures=False)

    def test_basic_signing_functionality(
        self, temp_cache_signing_enabled, temp_cache_delete_disabled
    ):
        """Signed entries are retrieved through receipt keys and lookup outcomes."""

        first = {"message": "Hello, World!", "numbers": [1, 2, 3, 4, 5]}
        first_receipt = temp_cache_signing_enabled.put(first)
        first_lookup = temp_cache_signing_enabled.lookup(
            cache_key=first_receipt.receipt.key
        )

        assert first_lookup.outcome is CacheOutcome.HIT
        assert first_lookup.value == first

        second = {"message": "Second test", "value": 42}
        second_receipt = temp_cache_delete_disabled.put(second)
        second_lookup = temp_cache_delete_disabled.lookup(
            cache_key=second_receipt.receipt.key
        )

        assert second_lookup.outcome is CacheOutcome.HIT
        assert second_lookup.value == second

    def test_config_logging(self, caplog):
        """The nested security configuration remains inspectable to diagnostics."""

        with caplog.at_level(logging.DEBUG):
            config = CacheConfig(
                security=SecurityConfig(delete_invalid_signatures=False)
            )

        assert not config.security.delete_invalid_signatures

    def test_in_memory_key_configuration_and_flat_alias_rejection(self):
        """In-memory signing-key policy also belongs exclusively to SecurityConfig."""

        assert not CacheConfig().security.use_in_memory_key
        assert CacheConfig(
            security=SecurityConfig(use_in_memory_key=True)
        ).security.use_in_memory_key
        assert not CacheConfig(
            security=SecurityConfig(use_in_memory_key=False)
        ).security.use_in_memory_key

        with pytest.raises(TypeError):
            CacheConfig(use_in_memory_key=True)

    def test_in_memory_key_instances_preserve_typed_signed_reads(self, tmp_path):
        """An in-memory signing key supports a live cache without implicit ownership."""

        cache = _signed_cache(tmp_path / "memory-key", use_in_memory_key=True)
        try:
            value = {"test": "data", "value": 42}
            receipt = cache.put(value)

            result = cache.lookup(cache_key=receipt.receipt.key)

            assert result.outcome is CacheOutcome.HIT
            assert result.value == value
        finally:
            cache.close()
