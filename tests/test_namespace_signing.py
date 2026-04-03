"""
Tests for namespace registry signing (CACHE-0xc).

Validates that:
1. Namespace rows are signed on first init when signing is enabled.
2. Signatures survive reconnect and are verified.
3. Tampered signatures are detected (warning, non-fatal).
4. Signing is skipped when disabled.
5. sign_namespace / verify_namespace round-trip correctly.
6. set_namespace_signature works for all backends.
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig, SecurityConfig
from cacheness.security import CacheEntrySigner


# ---------------------------------------------------------------------------
# Low-level unit tests for CacheEntrySigner namespace methods
# ---------------------------------------------------------------------------


class TestSignerNamespaceMethods:
    """Unit tests for sign_namespace / verify_namespace on CacheEntrySigner."""

    @pytest.fixture
    def signer(self, tmp_path):
        return CacheEntrySigner(key_file_path=tmp_path / "key.bin")

    def test_sign_and_verify_roundtrip(self, signer):
        ns_data = {
            "namespace_id": "default",
            "display_name": "Default",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        sig = signer.sign_namespace(ns_data)
        assert sig.startswith("ns")  # ns1: or ns2: depending on HKDF
        assert ":" in sig
        assert len(sig) > 4  # "nsN:" + hex
        assert signer.verify_namespace(ns_data, sig) is True

    def test_verify_detects_tampering(self, signer):
        ns_data = {
            "namespace_id": "default",
            "display_name": "Default",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        sig = signer.sign_namespace(ns_data)

        # Tamper with display_name
        tampered = {**ns_data, "display_name": "Hacked"}
        assert signer.verify_namespace(tampered, sig) is False

    def test_verify_detects_tampered_namespace_id(self, signer):
        ns_data = {
            "namespace_id": "default",
            "display_name": "Default",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        sig = signer.sign_namespace(ns_data)

        tampered = {**ns_data, "namespace_id": "evil"}
        assert signer.verify_namespace(tampered, sig) is False

    def test_verify_rejects_none_signature(self, signer):
        ns_data = {"namespace_id": "x", "display_name": "", "created_at": None}
        assert signer.verify_namespace(ns_data, None) is False

    def test_verify_rejects_empty_signature(self, signer):
        ns_data = {"namespace_id": "x", "display_name": "", "created_at": None}
        assert signer.verify_namespace(ns_data, "") is False

    def test_verify_rejects_garbage_format(self, signer):
        ns_data = {"namespace_id": "x", "display_name": "", "created_at": None}
        assert signer.verify_namespace(ns_data, "not-a-valid-sig") is False

    def test_different_keys_produce_different_sigs(self, tmp_path):
        signer_a = CacheEntrySigner(key_file_path=tmp_path / "key_a.bin")
        signer_b = CacheEntrySigner(key_file_path=tmp_path / "key_b.bin")

        ns_data = {
            "namespace_id": "test",
            "display_name": "Test",
            "created_at": datetime(2026, 6, 15, tzinfo=timezone.utc),
        }
        sig_a = signer_a.sign_namespace(ns_data)
        sig_b = signer_b.sign_namespace(ns_data)

        assert sig_a != sig_b
        assert signer_a.verify_namespace(ns_data, sig_a) is True
        assert signer_b.verify_namespace(ns_data, sig_b) is True
        # Cross-key verification must fail
        assert signer_a.verify_namespace(ns_data, sig_b) is False

    def test_schema_version_excluded_from_signature(self, signer):
        """schema_version changes during migrations — must NOT invalidate sig."""
        ns_data = {
            "namespace_id": "default",
            "display_name": "Default",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        sig = signer.sign_namespace(ns_data)

        # Adding schema_version should not affect the signature
        ns_data_with_version = {**ns_data, "schema_version": 99}
        assert signer.verify_namespace(ns_data_with_version, sig) is True


# ---------------------------------------------------------------------------
# Unit tests for key file permissions
# ---------------------------------------------------------------------------


class TestKeyFilePermissions:
    """Verify that _set_key_file_permissions applies OS-appropriate protections."""

    def test_key_file_permissions_applied_on_generate(self, tmp_path):
        """Key file gets restrictive permissions when generated."""
        import sys

        key_path = tmp_path / "key.bin"
        signer = CacheEntrySigner(key_file_path=key_path)

        assert key_path.exists()
        if sys.platform == "win32":
            # On Windows, verify icacls removed inherited permissions
            # and only the current user has access
            import subprocess

            result = subprocess.run(
                ["icacls", str(key_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            output = result.stdout
            # After ACL change: no inherited "(I)" entries,
            # only the current user should appear
            assert "(I)" not in output, "Inherited permissions should be removed"
            assert os.getlogin() in output, "Current user should have access"
        else:
            import stat

            mode = key_path.stat().st_mode
            assert mode & stat.S_IRWXG == 0, "Group should have no access"
            assert mode & stat.S_IRWXO == 0, "Others should have no access"

    def test_set_key_file_permissions_static_method(self, tmp_path):
        """_set_key_file_permissions works as a standalone static method."""
        import sys

        key_path = tmp_path / "test_key.bin"
        key_path.write_bytes(b"x" * 32)

        CacheEntrySigner._set_key_file_permissions(key_path)

        if sys.platform != "win32":
            import stat

            mode = key_path.stat().st_mode
            assert mode & stat.S_IRWXG == 0
            assert mode & stat.S_IRWXO == 0


# ---------------------------------------------------------------------------
# Integration tests: namespace signed on cache init
# ---------------------------------------------------------------------------


class TestNamespaceSigningIntegration:
    """Integration tests: cache init signs the namespace row."""

    @pytest.fixture
    def cache_dir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        if d.exists():
            shutil.rmtree(d)

    def _make_cache(self, cache_dir, *, signing=True, backend="sqlite"):
        config = CacheConfig(
            cache_dir=str(cache_dir / "cache"),
            metadata_backend=backend,
            security=SecurityConfig(enable_entry_signing=signing),
        )
        return UnifiedCache(config)

    def test_sqlite_namespace_signed_on_init(self, cache_dir):
        """First init should sign the default namespace."""
        cache = self._make_cache(cache_dir, backend="sqlite")
        ns = cache.metadata_backend.get_namespace("default")
        assert ns is not None
        assert ns.signature is not None
        assert ns.signature.startswith(("ns1:", "ns2:"))
        cache.close()

    def test_json_namespace_signed_on_init(self, cache_dir):
        """First init should sign the default namespace (JSON backend)."""
        cache = self._make_cache(cache_dir, backend="json")
        ns = cache.metadata_backend.get_namespace("default")
        assert ns is not None
        assert ns.signature is not None
        assert ns.signature.startswith(("ns1:", "ns2:"))
        cache.close()

    def test_signature_survives_reconnect(self, cache_dir):
        """Re-opening the cache should verify (not re-sign) the namespace."""
        cache1 = self._make_cache(cache_dir, backend="sqlite")
        ns1 = cache1.metadata_backend.get_namespace("default")
        sig1 = ns1.signature
        cache1.close()

        # Re-open — same key → same signature should be verified
        cache2 = self._make_cache(cache_dir, backend="sqlite")
        ns2 = cache2.metadata_backend.get_namespace("default")
        assert ns2.signature == sig1  # not re-signed
        cache2.close()

    def test_signing_disabled_leaves_null(self, cache_dir):
        """When signing is disabled, signature should remain NULL."""
        cache = self._make_cache(cache_dir, signing=False)
        ns = cache.metadata_backend.get_namespace("default")
        assert ns.signature is None
        cache.close()

    def test_custom_namespace_signed(self, cache_dir):
        """create_namespace via core should also get a signed namespace
        when the user reconnects with that namespace."""
        # Create the namespace via a cache on default
        cache = self._make_cache(cache_dir, backend="sqlite")
        cache.metadata_backend.create_namespace("team_a", display_name="Team A")
        cache.close()

        # Open a cache on team_a — namespace should get signed
        config = CacheConfig(
            cache_dir=str(cache_dir / "cache"),
            metadata_backend="sqlite",
            namespace="team_a",
            security=SecurityConfig(enable_entry_signing=True),
        )
        cache2 = UnifiedCache(config)
        ns = cache2.metadata_backend.get_namespace("team_a")
        assert ns is not None
        assert ns.signature is not None
        assert ns.signature.startswith(("ns1:", "ns2:"))
        cache2.close()


# ---------------------------------------------------------------------------
# Backend-level tests: set_namespace_signature
# ---------------------------------------------------------------------------


class TestSetNamespaceSignature:
    """Direct tests that set_namespace_signature persists the value."""

    def test_sqlite_set_and_read(self, tmp_path):
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="sqlite",
        )
        cache = UnifiedCache(config)
        backend = cache.metadata_backend

        backend.set_namespace_signature("default", "ns1:deadbeef")
        ns = backend.get_namespace("default")
        assert ns.signature == "ns1:deadbeef"
        cache.close()

    def test_json_set_and_read(self, tmp_path):
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
        )
        cache = UnifiedCache(config)
        backend = cache.metadata_backend

        # The init already signs it; overwrite to test the method
        backend.set_namespace_signature("default", "ns1:cafebabe")
        ns = backend.get_namespace("default")
        assert ns.signature == "ns1:cafebabe"
        cache.close()
