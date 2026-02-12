"""
Security Module
==============

Cryptographic signing for cache metadata entries to prevent tampering.
Uses HMAC-SHA256 for fast, secure signatures of critical metadata fields.

Features:
- HMAC-based signing for metadata integrity
- Configurable signed fields
- Automatic key generation and management
- Key rotation support

Usage:
    from cacheness.storage.security import CacheEntrySigner

    signer = CacheEntrySigner(key_file_path=Path("cache/.signing_key"))

    # Sign an entry
    signature = signer.sign_entry(entry_data)

    # Verify an entry
    is_valid = signer.verify_entry(entry_data, signature)
"""

# Re-export from parent security.py
from ..security import CacheEntrySigner

__all__ = [
    "CacheEntrySigner",
]
