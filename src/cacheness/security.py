"""
Cache Entry Security and Signing
===============================

This module provides cryptographic signing for cache metadata entries to prevent tampering.
Uses HMAC-SHA256 for fast, secure signatures of critical metadata fields.

Features:
- HMAC-based signing for metadata integrity
- Configurable security levels (minimal, enhanced, paranoid)
- Automatic key generation and management
- Backward compatibility with unsigned entries
- Key rotation support

Security Model:
- Signs only immutable fields to prevent signature invalidation
- Uses deterministic field ordering for consistent signatures
- Stores signature alongside entry metadata
- Verifies signatures on cache retrieval
"""

import hmac
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheEntrySigner:
    """
    HMAC-based cache entry signer for metadata integrity protection.
    
    Provides cryptographic signatures for cache entry metadata to detect tampering
    with the SQLite database or JSON metadata files.
    """
    
    # Default enhanced fields (previously "enhanced" security level)
    DEFAULT_SIGNED_FIELDS = ["cache_key", "file_hash", "data_type", "file_size", "created_at", "prefix"]
    
    def __init__(self, key_file_path: Path, custom_fields: Optional[List[str]] = None, 
                 use_in_memory_key: bool = False):
        """
        Initialize the cache entry signer.
        
        Args:
            key_file_path: Path to the signing key file (ignored if use_in_memory_key=True)
            custom_fields: Custom list of fields to sign (if None, uses default enhanced fields)
            use_in_memory_key: If True, use in-memory key instead of persistent file
        """
        self.key_file_path = key_file_path
        self.use_in_memory_key = use_in_memory_key
        self.signed_fields = custom_fields or self.DEFAULT_SIGNED_FIELDS
        self.custom_fields = custom_fields
        self.use_in_memory_key = use_in_memory_key
        self.secret_key = self._load_or_generate_key()
        
        # Determine which fields to sign
        self.signed_fields = custom_fields or self.DEFAULT_SIGNED_FIELDS
        
        key_type = "in-memory" if use_in_memory_key else "persistent"
        logger.debug(f"Cache signer initialized: fields={self.signed_fields}, key_type={key_type}")
    
    def _load_or_generate_key(self) -> bytes:
        """Load existing signing key or generate a new one."""
        # If using in-memory key, always generate new one
        if self.use_in_memory_key:
            logger.info("Using in-memory signing key (not persistent)")
            return secrets.token_bytes(32)
        
        try:
            if self.key_file_path.exists():
                # Load existing key
                key = self.key_file_path.read_bytes()
                if len(key) != 32:
                    logger.warning(f"Invalid key length ({len(key)} bytes), generating new key")
                    return self._generate_new_key()
                logger.debug(f"Loaded signing key from {self.key_file_path}")
                return key
            else:
                return self._generate_new_key()
        except Exception as e:
            logger.warning(f"Failed to load signing key: {e}, generating new key")
            return self._generate_new_key()
    
    def _generate_new_key(self) -> bytes:
        """Generate a new 32-byte signing key and save it (unless using in-memory key)."""
        # Generate 32-byte key for HMAC-SHA256
        key = secrets.token_bytes(32)
        
        # Skip file operations for in-memory keys
        if self.use_in_memory_key:
            logger.debug("Generated in-memory signing key")
            return key
        
        try:
            # Ensure directory exists
            self.key_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save key with restrictive permissions
            self.key_file_path.write_bytes(key)
            
            # Set restrictive file permissions (owner read/write only)
            try:
                self.key_file_path.chmod(0o600)
            except Exception as e:
                logger.warning(f"Failed to set restrictive permissions on key file: {e}")
            
            logger.info(f"Generated new signing key: {self.key_file_path}")
            return key
            
        except Exception as e:
            logger.error(f"Failed to generate signing key: {e}")
            # Fallback to in-memory key (not persistent)
            logger.warning("Using in-memory signing key (not persistent)")
            return key
    
    def _create_signature_payload(self, entry_data: Dict[str, Any]) -> str:
        """
        Create deterministic payload string from entry data.
        
        Args:
            entry_data: Complete entry data dictionary
            
        Returns:
            Deterministic string representation of signed fields
        """
        # Extract values in consistent order (sorted field names)
        values = []
        for field in sorted(self.signed_fields):
            value = entry_data.get(field)
            
            # Handle None values
            if value is None:
                value = ""
            # Convert datetime objects to ISO format
            elif isinstance(value, datetime):
                value = value.isoformat()
            # Convert to string
            else:
                value = str(value)
            
            values.append(f"{field}:{value}")
        
        payload = "|".join(values)
        logger.debug(f"Signature payload: {payload[:100]}..." if len(payload) > 100 else f"Signature payload: {payload}")
        return payload
    
    def sign_entry(self, entry_data: Dict[str, Any]) -> str:
        """
        Create HMAC signature for cache entry.
        
        Args:
            entry_data: Complete entry data dictionary containing all fields
            
        Returns:
            Hex string of HMAC-SHA256 signature
        """
        try:
            # Create deterministic payload
            payload = self._create_signature_payload(entry_data)
            
            # Create HMAC signature
            signature = hmac.new(
                self.secret_key,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            logger.debug(f"Created signature for entry {entry_data.get('cache_key', 'unknown')}")
            return signature
            
        except Exception as e:
            logger.error(f"Failed to create signature: {e}")
            raise
    
    def verify_entry(self, entry_data: Dict[str, Any], stored_signature: str) -> bool:
        """
        Verify HMAC signature for cache entry.
        
        Args:
            entry_data: Complete entry data dictionary
            stored_signature: Previously stored signature to verify against
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Generate expected signature
            expected_signature = self.sign_entry(entry_data)
            
            # Use constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(expected_signature, stored_signature)
            
            if not is_valid:
                logger.warning(f"Signature verification failed for entry {entry_data.get('cache_key', 'unknown')}")
            else:
                logger.debug(f"Signature verified for entry {entry_data.get('cache_key', 'unknown')}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False
    
    def get_field_info(self) -> Dict[str, Any]:
        """Get information about the current signing configuration."""
        return {
            "signed_fields": self.signed_fields,
            "key_file": str(self.key_file_path),
            "key_exists": self.key_file_path.exists() if not self.use_in_memory_key else False,
            "use_in_memory_key": self.use_in_memory_key
        }


def create_cache_signer(cache_dir: Path, key_file: str = "cache_signing_key.bin",
                       custom_fields: Optional[List[str]] = None,
                       use_in_memory_key: bool = False) -> CacheEntrySigner:
    """
    Factory function to create a cache entry signer.
    
    Args:
        cache_dir: Cache directory where key file will be stored
        key_file: Name of the signing key file (ignored if use_in_memory_key=True)
        custom_fields: Custom list of fields to sign (if None, uses default enhanced fields)
        use_in_memory_key: If True, use in-memory key instead of persistent file
        
    Returns:
        Configured CacheEntrySigner instance
    """
    key_file_path = cache_dir / key_file
    return CacheEntrySigner(key_file_path, custom_fields, use_in_memory_key)
