#!/usr/bin/env python3
"""
Cross-Platform Verification Script

Verifies that Cacheness works correctly on the current platform.
Tests core functionality and platform-specific considerations.
"""

import platform
import sys
import tempfile
from pathlib import Path
import shutil

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_success(text):
    """Print success message."""
    try:
        print(f"✅ {text}")
    except UnicodeEncodeError:
        print(f"[OK] {text}")

def print_warning(text):
    """Print warning message."""
    try:
        print(f"⚠️  {text}")
    except UnicodeEncodeError:
        print(f"[WARNING] {text}")

def print_error(text):
    """Print error message."""
    try:
        print(f"❌ {text}")
    except UnicodeEncodeError:
        print(f"[ERROR] {text}")

def check_platform_info():
    """Display platform information."""
    print_header("Platform Information")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Python Implementation: {platform.python_implementation()}")

def check_imports():
    """Verify all required imports work."""
    print_header("Checking Imports")
    
    try:
        from cacheness import cacheness, CacheConfig
        print_success("Core imports successful")
    except ImportError as e:
        print_error(f"Core import failed: {e}")
        return False
    
    # Check optional dependencies
    optional = {
        "numpy": "NumPy arrays",
        "pandas": "Pandas DataFrames",
        "polars": "Polars DataFrames",
        "sqlalchemy": "SQLite backend",
        "blosc2": "Blosc2 compression",
    }
    
    for module, description in optional.items():
        try:
            __import__(module)
            print_success(f"{description} available")
        except ImportError:
            print_warning(f"{description} not installed (optional)")
    
    return True

def test_basic_caching():
    """Test basic caching operations."""
    print_header("Testing Basic Caching")
    
    try:
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create cache instance
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            # Test put/get
            test_data = {"test": "data", "number": 42}
            cache.put(test_data, key="test_key", version=1)
            print_success("Put operation successful")
            
            retrieved = cache.get(key="test_key", version=1)
            assert retrieved == test_data, "Retrieved data doesn't match"
            print_success("Get operation successful")
            
            # Test list entries
            entries = cache.list_entries()
            assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
            print_success("List entries successful")
            
            # Clean up
            cache.close()
            print_success("Cache close successful")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_error(f"Basic caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlite_backend():
    """Test SQLite backend functionality."""
    print_header("Testing SQLite Backend")
    
    try:
        import sqlalchemy
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
            )
            cache = cacheness(config)
            
            # Test operations
            cache.put({"data": "test"}, experiment="exp1")
            cache.put({"data": "test2"}, experiment="exp2")
            print_success("SQLite put operations successful")
            
            entries = cache.list_entries()
            assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
            print_success("SQLite query successful")
            
            cache.close()
            print_success("SQLite backend test passed")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except ImportError:
        print_warning("SQLAlchemy not installed, skipping SQLite tests")
        return True
    except Exception as e:
        print_error(f"SQLite backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_manager():
    """Test context manager functionality."""
    print_header("Testing Context Manager")
    
    try:
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            
            # Use context manager
            with cacheness(config) as cache:
                cache.put({"data": "test"}, key="ctx")
                result = cache.get(key="ctx")
                assert result == {"data": "test"}
            
            print_success("Context manager works correctly")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_error(f"Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print_header("Cacheness Cross-Platform Verification")
    
    tests = [
        ("Platform Info", check_platform_info),
        ("Imports", check_imports),
        ("Basic Caching", test_basic_caching),
        ("SQLite Backend", test_sqlite_backend),
        ("Context Manager", test_context_manager),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            if name == "Platform Info":
                test_func()
                results[name] = True
            else:
                results[name] = test_func()
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_header("Verification Summary")
    
    # Count only test results (exclude Platform Info)
    test_results = {k: v for k, v in results.items() if k != "Platform Info"}
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for name, result in test_results.items():
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print_success(f"All tests passed on {platform.system()}!")
        print("\nCacheness is fully compatible with your platform.")
        return 0
    else:
        print_warning(f"Some tests failed on {platform.system()}")
        print("\nPlease report any issues at: https://github.com/radioflyer28/cacheness/issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
