#!/usr/bin/env python3
"""
Simple Function Caching
========================

Add @cached to any function to cache its results on disk.
Subsequent calls with the same arguments return instantly.

Usage:
    uv run python examples/simple_function_caching.py
"""

from cacheness import cached


@cached(ttl_seconds="1h")
def expensive_computation(n):
    """Simulate a slow computation."""
    print(f"  Computing fib({n})...")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@cached.for_api(ttl_seconds="6h")
def fetch_user(user_id):
    """Simulate an API call — for_api adds error handling."""
    print(f"  Fetching user {user_id} from API...")
    return {"id": user_id, "name": f"User {user_id}", "active": True}


if __name__ == "__main__":
    # First call — executes the function
    result = expensive_computation(50)
    print(f"Result: {result}")

    # Second call — served from cache (no "Computing..." printed)
    result = expensive_computation(50)
    print(f"Cached: {result}\n")

    # Same for API-style caching
    user = fetch_user(42)
    print(f"User: {user}")

    user = fetch_user(42)
    print(f"Cached: {user}")
