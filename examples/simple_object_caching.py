#!/usr/bin/env python3
"""
Object Caching
==============

Cache any Python object — dataclasses, dicts, lists, nested structures.
Cacheness automatically serialises and deserialises them.

Usage:
    uv run python examples/simple_object_caching.py
"""

from dataclasses import dataclass
from cacheness import cached


@dataclass
class UserProfile:
    user_id: int
    name: str
    preferences: dict


@cached(ttl="12h")
def load_profile(user_id: int) -> UserProfile:
    """Simulate an expensive lookup."""
    print(f"  Loading profile {user_id}...")
    return UserProfile(
        user_id=user_id,
        name=f"User {user_id}",
        preferences={"theme": "dark", "lang": "en"},
    )


@cached(ttl="24h")
def process_items(items: list[str]) -> dict:
    """Simulate heavy processing."""
    print(f"  Processing {len(items)} items...")
    return {
        "processed": [s.upper() for s in items],
        "count": len(items),
    }


if __name__ == "__main__":
    profile = load_profile(123)
    print(f"Profile: {profile}")

    profile = load_profile(123)  # cached
    print(f"Cached:  {profile}\n")

    result = process_items(["hello", "world"])
    print(f"Result: {result}")

    result = process_items(["hello", "world"])  # cached
    print(f"Cached: {result}")
