#!/usr/bin/env python3
"""
Configuration Basics
====================

Common configuration patterns: custom cache directories,
TTL strategies, and metadata backends.

Usage:
    uv run python examples/simple_config_demo.py
"""

from cacheness import cached, cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig


# -- Custom cache directory ------------------------------------------------
@cached.for_api(ttl="6h", cache_dir="./weather_cache")
def fetch_weather(city):
    """Cached to a custom directory."""
    print(f"  Fetching weather for {city}...")
    return {"city": city, "temp": 72, "conditions": "sunny"}


# -- Short vs long TTL -----------------------------------------------------
@cached(ttl="2h")
def get_user_status(user_id):
    """Frequently changing data — short TTL."""
    print(f"  Checking status for user {user_id}...")
    return {"user_id": user_id, "status": "online"}


@cached(ttl="2d")
def generate_report(month, year):
    """Expensive, rarely changing — long TTL."""
    print(f"  Generating report for {month}/{year}...")
    return {"month": month, "year": year, "revenue": 150_000}


# -- Explicit cache instance -----------------------------------------------
config = CacheConfig(
    cache_dir="./analytics_cache",
    metadata_backend="sqlite",
    default_ttl="12h",
    max_cache_size="500mb",
)
analytics_cache = cacheness(config)


@cached(cache_instance=analytics_cache, ttl="1d")
def run_analytics(query):
    """Uses a dedicated cache instance with SQLite backend."""
    print(f"  Running analytics: {query}...")
    return {"query": query, "rows": 42_000}


if __name__ == "__main__":
    # Custom directory
    w = fetch_weather("Seattle")
    print(f"Weather: {w}")
    w = fetch_weather("Seattle")  # cached
    print(f"Cached:  {w}\n")

    # Short TTL
    s = get_user_status(123)
    print(f"Status: {s}")
    s = get_user_status(123)  # cached
    print(f"Cached: {s}\n")

    # Long TTL
    r = generate_report("March", 2024)
    print(f"Report: {r}")
    r = generate_report("March", 2024)  # cached
    print(f"Cached: {r}\n")

    # Dedicated cache instance
    a = run_analytics("top customers")
    print(f"Analytics: {a}")
    a = run_analytics("top customers")  # cached
    print(f"Cached:    {a}\n")

    # Non-destructive get — preserve entries on deserialization errors
    safe_config = CacheConfig(
        cache_dir="./safe_cache",
        metadata=CacheMetadataConfig(delete_on_error=False),
    )
    safe_cache = cacheness(safe_config)
    safe_cache.put({"important": True}, key="keep_me")
    print(
        f"Non-destructive get config: delete_on_error={safe_config.metadata.delete_on_error}"
    )
