#!/usr/bin/env python3
"""
API Request Caching (Advanced)
==============================

A class-based pattern for caching HTTP calls with per-endpoint TTLs
and a shared SQLite-backed cache instance.

Usage:
    uv run python examples/api_request_caching.py

Requires:
    uv add requests
"""

import requests

from cacheness import cached, cacheness, CacheConfig

# -- Shared cache for all API responses ---------------------------------------
api_config = CacheConfig(
    cache_dir="./api_cache",
    default_ttl="1d",
    metadata_backend="sqlite",
)
api_cache = cacheness(api_config)


class WeatherClient:
    """Weather API wrapper with layered TTLs."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://api.weatherapi.com/v1"

    @cached(cache_instance=api_cache, ttl_seconds="6h", key_prefix="weather")
    def current(self, city: str, units: str = "metric"):
        """Current conditions — refreshed every 6 hours."""
        print(f"  Fetching current weather for {city}...")
        r = requests.get(
            f"{self.base}/current.json",
            params={"key": self.api_key, "q": city, "units": units},
        )
        r.raise_for_status()
        return r.json()

    @cached(cache_instance=api_cache, ttl_seconds="1w", key_prefix="forecast")
    def forecast(self, city: str, days: int = 7):
        """Weekly forecast — refreshed once a week."""
        print(f"  Fetching {days}-day forecast for {city}...")
        r = requests.get(
            f"{self.base}/forecast.json",
            params={"key": self.api_key, "q": city, "days": days},
        )
        r.raise_for_status()
        return r.json()

    @cached(cache_instance=api_cache, ttl_seconds="1y", key_prefix="history")
    def historical(self, city: str, date: str):
        """Historical data — essentially immutable."""
        print(f"  Fetching historical data for {city} on {date}...")
        r = requests.get(
            f"{self.base}/history.json",
            params={"key": self.api_key, "q": city, "dt": date},
        )
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = WeatherClient("YOUR_API_KEY")

    # First call — hits the API
    data = client.current("London")
    print(f"Temp: {data['current']['temp_c']}°C")

    # Second call — served from cache
    data = client.current("London")
    print(f"Cached: {data['current']['temp_c']}°C\n")

    stats = api_cache.get_stats()
    print(
        f"Entries: {stats['total_entries']}, Size: {stats['total_size_bytes']:,} bytes"
    )
