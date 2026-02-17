#!/usr/bin/env python3
"""
API Response Caching
====================

Use @cached.for_api() to cache HTTP responses.
It adds automatic error handling on top of normal caching.

Usage:
    uv run python examples/simple_api_caching.py
"""

from cacheness import cached


@cached.for_api(ttl="6h")
def get_weather(city):
    """Fetch weather data (simulated)."""
    print(f"  Calling weather API for {city}...")
    return {"city": city, "temp_f": 72, "conditions": "sunny"}


@cached(ttl="4h")
def get_stock_price(symbol):
    """Fetch stock price (simulated)."""
    print(f"  Calling stock API for {symbol}...")
    return {"symbol": symbol, "price": 150.25}


if __name__ == "__main__":
    weather = get_weather("seattle")
    print(f"Weather: {weather}")

    weather = get_weather("seattle")  # cached
    print(f"Cached:  {weather}\n")

    stock = get_stock_price("AAPL")
    print(f"Stock: {stock}")

    stock = get_stock_price("AAPL")  # cached
    print(f"Cached: {stock}")
