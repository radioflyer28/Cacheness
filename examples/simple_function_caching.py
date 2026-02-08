#!/usr/bin/env python3
"""
Simple Function Caching Example
===============================

Shows basic function caching with minimal boilerplate.
Great starting point for new users.

Usage:
    python simple_function_caching.py
"""

from cacheness import cached
import pandas as pd
from datetime import datetime, timedelta


# Simple decorator-based caching for data processing
@cached.for_api(ttl_seconds=14400)  # 4 hours - Optimized for API calls
def get_stock_data(symbol, days=30):
    """Simulate fetching stock data."""
    print(f"📈 Fetching {symbol} data for {days} days...")

    # Simulate data (in real app, this would be API call)
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    return pd.DataFrame(
        {"date": dates, "symbol": symbol, "price": [100 + i for i in range(days)]}
    )


@cached.for_table(ttl_seconds=3600)  # 1 hour - Optimized for tabular data
def expensive_calculation(data, multiplier=1.5):
    """Simulate expensive data processing."""
    print("🔢 Running expensive calculation...")
    # Simulate heavy computation
    result = data.copy()
    result["calculated_value"] = result["price"] * multiplier
    return result


def main():
    """Demonstrate simple function caching."""

    print("=== Simple Function Caching Demo ===\n")

    # First call - fetches data
    df1 = get_stock_data("AAPL", 7)
    print(f"✅ Got {len(df1)} rows for AAPL\n")

    # Second call - uses cache (no API request)
    df2 = get_stock_data("AAPL", 7)
    print(f"✅ Cached: {len(df2)} rows for AAPL\n")

    # Different symbol - new cache entry
    df3 = get_stock_data("GOOGL", 7)
    print(f"✅ Got {len(df3)} rows for GOOGL\n")

    # Cache expensive calculations too
    calc1 = expensive_calculation(df1, 2.0)
    print(f"✅ Calculated {len(calc1)} values")

    # Second calculation with same params - uses cache
    calc2 = expensive_calculation(df1, 2.0)
    print(f"✅ Cached calculation: {len(calc2)} values")


if __name__ == "__main__":
    main()
