#!/usr/bin/env python3
"""
Beginner SQL Cache Example
==========================

Shows basic SQL caching for new users. Minimal setup, clear concepts.
Uses the new builder pattern - no inheritance required!

Usage:
    python beginner_sql_cache.py
"""

from cacheness.sql_cache import SqlCache
from sqlalchemy import Float, Integer
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(symbol, start_date, end_date):
    """Simulate fetching data from an API."""
    print(f"🌐 Fetching {symbol} from {start_date} to {end_date}")
    
    # Simulate data - in real app this would be API call
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.date())
        current += timedelta(days=1)
    
    return pd.DataFrame({
        'symbol': symbol,
        'date': dates,
        'price': [100.0 + i for i in range(len(dates))],
        'volume': [1000 + i * 10 for i in range(len(dates))]
    })

def main():
    """Demonstrate simple SQL caching."""
    
    print("=== Beginner SQL Cache Demo ===\n")
    
    # Create cache with the new builder - no adapter inheritance!
    cache = SqlCache.for_timeseries(
        "demo.db",
        data_fetcher=fetch_stock_data,
        ttl_seconds=7200,  # 2 hours
        price=Float,      # Additional columns
        volume=Integer
    )
    
    # First request - fetches from "API"
    data1 = cache.get_data(
        symbol="AAPL", 
        start_date="2024-01-01", 
        end_date="2024-01-07"
    )
    print(f"✅ Got {len(data1)} rows for AAPL\n")
    
    # Second request - uses cache 
    data2 = cache.get_data(
        symbol="AAPL", 
        start_date="2024-01-01", 
        end_date="2024-01-07"
    )
    print(f"✅ Cached: {len(data2)} rows for AAPL\n")
    
    # Partial overlap - fetches only missing data
    data3 = cache.get_data(
        symbol="AAPL", 
        start_date="2024-01-05", 
        end_date="2024-01-10"
    )
    print(f"✅ Smart fetch: {len(data3)} rows for AAPL (Jan 5-10)")

if __name__ == "__main__":
    main()