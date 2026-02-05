#!/usr/bin/env python3
"""
Simple Stock Cache Example
=========================

Demonstrates basic stock data caching using the new builder pattern.
Much simpler than the advanced example!

Requirements:
    pip install yfinance

Usage:
    python simple_stock_cache.py
"""

import pandas as pd
from datetime import datetime, timedelta
from cacheness.sql_cache import SqlCache
from sqlalchemy import Float, Integer

try:
    import yfinance as yf
except ImportError:
    print("📦 Installing yfinance...")
    import subprocess
    subprocess.check_call(["pip", "install", "yfinance"])
    import yfinance as yf

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    print(f"📈 Fetching {symbol} from {start_date} to {end_date}")
    
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"⚠️  No data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to the format our cache expects
        result = data.reset_index()
        result['symbol'] = symbol
        result['date'] = result['Date'].dt.date
        result = result[['symbol', 'date', 'Close', 'Volume']].copy()
        result.columns = ['symbol', 'date', 'close_price', 'volume']
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

def main():
    """Demonstrate simple stock caching."""
    
    print("=== Simple Stock Cache Demo ===\n")
    
    # Create stock cache with the new builder - no complex adapter!
    stock_cache = SqlCache.for_timeseries(
        "simple_stocks.db",
        data_fetcher=fetch_stock_data,
        ttl_seconds=14400,  # 4 hours
        close_price=Float,
        volume=Integer
    )
    
    # Test with a week of data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    print(f"📊 Requesting AAPL data: {start_date} to {end_date}")
    
    # First call - fetches from Yahoo Finance
    data1 = stock_cache.get_data(
        symbol="AAPL",
        start_date=str(start_date),
        end_date=str(end_date)
    )
    print(f"✅ Got {len(data1)} rows of AAPL data")
    if not data1.empty:
        latest = data1.iloc[-1]
        print(f"   Latest: {latest['date']} - ${latest['close_price']:.2f}")
    
    print("\n" + "="*50)
    
    # Second call - uses cache
    print("📊 Requesting same AAPL data again...")
    data2 = stock_cache.get_data(
        symbol="AAPL", 
        start_date=str(start_date),
        end_date=str(end_date)
    )
    print(f"✅ Cached: {len(data2)} rows (no API call!)")
    
    print("\n" + "="*50)
    
    # Try a different symbol
    print("📊 Requesting GOOGL data...")
    data3 = stock_cache.get_data(
        symbol="GOOGL",
        start_date=str(start_date), 
        end_date=str(end_date)
    )
    print(f"✅ Got {len(data3)} rows of GOOGL data")
    if not data3.empty:
        latest = data3.iloc[-1]
        print(f"   Latest: {latest['date']} - ${latest['close_price']:.2f}")
    
    print("\n🎯 Cache automatically handles:")
    print("   • Missing data detection")
    print("   • Efficient storage and retrieval") 
    print("   • TTL-based freshness")
    print("   • No complex adapter code needed!")

if __name__ == "__main__":
    main()