"""
Stock Data Cache Example - New Builder Pattern

This example demonstrates how to create a pull-through cache for stock market data
using the new simplified builder pattern. The cache automatically fetches missing data
from Yahoo Finance and stores it locally for fast subsequent access.

Key Features Demonstrated:
- Simple function-based data fetching (no inheritance needed)
- Automatic database selection (DuckDB for time-series)
- Intelligent missing data detection for time series
- Automatic upsert operations
- Cache statistics and management

Requirements:
    pip install yfinance sqlalchemy duckdb-engine pandas

Usage:
    python stock_cache_example.py
"""

import pandas as pd
from datetime import datetime
from sqlalchemy import String, Date, Float, BigInteger

# Import the SQL cache components
try:
    from cacheness.sql_cache import SqlCache
except ImportError:
    print("Error: cacheness SQL cache not available")
    print("Make sure SQLAlchemy is installed: pip install 'cacheness[sql]'")
    exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not available")
    print("Install with: pip install yfinance")
    exit(1)


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance API.
    
    This is a simple function that returns a DataFrame - no inheritance needed!
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date for data range
        end_date: End date for data range
        
    Returns:
        pd.DataFrame: Stock data with columns [symbol, date, open, high, low, close, volume, adjusted_close]
    """
    print(f"🔄 Fetching {symbol} data from {start_date} to {end_date}")
    
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"⚠️  No data available for {symbol} in the specified date range")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Add symbol column
        data['symbol'] = symbol.upper()
        
        # Rename columns to match our cache schema
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted_close'
        })
        
        # Select only the columns we need
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        data = data[columns]
        
        print(f"✅ Fetched {len(data)} records for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def demonstrate_stock_cache():
    """Demonstrate the stock cache functionality"""
    
    print("🚀 Stock Cache Example - New Builder Pattern")
    print("=" * 50)
    
    # Create cache using the new builder pattern - it's that simple!
    print("📦 Creating stock cache...")
    stock_cache = SqlCache.for_timeseries(
        "stock_cache.db",  # DuckDB automatically selected for time-series
        data_fetcher=fetch_stock_data,
        ttl_hours=24,  # Refresh daily
        
        # Define schema inline - no table definition needed
        symbol=String(10),
        date=Date,
        open=Float,
        high=Float,
        low=Float,
        close=Float,
        volume=BigInteger,
        adjusted_close=Float
    )
    
    print("✅ Cache created! Using DuckDB for optimal time-series performance")
    print()
    
    # Test symbols and date ranges
    symbols = ["AAPL", "GOOGL", "TSLA"]
    
    for symbol in symbols:
        print(f"📈 Getting {symbol} data...")
        
        # First call - will fetch from API
        start_time = datetime.now()
        data = stock_cache.get_data(
            symbol=symbol,
            start_date="2024-01-01", 
            end_date="2024-01-31"
        )
        first_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"   📊 Retrieved {len(data)} records in {first_duration:.2f}s (API fetch)")
        
        # Second call - should be instant from cache
        start_time = datetime.now()
        cached_data = stock_cache.get_data(
            symbol=symbol,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        second_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"   ⚡ Retrieved {len(cached_data)} records in {second_duration:.4f}s (cached)")
        print(f"   🔥 Cache speedup: {first_duration/second_duration:.1f}x faster!")
        print()
    
    # Demonstrate partial fetching
    print("🔍 Testing intelligent gap detection...")
    print("   Requesting extended date range (will fetch only missing data)")
    
    start_time = datetime.now()
    extended_data = stock_cache.get_data(
        symbol="AAPL",
        start_date="2024-01-01",  # Cached
        end_date="2024-02-15"     # Will fetch new data
    )
    gap_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"   📈 Extended data: {len(extended_data)} records in {gap_duration:.2f}s")
    print("   ✨ Only fetched missing data - cache intelligence at work!")
    print()
    
    # Show cache statistics
    print("📊 Cache Statistics:")
    print("-" * 20)
    stats = stock_cache.get_cache_stats()
    print(f"   Total entries: {stats['total_records']}")
    print(f"   Database size: {stats.get('database_size', 'N/A')}")
    print()
    
    # Show some actual data
    print("📋 Sample Data (AAPL recent):")
    print("-" * 30)
    sample = extended_data.tail(3)[['date', 'close', 'volume']]
    for _, row in sample.iterrows():
        print(f"   {row['date']}: ${row['close']:.2f} (vol: {row['volume']:,})")
    print()
    
    print("🎉 Demo complete! The cache will persist between runs.")
    print("💡 Try running this script again to see instant cache performance.")


if __name__ == "__main__":
    demonstrate_stock_cache()
