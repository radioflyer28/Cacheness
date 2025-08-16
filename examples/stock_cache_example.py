"""
Stock Data Cache Example

This example demonstrates how to create a pull-through cache for stock market data
using the SQLAlchemy-based SQL cache. The cache automatically fetches missing data
from Yahoo Finance and stores it locally for fast subsequent access.

Key Features Demonstrated:
- Table schema definition with SQLAlchemy
- Custom data adapter implementation
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
from sqlalchemy import (
    MetaData, Table, Column, String, Date, Float, BigInteger, Index
)

# Import the SQL cache components
try:
    from cacheness.sql_cache import SqlCache, SqlCacheAdapter
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


# Define the stock table schema
metadata = MetaData()

stock_table = Table(
    'stock_prices',
    metadata,
    Column('symbol', String(10), primary_key=True, nullable=False),
    Column('date', Date, primary_key=True, nullable=False),
    Column('open', Float),
    Column('high', Float),
    Column('low', Float),
    Column('close', Float),
    Column('volume', BigInteger),
    Column('adjusted_close', Float),
    
    # Add indexes for better query performance
    Index('idx_symbol_date', 'symbol', 'date'),
    Index('idx_symbol', 'symbol'),
    Index('idx_date', 'date')
)


class StockSqlCacheAdapter(SqlCacheAdapter):
    """
    Data adapter for fetching stock market data from Yahoo Finance.
    """
    
    def get_table_definition(self) -> Table:
        """Return the stock table definition"""
        return stock_table
    
    def parse_query_params(self, **kwargs) -> dict:
        """
        Parse and validate stock query parameters.
        
        Expected parameters:
        - symbol: Stock symbol (e.g., 'AAPL')
        - start_date: Start date for data range
        - end_date: End date for data range
        """
        if 'symbol' not in kwargs:
            raise ValueError("Symbol is required")
        
        if 'start_date' not in kwargs or 'end_date' not in kwargs:
            raise ValueError("Both start_date and end_date are required")
        
        return {
            'symbol': kwargs['symbol'].upper(),
            'date': {
                'start': pd.to_datetime(kwargs['start_date']).date(),
                'end': pd.to_datetime(kwargs['end_date']).date()
            }
        }
    
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance API.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            pd.DataFrame: Stock data with columns matching the table schema
        """
        symbol = kwargs.get('symbol')
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        
        print(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Reset index to get Date as a column
                data = data.reset_index()
                data['symbol'] = symbol
                
                # Clean column names to match our table schema
                column_mapping = {
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adjusted_close'
                }
                data = data.rename(columns=column_mapping)
                
                # Convert date to date type (remove time component)
                if 'date' in data.columns:
                    data['date'] = data['date'].dt.date
                
                # Select only columns that exist in our table
                table_cols = [col.name for col in stock_table.columns]
                available_cols = [col for col in table_cols if col in data.columns]
                data = data[available_cols]
                
                print(f"Retrieved {len(data)} records for {symbol}")
            else:
                print(f"No data found for {symbol}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()


class StockCache(SqlCache):
    """
    Stock-specific cache implementation with intelligent missing data detection.
    """
    
    def __init__(self, db_path: str = "stock_cache.db", ttl_hours: int = 24):
        """
        Initialize stock cache.
        
        Args:
            db_path: Path to the database file
            ttl_hours: Cache TTL in hours (24 hours = refresh daily)
        """
        adapter = StockSqlCacheAdapter()
        super().__init__(
            db_url=f"duckdb:///{db_path}",
            table=adapter.get_table_definition(),
            data_adapter=adapter,
            ttl_hours=ttl_hours
        )
    
    def _find_missing_data(
        self, 
        query_params: dict, 
        cached_data: pd.DataFrame
    ) -> list:
        """
        Find missing date ranges for stock data.
        
        This method analyzes the requested date range and existing cached data
        to determine what data needs to be fetched from the external source.
        """
        symbol = query_params['symbol']
        date_range = query_params['date']
        start_date = pd.to_datetime(date_range['start'])
        end_date = pd.to_datetime(date_range['end'])
        
        if cached_data.empty:
            # No cached data, fetch everything
            return [{
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }]
        
        # Find missing business days (stock market doesn't trade on weekends)
        expected_dates = set(pd.bdate_range(start_date, end_date))
        cached_dates = set(cached_data['date'])
        missing_dates = expected_dates - cached_dates
        
        if not missing_dates:
            return []
        
        # Group consecutive missing dates into ranges to minimize API calls
        sorted_missing = sorted(missing_dates)
        ranges = []
        
        if sorted_missing:
            range_start = sorted_missing[0]
            range_end = sorted_missing[0]
            
            for date in sorted_missing[1:]:
                # Allow gaps of up to 3 days (for weekends)
                if (date - range_end).days <= 3:
                    range_end = date
                else:
                    # End current range and start a new one
                    ranges.append({
                        'symbol': symbol,
                        'start_date': range_start.strftime('%Y-%m-%d'),
                        'end_date': range_end.strftime('%Y-%m-%d')
                    })
                    range_start = date
                    range_end = date
            
            # Add the final range
            ranges.append({
                'symbol': symbol,
                'start_date': range_start.strftime('%Y-%m-%d'),
                'end_date': range_end.strftime('%Y-%m-%d')
            })
        
        return ranges


def demo_stock_cache():
    """Demonstrate the stock cache functionality"""
    print("Stock Data Cache Demo")
    print("=" * 50)
    
    # Create cache instance
    cache = StockCache("stock_demo.db", ttl_hours=24)
    
    # Define some test queries
    test_queries = [
        {
            'symbol': 'AAPL',
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'description': 'Apple stock for January 2024'
        },
        {
            'symbol': 'GOOGL',
            'start_date': '2024-01-15',
            'end_date': '2024-02-15',
            'description': 'Google stock for mid-Jan to mid-Feb 2024'
        },
        {
            'symbol': 'AAPL',
            'start_date': '2024-01-15',
            'end_date': '2024-01-25',
            'description': 'Apple stock subset (should hit cache)'
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query['description']}")
        print(f"Symbol: {query['symbol']}, Range: {query['start_date']} to {query['end_date']}")
        
        start_time = datetime.now()
        
        # Get data (will fetch from API or cache as needed)
        data = cache.get_data(**{k: v for k, v in query.items() if k != 'description'})
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"Retrieved {len(data)} records in {elapsed:.2f}s")
        
        if not data.empty:
            print(f"Date range: {data['date'].min()} to {data['date'].max()}")
            print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Show cache statistics
    print("\nCache Statistics:")
    print("=" * 30)
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Demonstrate cache management
    print("\nCache Management Demo:")
    print("=" * 30)
    
    # Show expired entries cleanup
    expired_count = cache.cleanup_expired()
    print(f"Cleaned up {expired_count} expired entries")
    
    # Demonstrate invalidation
    print("Invalidating Apple data...")
    invalidated = cache.invalidate_cache(symbol='AAPL')
    print(f"Invalidated {invalidated} entries")
    
    # Updated stats
    stats = cache.get_cache_stats()
    print(f"Total records after cleanup: {stats['total_records']}")
    
    # Clean up
    cache.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_stock_cache()
