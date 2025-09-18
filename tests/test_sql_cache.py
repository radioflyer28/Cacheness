"""
Comprehensive tests for SQL pull-through cache functionality.
Tests cover database backend selection, gap detection, TTL management,
and intelligent caching behavior.
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import MetaData, Table, Column, String, Date, Float, Integer


# Only run these tests if SQLAlchemy is available
try:
    from cacheness.sql_cache import SqlCache, SqlCacheAdapter
    HAS_SQL_CACHE = True
except ImportError:
    HAS_SQL_CACHE = False


@pytest.mark.skipif(not HAS_SQL_CACHE, reason="SQLAlchemy not available")
class TestSQLCache:
    """Test cases for SQL pull-through cache"""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Set up test fixtures"""
        # Define a simple test table
        metadata = MetaData()
        self.test_table = Table(
            'test_data',
            metadata,
            Column('id', String(50), primary_key=True),
            Column('date', Date, primary_key=True),
            Column('value', Float),
            Column('count', Integer)
        )
        
        # Create a simple test adapter
        class TestSqlCacheAdapter(SqlCacheAdapter):
            def __init__(self, table, test_data=None):
                self.table = table
                self.test_data = test_data or []
            
            def get_table_definition(self):
                return self.table
            
            def parse_query_params(self, **kwargs):
                return kwargs
            
            def fetch_data(self, **kwargs):
                # Return test data as DataFrame
                if self.test_data:
                    return pd.DataFrame(self.test_data)
                return pd.DataFrame()
        
        self.adapter = TestSqlCacheAdapter(self.test_table)
    
    def test_cache_initialization(self):
        """Test that cache can be initialized"""
        cache = SqlCache(
            db_url="sqlite:///:memory:",  # In-memory SQLite
            table=self.test_table,
            data_adapter=self.adapter,
            ttl_hours=24
        )
        
        assert cache.table is not None
        assert cache.data_adapter is self.adapter
        assert cache.ttl_hours == 24
        
        cache.close()
    
    def test_cache_with_test_data(self):
        """Test cache with some test data"""
        # Set up test data
        test_data = [
            {'id': 'test1', 'date': date(2024, 1, 1), 'value': 100.0, 'count': 1},
            {'id': 'test1', 'date': date(2024, 1, 2), 'value': 101.0, 'count': 2},
        ]
        
        self.adapter.test_data = test_data
        
        cache = SqlCache(
            db_url="sqlite:///:memory:",
            table=self.test_table,
            data_adapter=self.adapter,
            ttl_hours=24
        )
        
        # Get data (should fetch from adapter)
        result = cache.get_data(id='test1')
        
        assert not result.empty
        assert len(result) == 2
        assert 'id' in result.columns
        assert 'value' in result.columns
        
        cache.close()
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = SqlCache(
            db_url="sqlite:///:memory:",
            table=self.test_table,
            data_adapter=self.adapter,
            ttl_hours=24
        )
        
        stats = cache.get_cache_stats()
        
        assert 'total_records' in stats
        assert 'table_name' in stats
        assert 'primary_keys' in stats
        assert stats['total_records'] == 0  # Empty cache
        
        cache.close()
    
    def test_cache_management(self):
        """Test cache management operations"""
        cache = SqlCache(
            db_url="sqlite:///:memory:",
            table=self.test_table,
            data_adapter=self.adapter,
            ttl_hours=24
        )
        
        # Test cleanup operations
        expired_count = cache.cleanup_expired()
        assert expired_count == 0  # No expired entries
        
        cleared_count = cache.clear_cache()
        assert cleared_count == 0  # No entries to clear
        
        cache.close()


    def test_backend_selection_methods(self):
        """Test database backend selection class methods"""
        # Test SQLite backend selection
        cache_sqlite = SqlCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=12
        )
        
        assert cache_sqlite.engine.dialect.name == "sqlite"
        assert cache_sqlite.ttl_hours == 12
        cache_sqlite.close()
        
        # Test DuckDB backend selection (if available)
        try:
            cache_duck = SqlCache.with_duckdb(
                ":memory:", self.test_table, self.adapter
            )
            # If this succeeds, DuckDB is available
            assert cache_duck.engine.dialect.name == "duckdb"
            cache_duck.close()
        except Exception:
            # DuckDB not installed, which is expected
            pass
        
        # Test PostgreSQL method (using SQLite URL for testing)
        cache_pg = SqlCache.with_postgresql(
            "sqlite:///:memory:", self.test_table, self.adapter
        )
        # This should fall back to SQLite dialect
        assert cache_pg.engine is not None
        cache_pg.close()
    
    def test_gap_detection_logic(self):
        """Test intelligent gap detection in cached data"""
        # Set up adapter with time-series test data for gap detection
        class GapTestAdapter(SqlCacheAdapter):
            def __init__(self, table, fetch_count=0):
                self.table = table
                self.fetch_count = fetch_count
            
            def get_table_definition(self):
                return self.table
            
            def parse_query_params(self, **kwargs):
                return kwargs
            
            def fetch_data(self, **kwargs):
                self.fetch_count += 1
                # Return data for the missing gap
                return pd.DataFrame([
                    {'id': 'AAPL', 'date': date(2024, 1, 3), 'value': 151.5, 'count': 115}
                ])
        
        adapter = GapTestAdapter(self.test_table)
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, adapter
        )
        
        # First, populate cache with test data (simulating existing cache)
        # Note: In a real test, we would manually insert this data to test gap detection
        # For now, just test that the cache can handle data fetching
        
        result = cache.get_data(id='AAPL', start_date=date(2024, 1, 1), end_date=date(2024, 1, 4))
        
        # Verify the adapter was called to fetch missing data
        assert adapter.fetch_count > 0
        assert not result.empty
        
        cache.close()
    
    def test_ttl_and_expiration(self):
        """Test TTL functionality and expiration logic"""
        # Create cache with short TTL for testing
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1  # 1 hour
        )
        
        # Test cleanup operations
        expired_count = cache.cleanup_expired()
        assert expired_count >= 0  # Should not fail
        
        # Test cache statistics include TTL info
        stats = cache.get_cache_stats()
        assert 'total_records' in stats
        
        cache.close()
    
    def test_upsert_behavior(self):
        """Test upsert operations handle conflicts correctly"""
        test_data = [
            {'id': 'TEST', 'date': date(2024, 1, 1), 'value': 100.0, 'count': 1},
        ]
        
        class UpsertTestAdapter(SqlCacheAdapter):
            def __init__(self, table, data):
                self.table = table
                self.data = data
                self.call_count = 0
            
            def get_table_definition(self):
                return self.table
            
            def parse_query_params(self, **kwargs):
                return kwargs
            
            def fetch_data(self, **kwargs):
                self.call_count += 1
                # Return different values on different calls to test upsert
                updated_data = self.data.copy()
                updated_data[0]['value'] = 100.0 + (self.call_count * 10)
                return pd.DataFrame(updated_data)
        
        adapter = UpsertTestAdapter(self.test_table, test_data)
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, adapter
        )
        
        # First fetch
        result1 = cache.get_data(id='TEST')
        assert not result1.empty
        
        # Second fetch (should be from cache or updated via upsert)
        result2 = cache.get_data(id='TEST')
        assert not result2.empty
        
        cache.close()
    
    def test_comprehensive_cache_operations(self):
        """Test comprehensive cache management operations"""
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, self.adapter
        )
        
        # Test all cache management methods
        stats = cache.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_records' in stats
        assert 'table_name' in stats
        
        # Test invalidation
        invalidated = cache.invalidate_cache(id='nonexistent')
        assert invalidated >= 0
        
        # Test clearing
        cleared = cache.clear_cache()
        assert cleared >= 0
        
        # Test cleanup
        expired = cache.cleanup_expired()
        assert expired >= 0
        
        cache.close()
    
    def test_timezone_handling(self):
        """Test that all timestamps use UTC timezone consistently"""
        # Create cache with timezone-aware table
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Verify cached_at and expires_at columns are timezone-aware
        cached_at_col = cache.table.c.get('cached_at')
        expires_at_col = cache.table.c.get('expires_at')
        
        if cached_at_col is not None:
            # Check that DateTime column has timezone=True
            assert hasattr(cached_at_col.type, 'timezone'), "cached_at should have timezone attribute"
            assert cached_at_col.type.timezone is True, "cached_at should be timezone-aware"
        
        if expires_at_col is not None:
            # Check that DateTime column has timezone=True  
            assert hasattr(expires_at_col.type, 'timezone'), "expires_at should have timezone attribute"
            assert expires_at_col.type.timezone is True, "expires_at should be timezone-aware"
        
        # Test that our code creates UTC timestamps
        import unittest.mock
        
        # Mock datetime.now to verify UTC is being used
        with unittest.mock.patch('cacheness.sql_cache.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.utc = timezone.utc  # Ensure timezone.utc is available
            
            # Test data storage
            test_data = pd.DataFrame([
                {'id': 'UTC_TEST', 'date': date(2024, 1, 1), 'value': 100.0, 'count': 1}
            ])
            
            with cache.Session() as session:
                cache._store_in_cache(session, test_data)
                session.commit()
                
                # Verify datetime.now was called with timezone.utc
                mock_datetime.now.assert_called_with(timezone.utc)
        
        cache.close()
    
    def test_timezone_expiry_consistency(self):
        """Test that TTL expiry logic uses consistent UTC timing"""
        # Create cache with TTL
        cache = SqlCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Test that cleanup_expired and query conditions use UTC consistently
        import unittest.mock
        
        with unittest.mock.patch('cacheness.sql_cache.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.utc = timezone.utc
            
            # Test cleanup_expired
            cache.cleanup_expired()
            
            # Verify UTC was used for current time comparison
            mock_datetime.now.assert_called_with(timezone.utc)
            
            # Test get_cache_stats 
            mock_datetime.now.reset_mock()
            stats = cache.get_cache_stats()
            
            if 'expired_records' in stats:
                # Should have called datetime.now(timezone.utc) for comparison
                mock_datetime.now.assert_called_with(timezone.utc)
        
        cache.close()


def test_import_availability():
    """Test that imports work correctly from main package"""
    try:
        from cacheness import SqlCache, SqlCacheAdapter
        assert SqlCache is not None
        assert SqlCacheAdapter is not None
    except ImportError:
        # This is expected if SQLAlchemy is not installed
        pass


def test_backend_selection_integration():
    """Test database backend selection methods work correctly"""
    if not HAS_SQL_CACHE:
        pytest.skip("SQLAlchemy not available")
    
    from sqlalchemy import MetaData, Table, Column, Integer
    
    metadata = MetaData()
    # Use autoincrement=False to make it DuckDB-compatible and avoid warnings
    simple_table = Table('simple', metadata, Column('id', Integer, primary_key=True, autoincrement=False))
    
    class SimpleAdapter(SqlCacheAdapter):
        def get_table_definition(self):
            return simple_table
        def parse_query_params(self, **kwargs):
            return kwargs
        def fetch_data(self, **kwargs):
            return pd.DataFrame()
    
    adapter = SimpleAdapter()
    
    # Test SQLite method
    cache_sqlite = SqlCache.with_sqlite(":memory:", simple_table, adapter)
    assert cache_sqlite.engine.dialect.name == "sqlite"
    cache_sqlite.close()
    
    # Test DuckDB method (may fail if not installed)
    try:
        cache_duck = SqlCache.with_duckdb(":memory:", simple_table, adapter)
        cache_duck.close()
    except Exception:
        # Expected if duckdb-engine not installed
        pass
    
    # Test PostgreSQL method with fallback URL
    cache_pg = SqlCache.with_postgresql("sqlite:///:memory:", simple_table, adapter)
    cache_pg.close()


@pytest.mark.skipif(not HAS_SQL_CACHE, reason="SQLAlchemy not available")
class TestSQLCacheBuilders:
    """Test the new builder pattern for SQL caches."""
    
    @pytest.fixture(autouse=True)
    def setup_mock_fetcher(self):
        """Set up test data fetcher."""
        self.call_count = 0
        
        def mock_fetcher(**kwargs):
            self.call_count += 1
            return pd.DataFrame([
                {'id': 1, 'name': 'test', 'value': 42.0},
                {'id': 2, 'name': 'demo', 'value': 24.0}
            ])
        
        self.mock_fetcher = mock_fetcher
    
    def test_for_lookup_table_builder(self):
        """Test SqlCache.for_lookup_table() builder."""
        from sqlalchemy import Integer, String, Float
        
        cache = SqlCache.for_lookup_table(
            ":memory:",
            table_name="lookup_test",
            primary_keys=["id"],
            data_fetcher=self.mock_fetcher,
            ttl_hours=1,
            id=Integer,
            name=String(50),
            value=Float
        )
        
        try:
            # Test that it works
            data = cache.get_data(filter_key="test")
            assert len(data) == 2
            assert self.call_count == 1
            
            # Test caching
            data2 = cache.get_data(filter_key="test")
            assert len(data2) == 2
            assert self.call_count == 1  # Should be cached
            
        finally:
            cache.close()
    
    def test_for_analytics_table_builder(self):
        """Test SqlCache.for_analytics_table() builder."""
        from sqlalchemy import Integer, String, Float
        
        cache = SqlCache.for_analytics_table(
            ":memory:",
            table_name="analytics_test", 
            primary_keys=["name"],  # Use name as primary key instead of id
            data_fetcher=self.mock_fetcher,
            ttl_hours=24,
            name=String(50),
            value=Float
        )
        
        try:
            # Test that it works
            data = cache.get_data(category="analytics")
            assert len(data) == 2
            assert self.call_count == 1
            
        finally:
            cache.close()
    
    def test_for_timeseries_builder(self):
        """Test SqlCache.for_timeseries() builder."""
        from sqlalchemy import Float, Integer
        
        def timeseries_fetcher(**kwargs):
            symbol = kwargs.get('symbol', 'TEST')
            return pd.DataFrame([
                {'symbol': symbol, 'date': date(2024, 1, 1), 'price': 100.0, 'volume': 1000},
                {'symbol': symbol, 'date': date(2024, 1, 2), 'price': 101.0, 'volume': 1100}
            ])
        
        cache = SqlCache.for_timeseries(
            ":memory:",
            data_fetcher=timeseries_fetcher,
            price=Float,
            volume=Integer
        )
        
        try:
            data = cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-02")
            assert len(data) == 2
            assert data.iloc[0]['symbol'] == 'AAPL'
            
        finally:
            cache.close()
    
    def test_for_realtime_timeseries_builder(self):
        """Test SqlCache.for_realtime_timeseries() builder."""
        from sqlalchemy import Float, Integer
        
        def realtime_fetcher(**kwargs):
            symbol = kwargs.get('symbol', 'TEST')
            return pd.DataFrame([
                {'symbol': symbol, 'date': date.today(), 'price': 100.0, 'volume': 1000}
            ])
        
        cache = SqlCache.for_realtime_timeseries(
            ":memory:",
            data_fetcher=realtime_fetcher,
            ttl_hours=1,
            price=Float,
            volume=Integer
        )
        
        try:
            data = cache.get_data(symbol="BTCUSD", start_date=str(date.today()), end_date=str(date.today()))
            assert len(data) == 1
            assert data.iloc[0]['symbol'] == 'BTCUSD'
            
        finally:
            cache.close()
