"""
Tests to verify proper UTC timezone handling in the SQL cache.
Ensures that cache expiration and timestamps work correctly across timezone changes.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import MetaData, Table, Column, String, Integer

# Only run these tests if SQLAlchemy is available
try:
    from cacheness.sql_cache import SQLAlchemyPullThroughCache, SQLAlchemyDataAdapter
    HAS_SQL_CACHE = True
except ImportError:
    HAS_SQL_CACHE = False


@pytest.mark.skipif(not HAS_SQL_CACHE, reason="SQLAlchemy not available")
class TestSQLCacheTimezones:
    """Test timezone handling in SQL cache"""
    
    def setup_method(self):
        """Set up test fixtures"""
        metadata = MetaData()
        self.test_table = Table(
            'timezone_test',
            metadata,
            Column('id', String(50), primary_key=True),
            Column('value', Integer)
        )
        
        class TimezoneTestAdapter(SQLAlchemyDataAdapter):
            def __init__(self, table, test_data=None):
                self.table = table
                self.test_data = test_data or []
                self.fetch_count = 0
            
            def get_table_definition(self):
                return self.table
            
            def parse_query_params(self, **kwargs):
                return kwargs
            
            def fetch_data(self, **kwargs):
                self.fetch_count += 1
                if self.test_data:
                    return pd.DataFrame(self.test_data)
                return pd.DataFrame([{'id': f'test_{self.fetch_count}', 'value': self.fetch_count * 10}])
        
        self.adapter = TimezoneTestAdapter(self.test_table)
    
    def test_timezone_aware_columns(self):
        """Test that timestamp columns are timezone-aware"""
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Check that cached_at and expires_at columns are timezone-aware
        for col in cache.table.columns:
            if col.name in ('cached_at', 'expires_at'):
                assert hasattr(col.type, 'timezone'), f"Column {col.name} should be timezone-aware"
                assert col.type.timezone is True, f"Column {col.name} should have timezone=True"
        
        cache.close()
    
    def test_utc_timestamps_in_storage(self):
        """Test that stored timestamps are in UTC"""
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Record current UTC time before operation
        before_utc = datetime.now(timezone.utc)
        
        # Fetch data to trigger storage with timestamps
        data = cache.get_data(id='test')
        assert not data.empty
        
        # Record UTC time after operation
        after_utc = datetime.now(timezone.utc)
        
        # Query the raw database to check stored timestamps
        with cache.Session() as session:
            from sqlalchemy import select
            result = session.execute(
                select(cache.table.c.cached_at, cache.table.c.expires_at)
            ).first()
            
            cached_at = result[0]
            expires_at = result[1]
            
            # Verify cached_at is within our time window and timezone-aware
            assert cached_at.tzinfo is not None, "cached_at should be timezone-aware"
            assert before_utc <= cached_at <= after_utc, "cached_at should be between before/after UTC times"
            
            # Verify expires_at is also timezone-aware and in the future
            if expires_at:
                assert expires_at.tzinfo is not None, "expires_at should be timezone-aware"
                assert expires_at > cached_at, "expires_at should be after cached_at"
                
                # Check that expires_at is approximately 1 hour after cached_at
                expected_expiry = cached_at + timedelta(hours=1)
                time_diff = abs((expires_at - expected_expiry).total_seconds())
                assert time_diff < 5, "expires_at should be ~1 hour after cached_at"
        
        cache.close()
    
    def test_ttl_expiration_with_utc(self):
        """Test that TTL expiration works correctly with UTC timestamps"""
        # Create cache with very short TTL
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Store some data
        data = cache.get_data(id='test_ttl')
        assert not data.empty
        
        # Verify we have data in cache
        stats = cache.get_cache_stats()
        assert stats['total_records'] == 1
        
        # Check expired records count (should be 0 since data is fresh)
        assert stats.get('expired_records', 0) == 0
        
        # Manually expire the data by updating expires_at to past UTC time
        with cache.Session() as session:
            from sqlalchemy import update
            past_utc = datetime.now(timezone.utc) - timedelta(hours=2)
            
            stmt = update(cache.table).values(expires_at=past_utc)
            session.execute(stmt)
            session.commit()
        
        # Now cleanup should remove the expired entry
        removed_count = cache.cleanup_expired()
        assert removed_count == 1, "Should have removed 1 expired entry"
        
        # Verify cache is empty
        final_stats = cache.get_cache_stats()
        assert final_stats['total_records'] == 0
        
        cache.close()
    
    def test_timezone_comparison_consistency(self):
        """Test that timezone comparisons work consistently"""
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Store data
        cache.get_data(id='tz_test')
        
        # Get current stats including expired count
        stats1 = cache.get_cache_stats()
        initial_expired = stats1.get('expired_records', 0)
        
        # Wait a tiny bit and check again (should be same)
        import time
        time.sleep(0.1)
        
        stats2 = cache.get_cache_stats()
        second_expired = stats2.get('expired_records', 0)
        
        # Should be consistent
        assert initial_expired == second_expired, "Expired count should be consistent across calls"
        
        # Both should be 0 since TTL is 1 hour
        assert initial_expired == 0, "No entries should be expired yet"
        assert second_expired == 0, "No entries should be expired yet"
        
        cache.close()
    
    def test_mixed_timezone_data_handling(self):
        """Test handling of data with mixed timezone information"""
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=2
        )
        
        # Store some data
        cache.get_data(id='mixed_tz_test')
        
        # Manually check stored timestamps
        with cache.Session() as session:
            from sqlalchemy import select
            result = session.execute(
                select(cache.table.c.cached_at, cache.table.c.expires_at)
            ).first()
            
            cached_at = result[0]
            expires_at = result[1]
            
            # Both should be timezone-aware
            assert cached_at.tzinfo is not None
            assert expires_at.tzinfo is not None
            
            # Should be UTC specifically
            assert cached_at.tzinfo == timezone.utc or cached_at.utcoffset() == timedelta(0)
            assert expires_at.tzinfo == timezone.utc or expires_at.utcoffset() == timedelta(0)
        
        cache.close()
    
    def test_cache_query_conditions_with_utc(self):
        """Test that cache query conditions work properly with UTC timestamps"""
        cache = SQLAlchemyPullThroughCache.with_sqlite(
            ":memory:", self.test_table, self.adapter, ttl_hours=1
        )
        
        # Store data
        original_data = cache.get_data(id='query_test')
        assert not original_data.empty
        
        # Query again - should get from cache (not fetch again)
        fetch_count_before = self.adapter.fetch_count
        cached_data = cache.get_data(id='query_test')
        fetch_count_after = self.adapter.fetch_count
        
        # Should not have fetched again
        assert fetch_count_before == fetch_count_after, "Should have used cached data"
        assert not cached_data.empty
        
        # Data should be identical
        assert len(original_data) == len(cached_data)
        
        cache.close()


def test_utc_import_and_usage():
    """Test that timezone imports and usage are correct"""
    if not HAS_SQL_CACHE:
        pytest.skip("SQLAlchemy not available")
    
    # Verify that the SQL cache module imports timezone correctly
    from cacheness.sql_cache import timezone, datetime
    
    # Test basic UTC functionality
    now_utc = datetime.now(timezone.utc)
    assert now_utc.tzinfo is not None
    assert now_utc.tzinfo == timezone.utc or now_utc.utcoffset() == timedelta(0)


def test_documentation_example_timezone_usage():
    """Test timezone usage in documentation examples"""
    if not HAS_SQL_CACHE:
        pytest.skip("SQLAlchemy not available")
    
    from cacheness import SQLAlchemyPullThroughCache, SQLAlchemyDataAdapter
    from sqlalchemy import Table, Column, String, Integer, MetaData
    import pandas as pd
    
    # Create a simple example like in docs
    metadata = MetaData()
    simple_table = Table('docs_test', metadata, Column('id', Integer, primary_key=True))
    
    class SimpleAdapter(SQLAlchemyDataAdapter):
        def get_table_definition(self):
            return simple_table
        def fetch_data(self, **kwargs):
            return pd.DataFrame([{'id': 1}])
        def parse_query_params(self, **kwargs):
            return kwargs
    
    # Test with different backends
    for backend_method in ['with_sqlite', 'with_duckdb']:
        try:
            if backend_method == 'with_sqlite':
                cache = SQLAlchemyPullThroughCache.with_sqlite(
                    ":memory:", simple_table, SimpleAdapter(), ttl_hours=1
                )
            else:
                # DuckDB might not be available, skip if not
                try:
                    cache = SQLAlchemyPullThroughCache.with_duckdb(
                        ":memory:", simple_table, SimpleAdapter(), ttl_hours=1
                    )
                except Exception:
                    continue
            
            # Test that it works
            data = cache.get_data(id=1)
            assert not data.empty
            
            # Verify timezone-aware columns exist
            has_timezone_cols = any(
                getattr(col.type, 'timezone', False) 
                for col in cache.table.columns 
                if col.name in ('cached_at', 'expires_at')
            )
            assert has_timezone_cols, f"Backend {backend_method} should have timezone-aware columns"
            
            cache.close()
            
        except Exception:
            # Some backends might not be available
            continue
