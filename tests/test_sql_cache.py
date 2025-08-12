"""
Basic tests for SQL pull-through cache functionality.
"""

import pytest
import pandas as pd
from datetime import date
from sqlalchemy import MetaData, Table, Column, String, Date, Float, Integer


# Only run these tests if SQLAlchemy is available
try:
    from cacheness.sql_cache import SQLAlchemyPullThroughCache, SQLAlchemyDataAdapter
    HAS_SQL_CACHE = True
except ImportError:
    HAS_SQL_CACHE = False


@pytest.mark.skipif(not HAS_SQL_CACHE, reason="SQLAlchemy not available")
class TestSQLCache:
    """Test cases for SQL pull-through cache"""
    
    def setup_method(self):
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
        class TestDataAdapter(SQLAlchemyDataAdapter):
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
        
        self.adapter = TestDataAdapter(self.test_table)
    
    def test_cache_initialization(self):
        """Test that cache can be initialized"""
        cache = SQLAlchemyPullThroughCache(
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
        
        cache = SQLAlchemyPullThroughCache(
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
        cache = SQLAlchemyPullThroughCache(
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
        cache = SQLAlchemyPullThroughCache(
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


def test_import_availability():
    """Test that imports work correctly"""
    try:
        from cacheness import SQLAlchemyPullThroughCache, SQLAlchemyDataAdapter
        assert SQLAlchemyPullThroughCache is not None
        assert SQLAlchemyDataAdapter is not None
    except ImportError:
        # This is expected if SQLAlchemy is not installed
        pass
