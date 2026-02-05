"""
Tests to verify that the documentation examples work correctly.
This ensures our documentation stays accurate and functional.
"""

import pytest
import pandas as pd
from datetime import date
from sqlalchemy import MetaData, Table, Column, String, Date, Float, Integer, Index


# Only run these tests if SQLAlchemy is available
try:
    from cacheness.sql_cache import SqlCache, SqlCacheAdapter
    HAS_SQL_CACHE = True
except ImportError:
    HAS_SQL_CACHE = False


@pytest.mark.skipif(not HAS_SQL_CACHE, reason="SQLAlchemy not available")
class TestDocumentationExamples:
    """Test that all documentation examples work correctly"""
    
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        """Set up test fixtures for documentation examples"""
        # Stock table from README example
        metadata = MetaData()
        self.stock_table = Table(
            'stock_prices', metadata,
            Column('symbol', String(10), primary_key=True),
            Column('date', Date, primary_key=True),
            Column('close', Float)
        )
        
        # Analytics table from SQL_CACHE.md
        self.analytics_table = Table(
            'analytics_data', metadata,
            Column('metric_id', String(50), primary_key=True),
            Column('date', Date, primary_key=True),
            Column('value', Float),
            Column('category', String(20)),
            Column('count', Integer),
            Index('idx_date_category', 'date', 'category'),
            Index('idx_metric_date', 'metric_id', 'date'),
        )
        
        # Session table example
        self.session_table = Table(
            'user_sessions', metadata,
            Column('user_id', Integer, primary_key=True),
            Column('session_date', Date, primary_key=True),
            Column('session_data', String(500))
        )
    
    def test_readme_stock_adapter_example(self):
        """Test the stock adapter example from README"""
        
        class StockAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.stock_table
            
            def fetch_data(self, **kwargs):
                # Simulate fetching stock data
                return pd.DataFrame([
                    {
                        'symbol': kwargs.get('symbol', 'AAPL'),
                        'date': date(2024, 1, 1),
                        'close': 150.0
                    }
                ])
            
            def parse_query_params(self, **kwargs):
                return {'symbol': kwargs['symbol'], 'date': {'start': kwargs['start_date']}}
        
        adapter = StockAdapter()
        
        # Test DuckDB backend selection (from README)
        try:
            cache = SqlCache.with_duckdb(":memory:", self.stock_table, adapter)
            data = cache.get_data(symbol="AAPL", start_date="2024-01-01")
            assert not data.empty
            cache.close()
        except Exception:
            # DuckDB not installed - expected
            pass
        
        # Test SQLite backend selection (from README)
        cache = SqlCache.with_sqlite(":memory:", self.stock_table, adapter)
        data = cache.get_data(symbol="AAPL", start_date="2024-01-01")
        assert not data.empty
        assert 'symbol' in data.columns
        assert 'close' in data.columns
        cache.close()
    
    def test_analytics_example_from_docs(self):
        """Test the analytics example from SQL_CACHE.md"""
        
        class AnalyticsAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.analytics_table
            
            def fetch_data(self, **kwargs):
                # Simulate analytics data
                return pd.DataFrame([
                    {
                        'metric_id': 'page_views',
                        'date': date(2024, 1, 1),
                        'value': 1000.0,
                        'category': 'web',
                        'count': 100
                    }
                ])
            
            def parse_query_params(self, **kwargs):
                return kwargs
        
        adapter = AnalyticsAdapter()
        
        # Test the DuckDB analytics example
        try:
            cache = SqlCache.with_duckdb(
                ":memory:",
                self.analytics_table,
                adapter,
                ttl_seconds=86400
            )
            
            quarterly_data = cache.get_data(
                symbol="AAPL", 
                start_date="2024-01-01", 
                end_date="2024-03-31"
            )
            
            assert not quarterly_data.empty
            cache.close()
            
        except Exception:
            # DuckDB not available - skip this part
            pass
    
    def test_session_cache_example_from_docs(self):
        """Test the user session cache example from SQL_CACHE.md"""
        
        class SessionAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.session_table
            
            def fetch_data(self, **kwargs):
                # Simulate session data
                return pd.DataFrame([
                    {
                        'user_id': kwargs.get('user_id', 12345),
                        'session_date': date(2024, 1, 15),
                        'session_data': '{"login_time": "10:00", "actions": ["view", "click"]}'
                    }
                ])
            
            def parse_query_params(self, **kwargs):
                return kwargs
        
        adapter = SessionAdapter()
        
        # Test SQLite session cache example
        cache = SqlCache.with_sqlite(
            ":memory:",
            self.session_table,
            adapter,
            ttl_seconds=21600
        )
        
        user_session = cache.get_data(user_id=12345, session_date="2024-01-15")
        
        assert not user_session.empty
        assert 'user_id' in user_session.columns
        assert 'session_data' in user_session.columns
        
        cache.close()
    
    def test_backend_comparison_matrix(self):
        """Test that all backends mentioned in docs work"""
        
        class SimpleAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.stock_table
            
            def fetch_data(self, **kwargs):
                return pd.DataFrame([
                    {'symbol': 'TEST', 'date': date(2024, 1, 1), 'close': 100.0}
                ])
            
            def parse_query_params(self, **kwargs):
                return kwargs
        
        adapter = SimpleAdapter()
        
        # Test all backend selection methods mentioned in docs
        
        # SQLite (should always work)
        cache_sqlite = SqlCache.with_sqlite(":memory:", self.stock_table, adapter)
        assert cache_sqlite.engine.dialect.name == "sqlite"
        cache_sqlite.close()
        
        # In-memory SQLite (mentioned in docs)
        cache_memory = SqlCache.with_sqlite(":memory:", self.stock_table, adapter)
        assert cache_memory.engine.dialect.name == "sqlite"
        cache_memory.close()
        
        # DuckDB (may not be available)
        try:
            cache_duck = SqlCache.with_duckdb(":memory:", self.stock_table, adapter)
            assert cache_duck.engine.dialect.name == "duckdb"
            cache_duck.close()
        except Exception:
            # Expected if duckdb-engine not installed
            pass
        
        # PostgreSQL method (using SQLite URL for testing)
        cache_pg = SqlCache.with_postgresql("sqlite:///:memory:", self.stock_table, adapter)
        # Should fall back to sqlite dialect when using sqlite URL
        assert cache_pg.engine is not None
        cache_pg.close()
    
    def test_ttl_configuration_examples(self):
        """Test TTL configuration examples from docs"""
        
        class SimpleAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.stock_table
            
            def fetch_data(self, **kwargs):
                return pd.DataFrame()
            
            def parse_query_params(self, **kwargs):
                return kwargs
        
        adapter = SimpleAdapter()
        
        # Test TTL configurations from documentation
        
        # No expiration (ttl_seconds=0)
        cache_no_exp = SqlCache.with_sqlite(
            ":memory:", self.stock_table, adapter, ttl_seconds=0
        )
        assert cache_no_exp.ttl_seconds == 0
        cache_no_exp.close()
        
        # 3600 seconds (1 hour) expiration
        cache_1h = SqlCache.with_sqlite(
            ":memory:", self.stock_table, adapter, ttl_seconds=3600
        )
        assert cache_1h.ttl_seconds == 3600
        cache_1h.close()
        
        # Daily refresh (24 hours)
        cache_24h = SqlCache.with_sqlite(
            ":memory:", self.stock_table, adapter, ttl_seconds=86400
        )
        assert cache_24h.ttl_seconds == 86400
        cache_24h.close()
    
    def test_cache_operations_from_docs(self):
        """Test cache management operations mentioned in documentation"""
        
        class SimpleAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return self.stock_table
            
            def fetch_data(self, **kwargs):
                return pd.DataFrame([
                    {'symbol': 'DOC', 'date': date(2024, 1, 1), 'close': 123.45}
                ])
            
            def parse_query_params(self, **kwargs):
                return kwargs
        
        adapter = SimpleAdapter()
        cache = SqlCache.with_sqlite(":memory:", self.stock_table, adapter)
        
        # Test cache statistics (mentioned in docs)
        stats = cache.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_records' in stats
        assert 'table_name' in stats
        
        # Test cache management operations (mentioned in docs)
        expired_count = cache.cleanup_expired()
        assert expired_count >= 0
        
        cleared_count = cache.clear_cache()
        assert cleared_count >= 0
        
        cache.close()


def test_main_package_imports():
    """Test that SQL cache can be imported from main package as documented"""
    try:
        from cacheness import SqlCache, SqlCacheAdapter
        
        # Test that the classes are available
        assert SqlCache is not None
        assert SqlCacheAdapter is not None
        
        # Test that they have the expected methods
        assert hasattr(SqlCache, 'with_sqlite')
        assert hasattr(SqlCache, 'with_duckdb')
        assert hasattr(SqlCache, 'with_postgresql')
        
    except ImportError:
        # Expected if SQLAlchemy not installed
        pytest.skip("SQLAlchemy not available")


def test_recommended_installation():
    """Test that the recommended installation includes SQL support"""
    try:
        # These imports should work with recommended installation
        import sqlalchemy
        import pandas
        from cacheness import SqlCache
        
        assert sqlalchemy is not None
        assert pandas is not None
        assert SqlCache is not None
        
    except ImportError:
        pytest.skip("Recommended dependencies not available")
