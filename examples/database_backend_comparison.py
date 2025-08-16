"""
Database Backend Comparison Example

This example demonstrates how to choose between different database backends
for the SQL pull-through cache based on your use case requirements.

Key Differences:
- DuckDB: Optimized for analytical/columnar workloads (aggregations, time-series)
- SQLite: Optimized for transactional/row-wise operations (ACID, concurrent reads)
- PostgreSQL: Full-featured production RDBMS (high concurrency, advanced features)

Usage patterns:
- DuckDB: Time-series analysis, data science workflows, large dataset aggregations
- SQLite: Simple deployment, row-wise operations, moderate concurrency
- PostgreSQL: Production environments, high concurrency, complex queries
"""

import pandas as pd
from datetime import datetime, date
from sqlalchemy import MetaData, Table, Column, String, Date, Float, Integer, Index

# Import the SQL cache components
try:
    from cacheness.sql_cache import SqlCache, SqlCacheAdapter
except ImportError:
    print("Error: cacheness SQL cache not available")
    print("Make sure SQLAlchemy is installed: pip install 'cacheness[sql]'")
    exit(1)


# Define a sample table schema
metadata = MetaData()

analytics_table = Table(
    'analytics_data',
    metadata,
    Column('metric_id', String(50), primary_key=True),
    Column('date', Date, primary_key=True),
    Column('value', Float),
    Column('category', String(20)),
    Column('count', Integer),
    
    # Indexes for different query patterns
    Index('idx_date_category', 'date', 'category'),  # Time-series queries
    Index('idx_metric_date', 'metric_id', 'date'),   # Metric tracking
)


class AnalyticsSqlCacheAdapter(SqlCacheAdapter):
    """Sample data adapter that generates synthetic analytics data"""
    
    def get_table_definition(self):
        return analytics_table
    
    def parse_query_params(self, **kwargs):
        return {
            'metric_id': kwargs.get('metric_id'),
            'date': {
                'start': kwargs.get('start_date'),
                'end': kwargs.get('end_date')
            },
            'category': kwargs.get('category')
        }
    
    def fetch_data(self, **kwargs):
        """Generate synthetic data for demonstration"""
        import numpy as np
        
        metric_id = kwargs.get('metric_id', 'page_views')
        start_date = pd.to_datetime(kwargs.get('start_date', '2024-01-01'))
        end_date = pd.to_datetime(kwargs.get('end_date', '2024-01-31'))
        category = kwargs.get('category', 'web')
        
        print(f"Fetching {metric_id} data from {start_date.date()} to {end_date.date()}")
        
        # Generate synthetic time series data
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)  # For reproducible results
        
        data = []
        for i, date_val in enumerate(dates):
            # Simulate different patterns
            base_value = 1000 + i * 10  # Growing trend
            noise = np.random.normal(0, 100)  # Random variation
            weekend_factor = 0.7 if date_val.weekday() >= 5 else 1.0  # Weekend dip
            
            data.append({
                'metric_id': metric_id,
                'date': date_val.date(),
                'value': max(0, base_value * weekend_factor + noise),
                'category': category,
                'count': np.random.randint(50, 200)
            })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records")
        return df


def demonstrate_duckdb_backend():
    """
    Demonstrate DuckDB backend - ideal for analytical workloads
    """
    print("\n" + "="*60)
    print("DuckDB Backend - Analytical/Columnar Workloads")
    print("="*60)
    
    # Create cache with DuckDB backend (optimized for analytics)
    cache = SqlCache.with_duckdb(
        db_path="analytics_duckdb.db",
        table=analytics_table,
        data_adapter=AnalyticsSqlCacheAdapter(),
        ttl_hours=24
    )
    
    print(f"✓ Created cache with backend: {cache.engine.dialect.name}")
    
    # Fetch some data
    start_time = datetime.now()
    data = cache.get_data(
        metric_id="page_views",
        start_date="2024-01-01",
        end_date="2024-01-31",
        category="web"
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✓ Retrieved {len(data)} records in {elapsed:.3f}s")
    
    # Demonstrate analytical queries (this would be even faster with DuckDB)
    if not data.empty:
        print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"✓ Average value: {data['value'].mean():.2f}")
        print(f"✓ Total count: {data['count'].sum()}")
    
    # Show cache stats
    stats = cache.get_cache_stats()
    print(f"✓ Cache contains {stats['total_records']} records")
    
    cache.close()
    print("✓ DuckDB demo completed")


def demonstrate_sqlite_backend():
    """
    Demonstrate SQLite backend - ideal for transactional workloads
    """
    print("\n" + "="*60)
    print("SQLite Backend - Transactional/Row-wise Operations")
    print("="*60)
    
    # Create cache with SQLite backend (optimized for transactions)
    cache = SqlCache.with_sqlite(
        db_path="analytics_sqlite.db",
        table=analytics_table,
        data_adapter=AnalyticsSqlCacheAdapter(),
        ttl_hours=24
    )
    
    print(f"✓ Created cache with backend: {cache.engine.dialect.name}")
    
    # Fetch data
    start_time = datetime.now()
    data = cache.get_data(
        metric_id="user_sessions",
        start_date="2024-01-15",
        end_date="2024-01-20",
        category="mobile"
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✓ Retrieved {len(data)} records in {elapsed:.3f}s")
    
    # Demonstrate row-wise operations
    if not data.empty:
        print(f"✓ Row-by-row processing example:")
        for _, row in data.head(3).iterrows():
            print(f"  - {row['date']}: {row['value']:.1f} ({row['category']})")
    
    # Show cache management capabilities
    stats = cache.get_cache_stats()
    print(f"✓ Cache contains {stats['total_records']} records")
    
    # Demonstrate invalidation (transactional operation)
    invalidated = cache.invalidate_cache(metric_id="user_sessions")
    print(f"✓ Invalidated {invalidated} records (transactional operation)")
    
    cache.close()
    print("✓ SQLite demo completed")


def demonstrate_in_memory_sqlite():
    """
    Demonstrate in-memory SQLite for testing/temporary use
    """
    print("\n" + "="*60)
    print("In-Memory SQLite - Testing & Temporary Data")
    print("="*60)
    
    # Create in-memory cache (perfect for testing)
    cache = SqlCache.with_sqlite(
        db_path=":memory:",  # In-memory database
        table=analytics_table,
        data_adapter=AnalyticsSqlCacheAdapter(),
        ttl_hours=1  # Short TTL for temporary data
    )
    
    print(f"✓ Created in-memory cache with backend: {cache.engine.dialect.name}")
    
    # Fetch some test data
    data = cache.get_data(
        metric_id="test_metric",
        start_date="2024-01-01",
        end_date="2024-01-03",
        category="test"
    )
    
    print(f"✓ Retrieved {len(data)} test records")
    print("✓ Data stored in memory - will be lost when process ends")
    
    stats = cache.get_cache_stats()
    print(f"✓ Memory cache contains {stats['total_records']} records")
    
    cache.close()
    print("✓ In-memory demo completed")


def compare_backends():
    """
    Compare performance characteristics of different backends
    """
    print("\n" + "="*60)
    print("Backend Performance Comparison")
    print("="*60)
    
    backends = [
        ("DuckDB", lambda: SqlCache.with_duckdb(
            "perf_test_duckdb.db", analytics_table, AnalyticsSqlCacheAdapter()
        )),
        ("SQLite", lambda: SqlCache.with_sqlite(
            "perf_test_sqlite.db", analytics_table, AnalyticsSqlCacheAdapter()
        )),
        ("In-Memory SQLite", lambda: SqlCache.with_sqlite(
            ":memory:", analytics_table, AnalyticsSqlCacheAdapter()
        ))
    ]
    
    for backend_name, cache_factory in backends:
        print(f"\n{backend_name}:")
        cache = cache_factory()
        
        # Time data fetching
        start_time = datetime.now()
        data = cache.get_data(
            metric_id=f"perf_test_{backend_name.lower().replace(' ', '_')}",
            start_date="2024-01-01",
            end_date="2024-01-07",
            category="performance"
        )
        fetch_time = (datetime.now() - start_time).total_seconds()
        
        # Time cache operations
        start_time = datetime.now()
        stats = cache.get_cache_stats()
        stats_time = (datetime.now() - start_time).total_seconds()
        
        print(f"  - Data fetch: {fetch_time:.3f}s ({len(data)} records)")
        print(f"  - Stats query: {stats_time:.3f}s")
        print(f"  - Best for: {get_backend_use_case(backend_name)}")
        
        cache.close()


def get_backend_use_case(backend_name):
    """Return the ideal use case for each backend"""
    use_cases = {
        "DuckDB": "Time-series analysis, aggregations, data science workflows",
        "SQLite": "Transactional operations, moderate concurrency, simple deployment",
        "In-Memory SQLite": "Testing, temporary data, development workflows"
    }
    return use_cases.get(backend_name, "General purpose")


def main():
    """Run all demonstrations"""
    print("SQL Pull-Through Cache - Database Backend Comparison")
    print("This demo shows how to choose the right database backend for your use case")
    
    try:
        # Demonstrate each backend
        demonstrate_duckdb_backend()
        demonstrate_sqlite_backend()
        demonstrate_in_memory_sqlite()
        compare_backends()
        
        print("\n" + "="*60)
        print("RECOMMENDATION GUIDE")
        print("="*60)
        print("🔹 Use DuckDB when:")
        print("  - Analyzing time-series data")
        print("  - Running aggregation queries")
        print("  - Processing large datasets")
        print("  - Doing data science work")
        print()
        print("🔹 Use SQLite when:")
        print("  - Need ACID transactions")
        print("  - Moderate concurrent access")
        print("  - Simple deployment requirements")
        print("  - Row-wise data operations")
        print()
        print("🔹 Use PostgreSQL when:")
        print("  - Production environments")
        print("  - High concurrency needs")
        print("  - Advanced SQL features required")
        print("  - Need horizontal scaling")
        print()
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
