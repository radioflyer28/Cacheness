#!/usr/bin/env python3
"""
Simple Database Backend Example
===============================

Shows when to use SQLite vs DuckDB without overwhelming details.
Much simpler than the complex backend comparison!

Usage:
    python simple_backend_demo.py
"""

from cacheness.sql_cache import SqlCache
from sqlalchemy import Integer, String, Float
import pandas as pd

# Simulate data fetchers
def fetch_user_lookup(user_id):
    """Fetch individual user - perfect for SQLite."""
    print(f"👤 Looking up user {user_id}")
    return pd.DataFrame([{
        'user_id': user_id,
        'name': f'User {user_id}',
        'email': f'user{user_id}@example.com',
        'department': 'Engineering'
    }])

def fetch_analytics_data(department):
    """Fetch bulk analytics data - perfect for DuckDB."""
    print(f"📊 Fetching analytics for {department}")
    # Simulate large dataset
    return pd.DataFrame([
        {
            'department': department,
            'employee_id': i,
            'performance_score': 85.0 + (i % 20), 
            'salary': 50000.0 + (i * 1000)
        }
        for i in range(100)  # Simulated bulk data
    ])

def main():
    """Demonstrate when to use different backends."""
    
    print("=== Simple Database Backend Demo ===\n")
    
    # SQLite Example - Individual Lookups
    print("🗃️  SQLITE: Individual User Lookups")
    print("   Best for: User profiles, product details, session data")
    
    # Using the lookup table builder (automatically uses SQLite)
    user_cache = SqlCache.for_lookup_table(
        "users_sqlite.db",  # Will use SQLite
        primary_keys=["user_id"],
        data_fetcher=fetch_user_lookup,
        user_id=Integer,
        name=String(100),
        email=String(200),
        department=String(50)
    )
    
    # Individual user lookups - SQLite excels here
    user1 = user_cache.get_data(user_id=123)
    print(f"✅ SQLite lookup: {user1.iloc[0]['name']}")
    
    user2 = user_cache.get_data(user_id=456)
    print(f"✅ SQLite lookup: {user2.iloc[0]['name']}\n")
    
    # DuckDB Example - Bulk Analytics
    print("🦆 DUCKDB: Bulk Analytics")
    print("   Best for: Reports, aggregations, data science")
    
    # Using the analytics table builder (automatically uses DuckDB)
    analytics_cache = SqlCache.for_analytics_table(
        "analytics_duckdb.db",  # Will use DuckDB
        primary_keys=["department", "employee_id"],
        data_fetcher=fetch_analytics_data,
        department=String(50),
        employee_id=Integer,
        performance_score=Float,
        salary=Float
    )
    
    # Bulk analytics - DuckDB excels here
    analytics1 = analytics_cache.get_data(department="Engineering")
    print(f"✅ DuckDB analytics: {len(analytics1)} employee records")
    
    if not analytics1.empty:
        avg_salary = analytics1['salary'].mean()
        avg_performance = analytics1['performance_score'].mean()
        print(f"   Average salary: ${avg_salary:,.0f}")
        print(f"   Average performance: {avg_performance:.1f}")
    
    print("\n🎯 When to Use Each:")
    print("   SQLite (for_lookup_table, for_realtime_timeseries):")
    print("     • Individual record lookups")
    print("     • Real-time data updates")
    print("     • Transaction-heavy workloads")
    print()
    print("   DuckDB (for_analytics_table, for_timeseries):")
    print("     • Bulk data processing")
    print("     • Analytics and reporting")
    print("     • Historical analysis")
    print()
    print("💡 The builders automatically choose the right backend!")

if __name__ == "__main__":
    main()