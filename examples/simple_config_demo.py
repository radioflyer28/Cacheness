#!/usr/bin/env python3
"""
Simple Configuration Example
============================

Shows basic configuration options without overwhelming technical details.
Much simpler than the complex serialization demo!

Usage:
    python simple_config_demo.py
"""

from cacheness import cached
import pandas as pd
from datetime import datetime

# ===== Basic Configuration Examples =====

# Example 1: Custom cache directory
@cached.for_api(ttl_hours=6)
def fetch_weather_default(city):
    """Uses default cache location."""
    print(f"🌤️  Fetching weather for {city} (default cache)")
    return {"city": city, "temp": 72, "conditions": "sunny"}

# Example 2: Different cache directories for different purposes
@cached.for_api(ttl_hours=4, cache_dir="./custom_weather_cache")
def fetch_weather_custom(city):
    """Uses custom cache location."""
    print(f"🌤️  Fetching weather for {city} (custom cache)")
    return {"city": city, "temp": 68, "conditions": "cloudy"}

# Example 3: Different TTL for different data types
@cached(ttl_hours=2)  # Short TTL for user data (returns dict)
def get_user_status(user_id):
    """Get user status - changes frequently."""
    print(f"👤 Checking status for user {user_id}")
    return {"user_id": user_id, "status": "online", "last_seen": str(datetime.now())}

@cached(ttl_hours=48)  # Long TTL for reports (returns DataFrame)
def generate_monthly_report(month, year):
    """Generate monthly report - expensive, changes rarely."""
    print(f"📊 Generating report for {month}/{year}")
    return pd.DataFrame({
        "month": [month],
        "year": [year], 
        "sales": [150000],
        "users": [1250]
    })

def main():
    """Demonstrate simple configuration options."""
    
    print("=== Simple Configuration Demo ===\n")
    
    # Default caching
    print("🔧 DEFAULT CACHING:")
    weather1 = fetch_weather_default("Seattle")
    print(f"✅ Weather: {weather1['temp']}°F, {weather1['conditions']}")
    
    weather2 = fetch_weather_default("Seattle")  # Cached
    print(f"✅ Cached weather: {weather2['temp']}°F\n")
    
    # Custom cache directory
    print("🔧 CUSTOM CACHE DIRECTORY:")
    weather3 = fetch_weather_custom("Portland") 
    print(f"✅ Custom weather: {weather3['temp']}°F, {weather3['conditions']}")
    
    weather4 = fetch_weather_custom("Portland")  # Cached
    print(f"✅ Cached custom: {weather4['temp']}°F\n")
    
    # Different TTL strategies
    print("🔧 TTL STRATEGIES:")
    
    # Short TTL for frequently changing data
    status1 = get_user_status(123)
    print(f"✅ User status: {status1['status']} (TTL: 2 hours)")
    
    # Long TTL for expensive, stable data
    report1 = generate_monthly_report("March", 2024)
    print(f"✅ Monthly report: ${report1.iloc[0]['sales']:,} sales (TTL: 48 hours)")
    
    print("\n🎯 Configuration Tips:")
    print("   • Use @cached() for any Python object (auto-optimized)")
    print("   • Use @cached.for_api() for external API calls")
    print("   • Adjust TTL based on how often data changes")
    print("   • UnifiedCache auto-detects optimal storage format")
    print("   • DataFrames → Parquet, NumPy → Blosc, Objects → Pickle")

if __name__ == "__main__":
    main()