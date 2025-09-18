#!/usr/bin/env python3
"""
Simple API Caching Example
==========================

Shows how to cache API responses with minimal boilerplate.
Perfect for new users to understand core concepts.

Usage:
    python simple_api_caching.py
"""

import requests
from cacheness import cached

# Method 1: API-optimized decorator (easiest)
@cached.for_api(ttl_hours=6)
def get_weather(city):
    """Get weather with automatic caching."""
    print(f"Fetching weather for {city}...")  # Only shows on cache miss
    response = requests.get(f"https://api.weather.gov/weather/{city}")
    return response.json()

@cached.for_api(ttl_hours=24) 
def get_news(category="technology"):
    """Get news with daily caching."""
    print(f"Fetching {category} news...")
    response = requests.get(f"https://api.news.com/{category}")
    return response.json()

# Method 2: Regular decorator (also works)
@cached(ttl_hours=4)
def get_stock_price(symbol):
    """Get stock price with standard caching."""
    print(f"Fetching stock price for {symbol}...")
    response = requests.get(f"https://api.stocks.com/{symbol}")
    return response.json()

def main():
    """Demonstrate simple API caching."""
    
    print("=== Simple API Caching Demo ===\n")
    
    # First call - makes API request
    weather1 = get_weather("seattle")
    print(f"✅ Weather: {weather1.get('temperature', 'N/A')}")
    
    # Second call - uses cache (no API request)
    weather2 = get_weather("seattle") 
    print(f"✅ Cached weather: {weather2.get('temperature', 'N/A')}")
    
    # Different parameter - new cache entry
    weather3 = get_weather("portland")
    print(f"✅ Portland weather: {weather3.get('temperature', 'N/A')}\n")
    
    # Using standard decorator
    stock1 = get_stock_price("AAPL")
    print(f"✅ AAPL price: ${stock1.get('price', 'N/A')}")
    
    stock2 = get_stock_price("AAPL")  # Cached
    print(f"✅ AAPL cached: ${stock2.get('price', 'N/A')}")

if __name__ == "__main__":
    main()