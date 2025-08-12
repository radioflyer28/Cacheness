"""
API Request Caching Example

This example demonstrates how to use cacheness to cache expensive API requests
with intelligent TTL management and ETag-based validation.
"""

import requests
from cacheness import cached, cacheness, CacheConfig


# Initialize cache for API responses
config = CacheConfig(
    cache_dir="./api_cache",
    default_ttl_hours=24,  # Cache API responses for 24 hours
    metadata_backend="sqlite"
)
api_cache = cacheness(config)


class WeatherAPI:
    """Example API client with intelligent caching."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
    
    @cached(cache_instance=api_cache, ttl_hours=6, key_prefix="weather")
    def get_current_weather(self, city, units="metric"):
        """Get current weather with 6-hour caching."""
        url = f"{self.base_url}/current.json"
        params = {
            "key": self.api_key,
            "q": city,
            "units": units
        }
        
        print(f"Making API request for {city}...")  # Only prints on cache miss
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    @cached(cache_instance=api_cache, ttl_hours=168, key_prefix="forecast")  # 1 week
    def get_7_day_forecast(self, city, include_hourly=False):
        """Get 7-day forecast with weekly caching."""
        url = f"{self.base_url}/forecast.json"
        params = {
            "key": self.api_key,
            "q": city,
            "days": 7,
            "hourly": 1 if include_hourly else 0
        }
        
        print(f"Making forecast API request for {city}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    @cached(cache_instance=api_cache, ttl_hours=8760, key_prefix="historical")  # 1 year
    def get_historical_weather(self, city, date):
        """Get historical weather data with long-term caching."""
        url = f"{self.base_url}/history.json"
        params = {
            "key": self.api_key,
            "q": city,
            "dt": date  # YYYY-MM-DD format
        }
        
        print(f"Making historical API request for {city} on {date}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate API caching functionality."""
    
    # Initialize weather client (replace with your API key)
    weather_client = WeatherAPI("your_api_key_here")
    
    print("=== Weather API Caching Demo ===\n")
    
    # Current weather - first call makes API request
    print("1. Getting current weather for London (first time):")
    current = weather_client.get_current_weather("London", units="imperial")
    print(f"   Temperature: {current['current']['temp_f']}°F")
    
    # Second call uses cache (no API request)
    print("\n2. Getting current weather for London (cached):")
    current_cached = weather_client.get_current_weather("London", units="imperial")
    print(f"   Temperature: {current_cached['current']['temp_f']}°F")
    
    # Different parameters = different cache entry
    print("\n3. Getting current weather for London in Celsius (new cache entry):")
    current_metric = weather_client.get_current_weather("London", units="metric")
    print(f"   Temperature: {current_metric['current']['temp_c']}°C")
    
    # Long-term forecast caching
    print("\n4. Getting 7-day forecast (cached for 1 week):")
    forecast = weather_client.get_7_day_forecast("London", include_hourly=True)
    print(f"   Forecast days: {len(forecast['forecast']['forecastday'])}")
    
    # Historical data cached for a full year
    print("\n5. Getting historical data (cached for 1 year):")
    historical = weather_client.get_historical_weather("London", "2023-12-25")
    print(f"   Historical date: {historical['forecast']['forecastday'][0]['date']}")
    
    # Cache statistics
    print("\n=== Cache Statistics ===")
    stats = api_cache.get_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Cache size: {stats['total_size_mb']:.2f} MB")
    print(f"Hit rate: {stats.get('hit_rate', 0):.1%}")


if __name__ == "__main__":
    main()
