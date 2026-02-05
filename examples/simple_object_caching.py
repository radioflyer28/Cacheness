#!/usr/bin/env python3
"""
Simple Object Caching Example
=============================

Demonstrates caching complex objects without security warnings.
Much simpler than the dill security example!

Usage:
    python simple_object_caching.py
"""

from cacheness import cached
from dataclasses import dataclass
from typing import List

# Simple data classes for caching
@dataclass
class UserProfile:
    user_id: int
    name: str
    email: str
    preferences: dict

@dataclass 
class ProcessingResult:
    input_data: List[str]
    processed_items: List[str]
    stats: dict

@cached(ttl_seconds=43200)  # 12 hours
def get_user_profile(user_id: int) -> UserProfile:
    """Get user profile (simulated expensive lookup)."""
    print(f"🔍 Loading profile for user {user_id}")
    
    return UserProfile(
        user_id=user_id,
        name=f"User {user_id}",
        email=f"user{user_id}@example.com",
        preferences={
            "theme": "dark",
            "notifications": True,
            "language": "en"
        }
    )

@cached(ttl_seconds=86400)  # 24 hours
def process_data(items: List[str]) -> ProcessingResult:
    """Process a list of items (simulated expensive operation)."""
    print(f"⚙️  Processing {len(items)} items")
    
    # Simulate expensive processing
    processed = [item.upper() + "_PROCESSED" for item in items]
    
    stats = {
        "total_items": len(items),
        "processed_items": len(processed),
        "avg_length": sum(len(item) for item in items) / len(items) if items else 0
    }
    
    return ProcessingResult(
        input_data=items,
        processed_items=processed,
        stats=stats
    )

@cached(ttl_seconds=21600)  # 6 hours
def complex_calculation(data: dict) -> dict:
    """Perform complex calculations on dictionary data."""
    print(f"🧮 Computing complex calculation for {len(data)} keys")
    
    result = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            result[f"{key}_squared"] = value ** 2
            result[f"{key}_sqrt"] = value ** 0.5
        else:
            result[f"{key}_length"] = len(str(value))
    
    return result

def main():
    """Demonstrate simple object caching."""
    
    print("=== Simple Object Caching Demo ===\n")
    
    # Cache user profiles
    print("👤 USER PROFILE CACHING:")
    profile1 = get_user_profile(123)
    print(f"✅ Got profile: {profile1.name} ({profile1.email})")
    
    profile2 = get_user_profile(123)  # Cached
    print(f"✅ Cached profile: {profile2.name}")
    print(f"   Preferences: {profile2.preferences}\n")
    
    # Cache data processing results
    print("⚙️  DATA PROCESSING CACHING:")
    test_data = ["hello", "world", "cache", "example"]
    result1 = process_data(test_data)
    print(f"✅ Processed {result1.stats['total_items']} items")
    print(f"   Sample output: {result1.processed_items[0]}")
    
    result2 = process_data(test_data)  # Cached
    print(f"✅ Cached processing: {result2.stats['total_items']} items\n")
    
    # Cache complex objects
    print("🧮 COMPLEX OBJECT CACHING:")
    calc_input = {
        "temperature": 25.5,
        "pressure": 1013.25, 
        "location": "San Francisco",
        "measurements": 42
    }
    
    calc1 = complex_calculation(calc_input)
    print(f"✅ Calculated {len(calc1)} derived values")
    print(f"   temperature_squared: {calc1.get('temperature_squared', 'N/A')}")
    
    calc2 = complex_calculation(calc_input)  # Cached
    print(f"✅ Cached calculation: {len(calc2)} values")
    
    print("\n🎯 Benefits:")
    print("   • Automatic object serialization")
    print("   • Type safety with dataclasses") 
    print("   • No security warnings (using pickle)")
    print("   • Optimized for general Python objects")

if __name__ == "__main__":
    main()