#!/usr/bin/env python3
"""
Intelligent Storage Demo
=======================

Demonstrates how UnifiedCache automatically chooses optimal storage
formats based on data types.

Usage:
    python intelligent_storage_demo.py
"""

from cacheness import cached
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# ===== UnifiedCache: Intelligent Storage Examples =====


@dataclass
class UserProfile:
    user_id: int
    name: str
    email: str
    preferences: dict


@cached(ttl_seconds=43200)  # 12 hours - Stores as pickle - perfect for custom objects
def get_user_profile(user_id):
    """Custom object - cached with pickle."""
    print(f"🔍 Loading user profile: {user_id}")
    return UserProfile(
        user_id=user_id,
        name=f"User {user_id}",
        email=f"user{user_id}@example.com",
        preferences={"theme": "dark", "notifications": True},
    )


@cached(ttl_seconds=86400)  # 24 hours - Stores as Parquet - optimized for DataFrames
def generate_analytics_data(department):
    """DataFrame - cached with Parquet format."""
    print(f"📊 Generating analytics for {department}")
    return pd.DataFrame(
        {
            "employee_id": range(100),
            "department": [department] * 100,
            "performance": np.random.normal(85, 10, 100),
            "salary": np.random.normal(75000, 15000, 100),
        }
    )


@cached(ttl_seconds=21600)  # 6 hours - Stores with Blosc - optimized for NumPy
def compute_matrix(size):
    """NumPy array - cached with Blosc compression."""
    print(f"🧮 Computing {size}x{size} matrix")
    return np.random.random((size, size))


@cached.for_api(ttl_seconds=14400)  # 4 hours - Stores with LZ4 - fast for JSON
def fetch_api_data(endpoint):
    """API response - cached with fast LZ4 compression."""
    print(f"🌐 Fetching data from {endpoint}")
    return {
        "status": "success",
        "data": [{"id": i, "value": f"item_{i}"} for i in range(50)],
        "timestamp": str(datetime.now()),
    }


def main():
    """Demonstrate intelligent storage format optimization."""

    print("=== Intelligent Storage Demo ===\n")

    # UnifiedCache automatically chooses optimal storage
    print("🧠 UNIFIEDCACHE: Automatic Storage Optimization")

    # Custom objects → Pickle
    profile = get_user_profile(123)
    print(f"✅ Custom object (pickle): {profile.name}")

    # DataFrames → Parquet
    df = generate_analytics_data("Engineering")
    print(f"✅ DataFrame (parquet): {len(df)} rows")

    # NumPy arrays → Blosc
    matrix = compute_matrix(100)
    print(f"✅ NumPy array (blosc): {matrix.shape}")

    # API responses → LZ4
    api_data = fetch_api_data("users")
    print(f"✅ API response (lz4): {len(api_data['data'])} items\n")

    print("🎯 Key Benefits:")
    print("   • Automatic format optimization")
    print("     - DataFrames → Parquet (columnar)")
    print("     - NumPy → Blosc (numerical compression)")
    print("     - Objects → Pickle (serialization)")
    print("     - JSON → LZ4 (fast text compression)")


if __name__ == "__main__":
    main()
