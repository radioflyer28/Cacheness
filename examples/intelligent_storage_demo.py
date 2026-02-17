#!/usr/bin/env python3
"""
Intelligent Storage
===================

Cacheness automatically picks the best serialisation format
for each data type:

  DataFrames  → Parquet (columnar, fast)
  NumPy arrays → Blosc  (numerical compression)
  JSON / dicts → LZ4    (fast text compression)
  Everything else → Pickle

Usage:
    uv run python examples/intelligent_storage_demo.py
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cacheness import cached


@dataclass
class UserProfile:
    user_id: int
    name: str
    prefs: dict


@cached(ttl="12h")
def load_profile(user_id):
    """Custom object → pickle."""
    print("  Building profile...")
    return UserProfile(user_id, f"User {user_id}", {"theme": "dark"})


@cached(ttl="1d")
def make_report(department):
    """DataFrame → parquet."""
    print("  Generating report...")
    return pd.DataFrame(
        {
            "employee": range(100),
            "dept": [department] * 100,
            "score": np.random.normal(85, 10, 100),
        }
    )


@cached(ttl="6h")
def compute_matrix(size):
    """NumPy array → blosc."""
    print("  Computing matrix...")
    return np.random.random((size, size))


@cached.for_api(ttl="4h")
def fetch_items(endpoint):
    """Dict/JSON → LZ4."""
    print("  Fetching items...")
    return {"status": "ok", "items": list(range(50))}


if __name__ == "__main__":
    profile = load_profile(1)
    print(f"Profile (pickle):   {profile}")

    df = make_report("Engineering")
    print(f"Report  (parquet):  {len(df)} rows")

    mat = compute_matrix(100)
    print(f"Matrix  (blosc):    {mat.shape}")

    data = fetch_items("users")
    print(f"Items   (lz4):      {len(data['items'])} items")
