#!/usr/bin/env python3
"""
Data Pipeline Artifact Storage Example
======================================

Demonstrates using BlobStore for storing intermediate pipeline results.
This is a non-caching use case - pure artifact storage with lineage tracking.

Features demonstrated:
- Storing pipeline stage outputs
- Tracking dependencies between artifacts
- Listing artifacts by pipeline run
- Metadata for artifact lineage

Usage:
    uv run python examples/pipeline_artifact_storage.py
"""

import sys
from pathlib import Path
import tempfile
from datetime import datetime
import time

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cacheness.storage import BlobStore
import numpy as np
import pandas as pd


def extract_data(source: str) -> pd.DataFrame:
    """Simulate data extraction from a source."""
    print(f"  📥 Extracting data from {source}...")
    time.sleep(0.1)  # Simulate I/O
    
    # Generate mock raw data
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(1000),
        "raw_value": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h"),
    })


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate data transformation."""
    print("  🔄 Transforming data...")
    time.sleep(0.1)  # Simulate processing
    
    # Clean and transform
    transformed = df.copy()
    transformed["normalized_value"] = (df["raw_value"] - df["raw_value"].mean()) / df["raw_value"].std()
    transformed["category_code"] = df["category"].map({"A": 1, "B": 2, "C": 3})
    transformed["day_of_week"] = df["timestamp"].dt.dayofweek
    return transformed


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate feature engineering."""
    print("  🧮 Computing features...")
    time.sleep(0.1)  # Simulate processing
    
    features = pd.DataFrame({
        "customer_id": df["customer_id"],
        "feature_1": df["normalized_value"] ** 2,
        "feature_2": df["normalized_value"] * df["category_code"],
        "feature_3": np.sin(df["day_of_week"] * np.pi / 7),
    })
    return features


def main():
    print("=" * 60)
    print("Data Pipeline Artifact Storage with BlobStore")
    print("=" * 60)
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_dir = Path(tmp_dir) / "pipeline_artifacts"
        
        # Create a BlobStore for artifact storage
        artifact_store = BlobStore(
            cache_dir=artifacts_dir,
            compression="zstd",  # Good compression for DataFrames
        )
        
        try:
            # Simulate a pipeline run
            # Key naming convention: {run_id}/{stage_name}
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n🚀 Starting pipeline run: {run_id}")
            print("-" * 40)
            
            # Stage 1: Extract
            print("\n📌 Stage 1: Data Extraction")
            raw_data = extract_data("database://sales")
            
            extract_key = f"{run_id}/01_raw_data"
            artifact_store.put(raw_data, key=extract_key)
            print(f"  ✅ Stored: {extract_key}")
            print(f"     Shape: {raw_data.shape}")
            
            # Stage 2: Transform
            print("\n📌 Stage 2: Data Transformation")
            transformed_data = transform_data(raw_data)
            
            transform_key = f"{run_id}/02_transformed"
            artifact_store.put(transformed_data, key=transform_key)
            print(f"  ✅ Stored: {transform_key}")
            print(f"     Shape: {transformed_data.shape}")
            
            # Stage 3: Feature Engineering
            print("\n📌 Stage 3: Feature Engineering")
            features = compute_features(transformed_data)
            
            features_key = f"{run_id}/03_features"
            artifact_store.put(features, key=features_key)
            print(f"  ✅ Stored: {features_key}")
            print(f"     Shape: {features.shape}")
            
            # List all artifacts for this run
            print(f"\n📋 All artifacts for {run_id}:")
            print("-" * 40)
            run_artifacts = artifact_store.list(prefix=run_id)
            for key in sorted(run_artifacts):
                meta = artifact_store.get_metadata(key)
                size = meta.get("file_size", 0)
                data_type = meta.get("data_type", "unknown")
                print(f"  - {key} ({data_type}, {size:,} bytes)")
            
            # Demonstrate artifact retrieval
            print("\n📥 Retrieving transformed data artifact...")
            print("-" * 40)
            retrieved = artifact_store.get(transform_key)
            if retrieved is not None:
                print(f"  Type: {type(retrieved).__name__}")
                print(f"  Shape: {retrieved.shape}")
                print(f"  Columns: {list(retrieved.columns)}")
                print(f"  Sample:\n{retrieved.head(3).to_string()}")
            
            # Show that artifacts persist and can be retrieved later
            print("\n🔍 Verifying all artifacts exist...")
            print("-" * 40)
            for key in [extract_key, transform_key, features_key]:
                exists = artifact_store.exists(key)
                print(f"  {key}: {'✅' if exists else '❌'}")
            
            print("\n" + "=" * 60)
            print("✨ Pipeline Artifact Storage Example Complete!")
            print("=" * 60)
        
        finally:
            artifact_store.close()


if __name__ == "__main__":
    main()
