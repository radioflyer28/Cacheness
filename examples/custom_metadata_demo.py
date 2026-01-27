#!/usr/bin/env python3
"""
Simple Custom Metadata Example
==============================

This example demonstrates the core custom metadata functionality:
1. Creating custom metadata schemas
2. Storing cache entries with custom metadata (single or multiple types)
3. Querying cache entries using custom metadata (query_custom)
4. Querying built-in cache metadata (query_meta)

Two Query Methods:
- query_custom(schema_name): Query dedicated custom metadata tables (SQLAlchemy models)
- query_meta(**filters): Query built-in metadata stored with cache entries (requires store_cache_key_params=True)

Supported custom_metadata formats:
- Single object: custom_metadata=experiment_metadata
- Multiple objects: custom_metadata=[experiment_metadata, performance_metadata]
- Dictionary (legacy): custom_metadata={"experiments": experiment_metadata}

Usage:
    uv run python examples/custom_metadata_demo.py
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    is_custom_metadata_available,
    migrate_custom_metadata_tables,
)
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer

# Check if custom metadata is available
if not is_custom_metadata_available():
    print("❌ SQLAlchemy not available - custom metadata will not work")
    print("Install with: uv add sqlalchemy")
    sys.exit(1)

print("✅ Custom metadata functionality is available")


# Define a simple custom metadata schema
@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    """Custom metadata for ML experiments."""

    __tablename__ = "custom_experiments"

    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False, index=True)
    created_by = Column(String(100), nullable=False, index=True)


@custom_metadata_model("performance")
class PerformanceMetadata(Base, CustomMetadataBase):
    """Custom metadata for performance metrics."""

    __tablename__ = "custom_performance"

    run_id = Column(String(100), nullable=False, unique=True, index=True)
    training_time_seconds = Column(Integer, nullable=False, index=True)
    memory_usage_mb = Column(Float, nullable=False, index=True)


def main():
    """Demonstrate core custom metadata functionality."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🏗️  Using temporary cache directory: {temp_dir}")

        # Create cache configuration with SQLite backend (required for custom metadata)
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )

        # Initialize cache
        cache = cacheness(config)

        # Ensure custom metadata tables are created
        migrate_custom_metadata_tables()

        print("\n💾 Storing Cache Entries with Custom Metadata")
        print("=" * 50)

        # Example 1: Store a model with single metadata object (new simple syntax)
        model_weights = np.random.random((100, 50))
        
        experiment_metadata = ExperimentMetadata(
            experiment_id="exp_001",
            model_type="xgboost", 
            accuracy=0.95,
            epochs=100,
            created_by="alice"
        )
        
        experiment_key = cache.put(
            model_weights,
            experiment="exp_001", 
            model="xgboost",
            custom_metadata=experiment_metadata
        )
        print(f"✅ Stored experiment 1: {experiment_key[:12]}...")

        # Example 2: Store another model with different metadata
        model_data = {"architecture": "cnn", "layers": 5, "parameters": 1000000}
        
        experiment_metadata2 = ExperimentMetadata(
            experiment_id="exp_002",
            model_type="cnn",
            accuracy=0.88,
            epochs=50,
            created_by="bob"
        )
        
        experiment_key2 = cache.put(
            model_data,
            experiment="exp_002",
            model="cnn", 
            custom_metadata=experiment_metadata2
        )
        print(f"✅ Stored experiment 2: {experiment_key2[:12]}...")

        # Example 3: Store a third model
        results = {"validation_loss": 0.12, "test_accuracy": 0.92}
        
        experiment_metadata3 = ExperimentMetadata(
            experiment_id="exp_003", 
            model_type="random_forest",
            accuracy=0.92,
            epochs=0,  # No epochs for random forest
            created_by="alice"
        )
        
        experiment_key3 = cache.put(
            results,
            experiment="exp_003",
            model="random_forest",
            custom_metadata=experiment_metadata3
        )
        print(f"✅ Stored experiment 3: {experiment_key3[:12]}...")

        # Example 4: Store with multiple metadata types (list format)
        complex_model = {"model": "deep_learning", "layers": 50}
        
        experiment_metadata4 = ExperimentMetadata(
            experiment_id="exp_004",
            model_type="deep_learning",
            accuracy=0.98,
            epochs=200,
            created_by="charlie"
        )
        
        performance_metadata4 = PerformanceMetadata(
            run_id="run_004",
            training_time_seconds=7200,  # 2 hours
            memory_usage_mb=8192.5  # 8GB
        )
        
        experiment_key4 = cache.put(
            complex_model,
            experiment="exp_004",
            model="deep_learning",
            custom_metadata=[experiment_metadata4, performance_metadata4]  # List of metadata
        )
        print(f"✅ Stored experiment 4 with multiple metadata types: {experiment_key4[:12]}...")

        print("\n🔍 Querying Cache with Custom Metadata")
        print("=" * 45)

        # Query 1: Find all experiments by alice using context manager for advanced filtering
        with cache.query_custom_session("experiments") as query:
            alice_experiments = query.filter(ExperimentMetadata.created_by == "alice").all()
            print(f"✅ Found {len(alice_experiments)} experiments by alice:")
            for exp in alice_experiments:
                print(f"   - {exp.experiment_id}: {exp.model_type} (accuracy: {exp.accuracy})")

            # Query 2: Find high-accuracy models (>= 0.9)
            high_accuracy = query.filter(ExperimentMetadata.accuracy >= 0.9).all()
            print(f"\n✅ Found {len(high_accuracy)} high-accuracy experiments (>= 0.9):")
            for exp in high_accuracy:
                print(f"   - {exp.experiment_id}: {exp.model_type} (accuracy: {exp.accuracy})")

            # Query 3: Find models by type
            xgboost_models = query.filter(ExperimentMetadata.model_type == "xgboost").all()
            print(f"\n✅ Found {len(xgboost_models)} XGBoost experiments:")
            for exp in xgboost_models:
                print(f"   - {exp.experiment_id}: epochs={exp.epochs}, accuracy={exp.accuracy}")

            # Query 4: Complex query - models with epochs > 0 and accuracy > 0.9
            trained_high_acc = query.filter(
                ExperimentMetadata.epochs > 0,
                ExperimentMetadata.accuracy > 0.9
            ).all()
            print(f"\n✅ Found {len(trained_high_acc)} trained models with high accuracy:")
            for exp in trained_high_acc:
                print(f"   - {exp.experiment_id}: {exp.model_type} ({exp.epochs} epochs, {exp.accuracy} accuracy)")

        # Query 5: Query performance metadata
        with cache.query_custom_session("performance") as perf_query:
            long_running = perf_query.filter(PerformanceMetadata.training_time_seconds > 3600).all()
            print(f"\n✅ Found {len(long_running)} long-running experiments (> 1 hour):")
            for perf in long_running:
                print(f"   - {perf.run_id}: {perf.training_time_seconds}s, {perf.memory_usage_mb:.1f}MB")

        print("\n🔍 Querying Built-in Cache Metadata")
        print("=" * 40)
        
        # Now let's also demonstrate query_meta() for built-in metadata
        # First, create a new cache that stores cache key parameters
        print("📝 Creating cache with parameter storage enabled...")
        config_with_params = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
            store_cache_key_params=True,  # Enable parameter storage
        )
        param_cache = cacheness(config_with_params)
        
        # Store some test data with parameters
        param_cache.put("test_model_1", experiment="param_exp_1", model_type="xgboost", accuracy=0.95)
        param_cache.put("test_model_2", experiment="param_exp_2", model_type="cnn", accuracy=0.88) 
        param_cache.put("test_model_3", experiment="param_exp_3", model_type="random_forest", accuracy=0.92)
        
        # Query using built-in metadata
        print("\n🔍 Querying by cache key parameters...")
        
        # First, let's see what parameters are actually stored
        all_param_entries = param_cache.query_meta()
        if all_param_entries:
            print(f"✅ Found {len(all_param_entries)} total entries with stored parameters")
            print("📋 Sample stored parameters:")
            for i, entry in enumerate(all_param_entries[:2]):  # Show first 2
                params = entry.get('cache_key_params', {})
                print(f"   Entry {i+1}: {params}")
        
        # Query by model type (checking the actual format) 
        xgboost_entries = param_cache.query_meta(model_type="str:xgboost")
        if xgboost_entries:
            print(f"\n✅ Found {len(xgboost_entries)} XGBoost entries:")
            for entry in xgboost_entries:
                params = entry.get('cache_key_params', {})
                print(f"   - {entry['cache_key'][:12]}... (experiment: {params.get('experiment', 'N/A')})")
        else:
            print("\n🔍 No XGBoost entries found, trying different patterns...")
        
        # Query by experiment (with proper serialization format)
        specific_exp = param_cache.query_meta(experiment="str:param_exp_1")
        if specific_exp:
            print(f"\n✅ Found {len(specific_exp)} entries for param_exp_1:")
            for entry in specific_exp:
                params = entry.get('cache_key_params', {})
                print(f"   - {entry['description']} (model: {params.get('model_type', 'N/A')})")
        else:
            print("\n🔍 No entries found for param_exp_1 with str: prefix")

        print("\n📊 Cache Statistics")
        print("=" * 20)
        stats = cache.get_stats()
        print(f"Total entries: {stats['total_entries']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")

        print("\n🎉 Custom metadata demo completed successfully!")


if __name__ == "__main__":
    main()
