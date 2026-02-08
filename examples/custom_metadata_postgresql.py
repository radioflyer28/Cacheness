#!/usr/bin/env python3
"""
PostgreSQL Custom Metadata Example
==================================

This example demonstrates custom SQLAlchemy metadata tables with PostgreSQL backend:
1. Define custom metadata models with @custom_metadata_model decorator
2. Configure cache with PostgreSQL metadata backend
3. Store cache entries with typed, queryable custom metadata
4. Query custom metadata using SQLAlchemy ORM
5. Correlate across multiple custom metadata tables via cache_key
6. Automatic cascade deletion when cache entries are removed

Architecture:
- PostgreSQL stores both infrastructure metadata (cache_entries) and custom metadata
- Custom metadata models use direct foreign key to cache_entries.cache_key
- Each custom metadata record belongs to exactly one cache entry (one-to-many)
- Cascade delete automatically cleans up custom metadata when cache entry deleted

Prerequisites:
    PostgreSQL server running with database created:
    $ createdb cacheness_demo

    Or Docker:
    $ docker run --name postgres-cacheness -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=cacheness_demo -p 5432:5432 -d postgres:15

Usage:
    # Update connection URL if needed
    uv run python examples/custom_metadata_postgresql.py
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
from datetime import datetime, timezone

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    is_custom_metadata_available,
    migrate_custom_metadata_tables,
)
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean

# Check dependencies
if not is_custom_metadata_available():
    print("❌ SQLAlchemy not available")
    print("Install with: uv add sqlalchemy psycopg")
    sys.exit(1)

try:
    import psycopg
except ImportError:
    print("❌ psycopg not available - required for PostgreSQL")
    print("Install with: uv add 'psycopg[binary]'")
    sys.exit(1)

print("✅ PostgreSQL custom metadata dependencies available\n")


# Define custom metadata models
@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    """Custom metadata for ML experiments."""

    __tablename__ = "custom_ml_experiments"

    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    f1_score = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False, index=True)
    is_production = Column(Boolean, default=False, index=True)
    created_by = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


@custom_metadata_model("performance")
class PerformanceMetadata(Base, CustomMetadataBase):
    """Custom metadata for training performance metrics."""

    __tablename__ = "custom_training_performance"

    run_id = Column(String(100), nullable=False, unique=True, index=True)
    training_time_seconds = Column(Float, nullable=False, index=True)
    memory_usage_mb = Column(Float, nullable=False, index=True)
    gpu_utilization = Column(Float, nullable=True, index=True)
    dataset_size = Column(Integer, nullable=False)


def main():
    """Demonstrate PostgreSQL custom metadata functionality."""

    # Configure connection to PostgreSQL
    # Update this URL for your environment
    pg_url = "postgresql://postgres:postgres@localhost:5432/cacheness_demo"

    print(f"Connecting to PostgreSQL: {pg_url}\n")

    try:
        # Create cache with PostgreSQL backend
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,  # For blob storage (model files)
                metadata=CacheMetadataConfig(
                    metadata_backend="postgresql",
                    metadata_backend_options={"connection_url": pg_url},
                ),
            )

            cache = cacheness(config=config)

            # Migrate custom metadata tables
            print("📦 Creating custom metadata tables in PostgreSQL...")
            if hasattr(cache.metadata_backend, "engine"):
                migrate_custom_metadata_tables(engine=cache.metadata_backend.engine)
            print("✅ Tables created\n")

            # ================================================================
            # Example 1: Store single custom metadata
            # ================================================================
            print("=" * 60)
            print("Example 1: Store with single custom metadata")
            print("=" * 60)

            experiment1 = ExperimentMetadata(
                experiment_id="exp_001",
                model_type="xgboost",
                accuracy=0.94,
                f1_score=0.92,
                epochs=100,
                is_production=True,
                created_by="alice",
            )

            # Train a simple model (simulated)
            model_data = {"weights": np.random.randn(10, 5), "bias": np.random.randn(5)}

            cache.put(
                model_data,
                experiment="exp_001",
                custom_metadata=experiment1,
            )
            print(f"✅ Stored experiment: {experiment1.experiment_id}")
            print(f"   Model type: {experiment1.model_type}")
            print(f"   Accuracy: {experiment1.accuracy}")
            print()

            # ================================================================
            # Example 2: Store multiple custom metadata types
            # ================================================================
            print("=" * 60)
            print("Example 2: Store with multiple custom metadata types")
            print("=" * 60)

            experiment2 = ExperimentMetadata(
                experiment_id="exp_002",
                model_type="lightgbm",
                accuracy=0.96,
                f1_score=0.95,
                epochs=150,
                is_production=False,
                created_by="bob",
            )

            performance2 = PerformanceMetadata(
                run_id="run_002",
                training_time_seconds=245.5,
                memory_usage_mb=2048.0,
                gpu_utilization=85.3,
                dataset_size=500000,
            )

            cache.put(
                {"weights": np.random.randn(15, 10), "bias": np.random.randn(10)},
                experiment="exp_002",
                custom_metadata=[
                    experiment2,
                    performance2,
                ],  # Multiple metadata objects
            )
            print(f"✅ Stored experiment: {experiment2.experiment_id}")
            print(f"   Model type: {experiment2.model_type}")
            print(f"   Accuracy: {experiment2.accuracy}")
            print(f"   Training time: {performance2.training_time_seconds}s")
            print()

            # ================================================================
            # Example 3: Store more experiments for querying
            # ================================================================
            print("=" * 60)
            print("Example 3: Store additional experiments")
            print("=" * 60)

            experiments = [
                ("exp_003", "random_forest", 0.89, 0.87, 80, "alice"),
                ("exp_004", "neural_net", 0.93, 0.91, 200, "bob"),
                ("exp_005", "xgboost", 0.97, 0.96, 120, "alice"),
            ]

            for exp_id, model_type, acc, f1, epochs, user in experiments:
                exp = ExperimentMetadata(
                    experiment_id=exp_id,
                    model_type=model_type,
                    accuracy=acc,
                    f1_score=f1,
                    epochs=epochs,
                    is_production=(acc >= 0.95),
                    created_by=user,
                )
                cache.put(
                    {"weights": np.random.randn(10, 5)},
                    experiment=exp_id,
                    custom_metadata=exp,
                )
                print(f"✅ Stored {exp_id}: {model_type} (acc={acc})")
            print()

            # ================================================================
            # Example 4: Query custom metadata with filters
            # ================================================================
            print("=" * 60)
            print("Example 4: Query high-accuracy experiments")
            print("=" * 60)

            with cache.query_custom_session("experiments") as query:
                high_accuracy = query.filter(ExperimentMetadata.accuracy >= 0.95).all()

                print(
                    f"Found {len(high_accuracy)} experiments with accuracy >= 0.95:\n"
                )
                for exp in high_accuracy:
                    print(f"  • {exp.experiment_id}")
                    print(f"    Model: {exp.model_type}")
                    print(f"    Accuracy: {exp.accuracy:.3f}")
                    print(f"    F1 Score: {exp.f1_score:.3f}")
                    print(f"    Production: {'Yes' if exp.is_production else 'No'}")
                    print()

            # ================================================================
            # Example 5: Query by user
            # ================================================================
            print("=" * 60)
            print("Example 5: Query experiments by user")
            print("=" * 60)

            with cache.query_custom_session("experiments") as query:
                alice_experiments = (
                    query.filter(ExperimentMetadata.created_by == "alice")
                    .order_by(ExperimentMetadata.accuracy.desc())
                    .all()
                )

                print("Alice's experiments (sorted by accuracy):\n")
                for exp in alice_experiments:
                    print(
                        f"  • {exp.experiment_id}: {exp.model_type} - {exp.accuracy:.3f}"
                    )
            print()

            # ================================================================
            # Example 6: Query by model type
            # ================================================================
            print("=" * 60)
            print("Example 6: Query XGBoost experiments")
            print("=" * 60)

            with cache.query_custom_session("experiments") as query:
                xgboost_experiments = query.filter(
                    ExperimentMetadata.model_type == "xgboost"
                ).all()

                print("XGBoost experiments:\n")
                for exp in xgboost_experiments:
                    print(f"  • {exp.experiment_id}")
                    print(f"    Accuracy: {exp.accuracy:.3f}")
                    print(f"    Epochs: {exp.epochs}")
                    print()

            # ================================================================
            # Example 7: Correlate across custom tables via cache_key
            # ================================================================
            print("=" * 60)
            print("Example 7: Correlate experiments and performance")
            print("=" * 60)

            # Find high-accuracy experiments
            with cache.query_custom_session("experiments") as query:
                high_acc_exps = query.filter(ExperimentMetadata.accuracy >= 0.95).all()
                high_acc_keys = {exp.cache_key for exp in high_acc_exps}

            # Get corresponding performance metrics
            with cache.query_custom_session("performance") as query:
                perf_metrics = query.filter(
                    PerformanceMetadata.cache_key.in_(high_acc_keys)
                ).all()

                print("Performance metrics for high-accuracy experiments:\n")
                for perf in perf_metrics:
                    print(f"  • {perf.run_id}")
                    print(f"    Training time: {perf.training_time_seconds}s")
                    print(f"    Memory: {perf.memory_usage_mb} MB")
                    print(f"    GPU utilization: {perf.gpu_utilization}%")
                    print()

            # ================================================================
            # Example 8: Retrieve custom metadata for specific entry
            # ================================================================
            print("=" * 60)
            print("Example 8: Get custom metadata for specific entry")
            print("=" * 60)

            custom_meta = cache.get_custom_metadata_for_entry(experiment="exp_002")

            if "experiments" in custom_meta:
                exp = custom_meta["experiments"]
                print("Experiment metadata for exp_002:")
                print(f"  Model type: {exp.model_type}")
                print(f"  Accuracy: {exp.accuracy}")
                print(f"  Created by: {exp.created_by}")
                print()

            if "performance" in custom_meta:
                perf = custom_meta["performance"]
                print("Performance metadata for exp_002:")
                print(f"  Training time: {perf.training_time_seconds}s")
                print(f"  Memory usage: {perf.memory_usage_mb} MB")
                print()

            # ================================================================
            # Example 9: Automatic cascade deletion
            # ================================================================
            print("=" * 60)
            print("Example 9: Automatic cascade deletion")
            print("=" * 60)

            print("Before deletion:")
            with cache.query_custom_session("experiments") as query:
                count_before = query.count()
                print(f"  Total experiments: {count_before}")

            # Delete a cache entry
            print("\nDeleting cache entry for exp_003...")
            cache.invalidate(experiment="exp_003")

            print("\nAfter deletion:")
            with cache.query_custom_session("experiments") as query:
                count_after = query.count()
                print(f"  Total experiments: {count_after}")
                print(
                    f"  ✅ Custom metadata automatically cascade-deleted (removed {count_before - count_after})"
                )

            # Verify metadata is gone
            meta = cache.get_custom_metadata_for_entry(experiment="exp_003")
            print(f"\nget_custom_metadata_for_entry(exp_003): {meta}")
            print("  ✅ Empty dict confirms metadata was cascade-deleted")
            print()

            # ================================================================
            # Summary
            # ================================================================
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            print("✅ PostgreSQL custom metadata features:")
            print("  • Typed, queryable metadata with SQLAlchemy ORM")
            print("  • Multiple custom metadata tables per cache entry")
            print("  • Advanced filtering, sorting, and aggregation")
            print("  • Correlation across tables via cache_key")
            print("  • Automatic cascade deletion (no orphaned metadata)")
            print("  • Production-ready with PostgreSQL backend")
            print()

            # Cleanup
            cache.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure PostgreSQL is running: pg_isready")
        print("  2. Create database: createdb cacheness_demo")
        print("  3. Check connection URL in script")
        print("  4. Verify credentials and permissions")
        sys.exit(1)


if __name__ == "__main__":
    main()
