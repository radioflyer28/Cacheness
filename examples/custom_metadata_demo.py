#!/usr/bin/env python3
"""
Complete Custom Metadata Integration Example
==========================================

This example demonstrates the full integration of custom metadata functionality
with the cacheness library, including:

1. Schema definition with registry decorator
2. Storing cache entries with custom metadata
3. Retrieving cache entries with their metadata
4. Advanced querying capabilities
5. Migration and maintenance utilities

Usage:
    uv run python examples/complete_custom_metadata_demo.py
"""

import sys
from pathlib import Path
import tempfile

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timezone
from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    is_custom_metadata_available,
    migrate_custom_metadata_tables,
    list_registered_schemas,
    cleanup_orphaned_metadata,
)
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer, Text, DateTime
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Check if custom metadata is available
if not is_custom_metadata_available():
    print("❌ SQLAlchemy not available - custom metadata will not work")
    print("Install with: uv add sqlalchemy")
    sys.exit(1)

print("✅ Custom metadata functionality is available")


# Define custom metadata schemas
@custom_metadata_model("ml_experiments")
class MLExperimentMetadata(Base, CustomMetadataBase):
    """Custom metadata for ML experiments with comprehensive tracking."""

    __tablename__ = "custom_ml_experiments"

    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    f1_score = Column(Float, nullable=True, index=True)
    precision = Column(Float, nullable=True, index=True)
    recall = Column(Float, nullable=True, index=True)
    training_time_minutes = Column(Integer, nullable=True, index=True)
    hyperparams = Column(Text, nullable=True)  # JSON serialized
    feature_count = Column(Integer, nullable=True, index=True)
    sample_count = Column(Integer, nullable=True, index=True)
    created_by = Column(String(100), nullable=False, index=True)
    environment = Column(
        String(50), nullable=False, index=True
    )  # "dev", "staging", "prod"
    notes = Column(Text, nullable=True)


@custom_metadata_model("data_pipelines")
class DataPipelineMetadata(Base, CustomMetadataBase):
    """Custom metadata for data processing and ETL pipelines."""

    __tablename__ = "custom_data_pipelines"

    pipeline_id = Column(String(100), nullable=False, unique=True, index=True)
    pipeline_name = Column(String(200), nullable=False, index=True)
    pipeline_version = Column(String(20), nullable=False, index=True)
    source_system = Column(String(100), nullable=False, index=True)
    target_system = Column(String(100), nullable=False, index=True)
    data_source = Column(Text, nullable=False)  # JSON list of source tables/files
    execution_time_seconds = Column(Integer, nullable=False, index=True)
    rows_processed = Column(Integer, nullable=False, index=True)
    bytes_processed = Column(Integer, nullable=True, index=True)
    status = Column(
        String(20), nullable=False, index=True
    )  # "success", "failed", "running"
    error_message = Column(Text, nullable=True)
    checksum = Column(String(64), nullable=True)  # Data integrity checksum
    triggered_by = Column(String(100), nullable=False, index=True)


@custom_metadata_model("model_deployments")
class ModelDeploymentMetadata(Base, CustomMetadataBase):
    """Custom metadata for model deployment tracking."""

    __tablename__ = "custom_model_deployments"

    deployment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(20), nullable=False, index=True)
    deployment_environment = Column(String(50), nullable=False, index=True)
    endpoint_url = Column(String(500), nullable=True)
    container_image = Column(String(200), nullable=True)
    cpu_requests = Column(Float, nullable=True)  # CPU cores
    memory_requests = Column(Integer, nullable=True)  # MB
    replicas = Column(Integer, nullable=False, index=True)
    deployment_config = Column(Text, nullable=True)  # JSON
    health_check_url = Column(String(500), nullable=True)
    deployed_by = Column(String(100), nullable=False, index=True)


def create_sample_data():
    """Create sample data to cache with custom metadata."""
    import numpy as np

    # Sample model data (simplified)
    model_data = {
        "weights": np.random.random((10, 5)),
        "biases": np.random.random(5),
        "architecture": "dense_neural_network",
        "input_shape": (10,),
        "output_shape": (5,),
    }

    # Sample processed dataset
    dataset = {
        "features": np.random.random((1000, 10)),
        "labels": np.random.randint(0, 5, 1000),
        "feature_names": [f"feature_{i}" for i in range(10)],
        "preprocessing_steps": ["normalization", "feature_selection"],
    }

    # Sample deployment artifact
    deployment_artifact = {
        "model_binary": b"fake_serialized_model_data",
        "requirements": ["numpy==1.21.0", "scikit-learn==1.0.0"],
        "dockerfile": "FROM python:3.9\nCOPY . .\nRUN pip install -r requirements.txt",
        "k8s_manifest": {"apiVersion": "apps/v1", "kind": "Deployment"},
    }

    return model_data, dataset, deployment_artifact


def main():
    """Demonstrate complete custom metadata integration."""

    # Create temporary directory for this demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🏗️  Using temporary cache directory: {temp_dir}")

        # Create cache configuration with SQLite backend (required for custom metadata)
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",  # Required for custom metadata
            store_cache_key_params=True,  # Enable parameter tracking
        )

        # Initialize cache
        cache = cacheness(config)

        print(
            f"\n📋 Registered Custom Metadata Schemas: {len(list_registered_schemas())}"
        )
        for schema_name in list_registered_schemas():
            print(f"  • {schema_name}")

        # Ensure custom metadata tables are created
        migrate_custom_metadata_tables()

        # Create sample data
        model_data, dataset, deployment_artifact = create_sample_data()

        print(f"\n💾 Storing Cache Entries with Custom Metadata")
        print("=" * 60)

        # Example 1: Store ML experiment with comprehensive metadata
        ml_metadata = MLExperimentMetadata(
            experiment_id="exp_xgb_customer_churn_v3",
            model_type="xgboost",
            dataset_name="customer_churn_dataset_v2",
            accuracy=0.947,
            f1_score=0.923,
            precision=0.956,
            recall=0.891,
            training_time_minutes=45,
            hyperparams='{"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8}',
            feature_count=42,
            sample_count=50000,
            created_by="alice_ml",
            environment="production",
            notes="Final production model with optimized hyperparameters. Includes feature importance analysis.",
        )

        # Store the model with custom metadata
        cache.put(
            model_data,
            experiment="customer_churn",
            model_type="xgboost",
            version="v3.0",
            custom_metadata={"ml_experiments": ml_metadata},
            description="High-accuracy customer churn prediction model",
        )
        print("✅ Stored ML experiment with custom metadata")

        # Example 2: Store data pipeline results with metadata
        pipeline_metadata = DataPipelineMetadata(
            pipeline_id="daily_customer_features_v2_1",
            pipeline_name="Daily Customer Feature Engineering",
            pipeline_version="v2.1",
            source_system="data_warehouse",
            target_system="feature_store",
            data_source='["customer_profiles", "transaction_history", "support_tickets", "product_usage"]',
            execution_time_seconds=892,
            rows_processed=125000,
            bytes_processed=1024 * 1024 * 500,  # 500MB
            status="success",
            checksum="sha256:a1b2c3d4e5f6789...",
            triggered_by="airflow_scheduler",
        )

        cache.put(
            dataset,
            pipeline="daily_customer_features",
            date="2025-08-11",
            version="v2.1",
            custom_metadata={"data_pipelines": pipeline_metadata},
            description="Daily customer feature engineering output",
        )
        print("✅ Stored data pipeline results with custom metadata")

        # Example 3: Store deployment artifact with metadata
        deployment_metadata = ModelDeploymentMetadata(
            deployment_id="churn_model_prod_20250811",
            model_name="customer_churn_predictor",
            model_version="v3.0",
            deployment_environment="production",
            endpoint_url="https://api.company.com/ml/churn/predict",
            container_image="company.com/ml/churn-predictor:v3.0",
            cpu_requests=2.0,
            memory_requests=4096,
            replicas=3,
            deployment_config='{"autoscaling": {"min": 2, "max": 10}, "monitoring": {"enabled": true}}',
            health_check_url="https://api.company.com/ml/churn/health",
            deployed_by="devops_team",
        )

        cache.put(
            deployment_artifact,
            model="customer_churn_predictor",
            environment="production",
            version="v3.0",
            custom_metadata={"model_deployments": deployment_metadata},
            description="Production deployment artifact for customer churn model",
        )
        print("✅ Stored deployment artifact with custom metadata")

        print(f"\n🔍 Retrieving Cache Entries with Custom Metadata")
        print("=" * 60)

        # Retrieve cache entries and their custom metadata
        retrieved_model = cache.get(
            experiment="customer_churn", model_type="xgboost", version="v3.0"
        )
        if retrieved_model is not None:
            print("✅ Retrieved ML model from cache")

            # Get custom metadata for this entry
            custom_meta = cache.get_custom_metadata_for_entry(
                experiment="customer_churn", model_type="xgboost", version="v3.0"
            )
            if "ml_experiments" in custom_meta:
                exp_meta = custom_meta["ml_experiments"]
                print(f"   • Experiment ID: {exp_meta.experiment_id}")
                print(f"   • Accuracy: {exp_meta.accuracy:.3f}")
                print(f"   • F1 Score: {exp_meta.f1_score:.3f}")
                print(f"   • Training Time: {exp_meta.training_time_minutes} minutes")
                print(f"   • Created By: {exp_meta.created_by}")
                print(f"   • Environment: {exp_meta.environment}")

        retrieved_dataset = cache.get(
            pipeline="daily_customer_features", date="2025-08-11", version="v2.1"
        )
        if retrieved_dataset is not None:
            print("✅ Retrieved dataset from cache")

            custom_meta = cache.get_custom_metadata_for_entry(
                pipeline="daily_customer_features", date="2025-08-11", version="v2.1"
            )
            if "data_pipelines" in custom_meta:
                pipe_meta = custom_meta["data_pipelines"]
                print(f"   • Pipeline: {pipe_meta.pipeline_name}")
                print(f"   • Rows Processed: {pipe_meta.rows_processed:,}")
                print(
                    f"   • Execution Time: {pipe_meta.execution_time_seconds} seconds"
                )
                print(f"   • Status: {pipe_meta.status}")

        print(f"\n📊 Advanced Querying Examples")
        print("=" * 60)

        # Example advanced queries (would work with full integration)
        try:
            # Query high-accuracy ML experiments
            ml_query = cache.query_custom_metadata("ml_experiments")
            if ml_query:
                print("🔍 Available for advanced querying:")
                print("   • High-accuracy models (accuracy >= 0.9)")
                print("   • Models by specific creators")
                print("   • Models in production environment")
                print("   • Models trained on specific datasets")

                # These would work with full SQLAlchemy integration:
                # high_accuracy = ml_query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
                # alice_models = ml_query.filter(MLExperimentMetadata.created_by == "alice_ml").all()
                # prod_models = ml_query.filter(MLExperimentMetadata.environment == "production").all()
        except Exception as e:
            print(f"   Note: Full querying integration pending: {e}")

        print(f"\n🛠️  Maintenance Operations")
        print("=" * 60)

        # Show cache statistics
        stats = cache.get_stats()
        print(f"Cache Statistics:")
        print(f"   • Total Entries: {stats.get('total_entries', 0)}")
        print(f"   • Total Size: {stats.get('total_size_mb', 0):.2f} MB")
        print(f"   • Cache Hits: {stats.get('cache_hits', 0)}")
        print(f"   • Cache Misses: {stats.get('cache_misses', 0)}")
        print(f"   • Hit Rate: {stats.get('hit_rate', 0):.1%}")

        # List all cache entries
        entries = cache.list_entries()
        print(f"\nCache Entries: {len(entries)}")
        for entry in entries:
            print(
                f"   • {entry['cache_key']}: {entry['description']} ({entry['size_mb']:.3f} MB)"
            )

        # Cleanup operations
        print(f"\n🧹 Performing cleanup operations")
        cleanup_orphaned_metadata()

        print(f"\n✅ Custom metadata integration demo completed successfully!")
        print(f"\nThis demonstrates:")
        print(f"   ✓ Schema definition with registry decorator")
        print(f"   ✓ Storing cache entries with structured metadata")
        print(f"   ✓ Retrieving metadata alongside cached data")
        print(f"   ✓ Advanced querying infrastructure")
        print(f"   ✓ Migration and maintenance utilities")
        print(f"   ✓ Production-ready metadata tracking")


if __name__ == "__main__":
    main()
