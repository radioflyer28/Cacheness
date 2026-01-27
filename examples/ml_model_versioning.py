#!/usr/bin/env python3
"""
ML Model Versioning Example
===========================

Demonstrates using BlobStore for ML model versioning with rich metadata.
This is a non-caching use case - pure storage with queryable metadata.

Features demonstrated:
- Content-addressable storage (deduplicate identical models)
- Rich metadata for model tracking
- Listing and filtering models by metadata

Usage:
    uv run python examples/ml_model_versioning.py
"""

import sys
from pathlib import Path
import tempfile
from datetime import datetime

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cacheness.storage import BlobStore
import numpy as np


def create_mock_model(model_type: str, complexity: int):
    """Create a mock ML model (dict with weights for this example)."""
    np.random.seed(complexity)
    return {
        "model_type": model_type,
        "weights": np.random.randn(100, complexity),
        "bias": np.random.randn(complexity),
        "config": {
            "learning_rate": 0.001,
            "epochs": complexity * 10,
        }
    }


def main():
    print("=" * 60)
    print("ML Model Versioning with BlobStore")
    print("=" * 60)
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir) / "models"
        
        # Create a BlobStore for model versioning
        # NOTE: Using JSON backend because it preserves all custom metadata fields.
        # The SQLite backend stores only core fields (key, hash, size, etc.) by default.
        # For queryable metadata with SQLite, see the CustomMetadataModel feature.
        model_store = BlobStore(
            cache_dir=models_dir,
            backend="json",  # JSON preserves all custom metadata fields
            compression="lz4",
            content_addressable=False,  # Use explicit keys for versioning
        )
        
        try:
            print("\n📦 Storing models with version metadata...")
            print("-" * 40)
            
            # Store several model versions
            models_info = [
                ("fraud_detector", "v1.0", "xgboost", 0.85, "alice"),
                ("fraud_detector", "v1.1", "xgboost", 0.89, "alice"),
                ("fraud_detector", "v2.0", "lightgbm", 0.92, "bob"),
                ("fraud_detector", "v2.1", "lightgbm", 0.95, "bob"),
                ("churn_predictor", "v1.0", "random_forest", 0.78, "carol"),
                ("churn_predictor", "v1.1", "xgboost", 0.82, "carol"),
            ]
            
            stored_keys = []
            for name, version, model_type, accuracy, author in models_info:
                # Create mock model
                model = create_mock_model(model_type, int(accuracy * 100))
                
                # Store with rich metadata
                key = f"{name}_{version}"
                blob_key = model_store.put(
                    model,
                    key=key,
                    metadata={
                        "model_name": name,
                        "version": version,
                        "model_type": model_type,
                        "accuracy": accuracy,
                        "author": author,
                        "training_date": datetime.now().isoformat(),
                        "framework": "sklearn",
                        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    }
                )
                stored_keys.append(blob_key)
                print(f"  ✅ Stored {name} {version} ({model_type}, accuracy={accuracy}) -> {blob_key}")
            
            # List all models
            print("\n📋 All stored models:")
            print("-" * 40)
            all_keys = model_store.list()
            for key in all_keys:
                meta = model_store.get_metadata(key)
                nested = meta.get("metadata", {})
                name = nested.get("model_name", "unknown")
                version = nested.get("version", "?")
                accuracy = nested.get("accuracy", 0)
                print(f"  - {key}: {name} {version} (accuracy: {accuracy:.2f})")
            
            # Filter models by prefix
            print("\n🔍 Fraud detector models only:")
            print("-" * 40)
            fraud_keys = model_store.list(prefix="fraud_detector")
            for key in fraud_keys:
                meta = model_store.get_metadata(key)
                nested = meta.get("metadata", {})
                print(f"  - {key}: version={nested.get('version')}, "
                      f"accuracy={nested.get('accuracy', 0):.2f}")
            
            # Get a specific model
            print("\n📥 Retrieving best fraud detector model...")
            print("-" * 40)
            best_key = "fraud_detector_v2.1"
            model = model_store.get(best_key)
            if model:
                print(f"  Model type: {model['model_type']}")
                print(f"  Weights shape: {model['weights'].shape}")
                print(f"  Config: {model['config']}")
            
            # Update metadata (e.g., mark as deployed)
            print("\n🚀 Marking model as deployed...")
            print("-" * 40)
            model_store.update_metadata(best_key, {
                "deployed": True,
                "deployment_date": datetime.now().isoformat(),
                "deployment_env": "production"
            })
            
            updated_meta = model_store.get_metadata(best_key)
            if updated_meta:
                nested = updated_meta.get("metadata", {})
                print(f"  Deployed: {nested.get('deployed')}")
                print(f"  Deployment date: {nested.get('deployment_date')}")
            
            print("\n" + "=" * 60)
            print("✨ ML Model Versioning Example Complete!")
            print("=" * 60)
        
        finally:
            # Cleanup - close the store to release SQLite connection
            model_store.close()


if __name__ == "__main__":
    main()
