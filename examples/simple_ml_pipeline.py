#!/usr/bin/env python3
"""
Simple ML Pipeline Example
==========================

Demonstrates caching in ML workflows with the new simplified API.
Much easier than the complex multi-stage version!

Usage:
    python simple_ml_pipeline.py
"""

from cacheness import cached
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Use the new analytics-optimized decorator for ML workflows
@cached.for_analytics(ttl_seconds=172800)  # 48 hours - Cache ML results for 2 days
def create_dataset(n_samples=1000, n_features=20, random_state=42):
    """Create a synthetic dataset for ML."""
    print(f"🔬 Creating dataset: {n_samples} samples, {n_features} features")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=random_state,
    )

    return pd.DataFrame(X), pd.Series(y)


@cached.for_analytics(ttl_seconds=604800)  # 168 hours - Cache models for 1 week
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest model."""
    print(f"🤖 Training model: {len(X_train)} samples, {n_estimators} trees")

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


@cached.for_lookup(ttl_seconds=86400)  # 24 hours - Cache predictions for 1 day
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print(f"📊 Evaluating model on {len(X_test)} test samples")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "n_test_samples": len(X_test),
    }


def main():
    """Demonstrate simple ML pipeline caching."""

    print("=== Simple ML Pipeline Demo ===\n")

    # Step 1: Create dataset (cached)
    X, y = create_dataset(n_samples=2000, n_features=15)
    print(f"✅ Dataset created: {X.shape}")

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test\n")

    # Step 3: Train model (cached - expensive operation)
    model = train_model(X_train, y_train, n_estimators=50)
    print("✅ Model trained and cached\n")

    # Step 4: Evaluate model (cached)
    results = evaluate_model(model, X_test, y_test)
    print(f"✅ Model accuracy: {results['accuracy']:.3f}")
    print(f"✅ Test samples: {results['n_test_samples']}\n")

    print("=" * 50)
    print("🔄 Running pipeline again (should use cache)...")

    # Run again - should use cached results
    X2, y2 = create_dataset(n_samples=2000, n_features=15)  # Cached
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )

    model2 = train_model(X_train2, y_train2, n_estimators=50)  # Cached
    results2 = evaluate_model(model2, X_test2, y_test2)  # Cached

    print(f"✅ Cached accuracy: {results2['accuracy']:.3f}")
    print("\n🎯 Benefits:")
    print("   • Expensive dataset creation cached")
    print("   • Model training cached (no retraining)")
    print("   • Evaluation results cached")
    print("   • Automatic analytics optimization")


if __name__ == "__main__":
    main()
