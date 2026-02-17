#!/usr/bin/env python3
"""
ML Pipeline Caching
===================

Cache expensive ML steps (data loading, feature engineering, training)
so re-runs are instant. No special ML decorator needed — @cached works
with DataFrames, NumPy arrays, and sklearn models out of the box.

Usage:
    uv run python examples/simple_ml_pipeline.py

Requires:
    uv add scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cacheness import cached


@cached(ttl="2d")
def create_dataset(n_samples=1000, n_features=20, seed=42):
    """Generate a synthetic classification dataset."""
    print("  Creating dataset...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        random_state=seed,
    )
    return pd.DataFrame(X), pd.Series(y)


@cached(ttl="1w")
def train_model(X_train, y_train, n_estimators=100, seed=42):
    """Train a Random Forest classifier."""
    print("  Training model...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model


@cached(ttl="1d")
def evaluate(model, X_test, y_test):
    """Compute accuracy on the test set."""
    print("  Evaluating model...")
    return {"accuracy": accuracy_score(y_test, model.predict(X_test))}


if __name__ == "__main__":
    # Run 1 — computes everything
    X, y = create_dataset(n_samples=2000, n_features=15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train, n_estimators=50)
    results = evaluate(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.3f}\n")

    # Run 2 — all cached (no "Creating / Training / Evaluating" printed)
    print("Running again (should be cached):")
    X2, y2 = create_dataset(n_samples=2000, n_features=15)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )
    model2 = train_model(X_tr2, y_tr2, n_estimators=50)
    results2 = evaluate(model2, X_te2, y_te2)
    print(f"Cached accuracy: {results2['accuracy']:.3f}")
