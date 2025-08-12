"""
Machine Learning Pipeline Example

This example demonstrates how to use cacheness for ML workflows
with automatic caching of expensive operations.
"""

from cacheness import cached, cacheness, CacheConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Initialize ML-focused cache
ml_config = CacheConfig(
    cache_dir="./ml_cache",
    default_ttl_hours=168,  # 1 week for model artifacts
    metadata_backend="sqlite",
    max_cache_size_mb=5000  # Larger cache for ML data
)
cache = cacheness(ml_config)


@cached(cache_instance=cache, ttl_hours=24, key_prefix="data_loading")
def load_and_preprocess_data(data_source, preprocessing_params):
    """Load and preprocess data with caching."""
    print(f"Loading data from {data_source}...")
    
    # Simulate expensive data loading
    if "iris" in data_source.lower():
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
    else:
        # For demo, create synthetic data
        np.random.seed(42)
        n_samples = preprocessing_params.get('n_samples', 1000)
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'target': np.random.randint(0, 3, n_samples)
        })
    
    # Apply preprocessing
    if preprocessing_params.get('normalize', False):
        feature_cols = [col for col in df.columns if col != 'target']
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    if preprocessing_params.get('handle_outliers') == 'clip':
        feature_cols = [col for col in df.columns if col != 'target']
        for col in feature_cols:
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    print(f"Processed dataset shape: {df.shape}")
    return df


@cached(cache_instance=cache, ttl_hours=72, key_prefix="feature_engineering")
def engineer_features(raw_data, feature_config):
    """Feature engineering with caching."""
    print("Engineering features...")
    
    df = raw_data.copy()
    feature_cols = [col for col in df.columns if col != 'target']
    
    # Add interaction features if requested
    if feature_config.get('include_interactions', False):
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # Add polynomial features if requested
    poly_degree = feature_config.get('polynomial_degree', 1)
    if poly_degree > 1:
        for col in feature_cols:
            for degree in range(2, poly_degree + 1):
                df[f'{col}_poly_{degree}'] = df[col] ** degree
    
    print(f"Feature engineered dataset shape: {df.shape}")
    return df


@cached(cache_instance=cache, ttl_hours=168, key_prefix="model_training")
def train_model(data, model_type, hyperparams):
    """Train model with automatic caching."""
    print(f"Training {model_type} model with hyperparams: {hyperparams}")
    
    # Prepare data
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model based on type
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', None),
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return model artifacts
    return {
        'model': model,
        'accuracy': accuracy,
        'feature_names': feature_cols,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'training_shape': X_train.shape,
        'test_shape': X_test.shape
    }


@cached(cache_instance=cache, ttl_hours=48, key_prefix="model_evaluation")
def evaluate_model_cross_validation(data, model_type, hyperparams, cv_folds=5):
    """Perform cross-validation with caching."""
    print(f"Performing {cv_folds}-fold CV for {model_type}...")
    
    from sklearn.model_selection import cross_val_score
    
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols]
    y = data['target']
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', None),
            random_state=42
        )
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    
    return {
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'model_type': model_type,
        'hyperparams': hyperparams
    }


def run_experiment(experiment_name, data_source, preprocessing_config, feature_config, model_configs):
    """Run a complete ML experiment with caching at each stage."""
    
    print(f"\n=== Running ML Experiment: {experiment_name} ===\n")
    
    # Stage 1: Data loading and preprocessing (cached)
    data = load_and_preprocess_data(data_source, preprocessing_config)
    
    # Stage 2: Feature engineering (cached)
    features = engineer_features(data, feature_config)
    
    # Stage 3: Model training and evaluation (cached per configuration)
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name}...")
        
        # Train model (cached)
        model_result = train_model(
            data=features,
            model_type=config['type'],
            hyperparams=config['hyperparams']
        )
        
        # Cross-validation (cached)
        cv_result = evaluate_model_cross_validation(
            data=features,
            model_type=config['type'],
            hyperparams=config['hyperparams']
        )
        
        results[model_name] = {
            'model_result': model_result,
            'cv_result': cv_result
        }
        
        print(f"  Accuracy: {model_result['accuracy']:.3f}")
        print(f"  CV Score: {cv_result['mean_cv_score']:.3f} ± {cv_result['std_cv_score']:.3f}")
    
    return results


def main():
    """Demonstrate ML pipeline caching."""
    
    # Experiment configuration
    preprocessing_config = {
        'normalize': True,
        'handle_outliers': 'clip',
        'n_samples': 1500
    }
    
    feature_config = {
        'include_interactions': True,
        'polynomial_degree': 2
    }
    
    model_configs = {
        'rf_small': {
            'type': 'random_forest',
            'hyperparams': {'n_estimators': 50, 'max_depth': 5}
        },
        'rf_large': {
            'type': 'random_forest',
            'hyperparams': {'n_estimators': 200, 'max_depth': 10}
        }
    }
    
    # Run experiment - first time will compute everything
    print("First run (computing all stages):")
    results1 = run_experiment(
        experiment_name="Feature_Selection_Study",
        data_source="synthetic_classification",
        preprocessing_config=preprocessing_config,
        feature_config=feature_config,
        model_configs=model_configs
    )
    
    # Run again with same parameters - should use cached results
    print("\n" + "="*60)
    print("Second run (using cached results):")
    results2 = run_experiment(
        experiment_name="Feature_Selection_Study",
        data_source="synthetic_classification",
        preprocessing_config=preprocessing_config,
        feature_config=feature_config,
        model_configs=model_configs
    )
    
    # Different configuration - will recompute affected stages
    print("\n" + "="*60)
    print("Third run (different feature config, partial cache usage):")
    
    feature_config_v2 = {
        'include_interactions': False,  # Different feature engineering
        'polynomial_degree': 1
    }
    
    results3 = run_experiment(
        experiment_name="Feature_Selection_Study_v2",
        data_source="synthetic_classification",
        preprocessing_config=preprocessing_config,  # Same preprocessing (cached)
        feature_config=feature_config_v2,          # Different features (recomputed)
        model_configs=model_configs                # Same models (recomputed with new features)
    )
    
    # Cache statistics
    print("\n=== Cache Statistics ===")
    stats = cache.get_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Cache size: {stats['total_size_mb']:.2f} MB")
    print(f"Backend: {stats.get('backend_type', 'unknown')}")
    
    # List cached entries
    print("\n=== Cached Pipeline Stages ===")
    entries = cache.list_entries()
    for entry in entries[:10]:  # Show first 10 entries
        print(f"  {entry.get('description', 'N/A')}: {entry.get('size_mb', 0):.2f}MB")


if __name__ == "__main__":
    main()
