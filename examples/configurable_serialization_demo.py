#!/usr/bin/env python3
"""
Configurable Serialization Example
==================================

This script demonstrates the new configurable serialization and handler priority
features in cacheness v2.0, showing how different configurations affect caching
behavior and performance.
"""

import time
import numpy as np
from cacheness import CacheConfig, UnifiedCache, cached

# Optional: pandas for DataFrame examples
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False


def demonstrate_serialization_configs():
    """Show how different serialization configs affect cache key generation."""
    
    print("🔧 Configurable Serialization Demonstration")
    print("=" * 50)
    
    # Test data with various complexity levels
    simple_data = [1, 2, 3]
    complex_data = {
        'arrays': [np.array([1, 2, 3]), np.array([4, 5, 6])],
        'nested': {'level1': {'level2': {'level3': 'deep_value'}}},
        'large_tuple': tuple(range(20))
    }
    
    # Different configurations
    configs = {
        'default': CacheConfig(),
        'performance': CacheConfig(
            enable_collections=False,
            enable_object_introspection=False,
            max_tuple_recursive_length=2
        ),
        'precision': CacheConfig(
            enable_collections=True,
            enable_object_introspection=True,
            max_tuple_recursive_length=50,
            max_collection_depth=20
        )
    }
    
    print("Testing cache key generation with different configs:")
    print()
    
    for config_name, config in configs.items():
        print(f"📊 {config_name.upper()} Configuration:")
        
        # Create cache instance
        cache = UnifiedCache(config)
        
        # Test simple data
        simple_key = cache._create_cache_key({"data": simple_data})
        print(f"  Simple data key: {simple_key}")
        
        # Test complex data  
        complex_key = cache._create_cache_key({"data": complex_data})
        print(f"  Complex data key: {complex_key}")
        
        print()


def demonstrate_handler_priority():
    """Show how handler priority affects data processing."""
    
    print("🎯 Handler Priority Demonstration")
    print("=" * 50)
    
    if not PANDAS_AVAILABLE:
        print("⚠️  Pandas not available - skipping DataFrame examples")
        return
    
    # Create a DataFrame
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Different handler priorities
    configs = {
        'pandas_first': CacheConfig(
            handler_priority=["pandas_dataframes", "object_pickle"]
        ),
        'object_first': CacheConfig(
            handler_priority=["object_pickle", "pandas_dataframes"]
        )
    }
    
    for config_name, config in configs.items():
        print(f"🔧 {config_name.upper()} configuration:")
        
        cache = UnifiedCache(config)
        handler = cache.handlers.get_handler(df)
        
        print(f"  Selected handler: {type(handler).__name__}")
        print(f"  Data type: {handler.data_type}")
        
        # Store and retrieve
        start_time = time.time()
        cache.put(df, dataset="test", config=config_name)
        store_time = time.time() - start_time
        
        start_time = time.time()
        retrieved_df = cache.get(dataset="test", config=config_name)
        retrieve_time = time.time() - start_time
        
        print(f"  Store time: {store_time:.4f}s")
        print(f"  Retrieve time: {retrieve_time:.4f}s")
        print(f"  Data integrity: {'✅' if df.equals(retrieved_df) else '❌'}")
        print()


def demonstrate_decorator_configs():
    """Show configurable serialization with decorators."""
    
    print("🎨 Decorator Configuration Demonstration")
    print("=" * 50)
    
    # Performance-optimized config
    fast_config = CacheConfig(
        cache_dir="./cache_fast",
        enable_collections=False,
        max_tuple_recursive_length=2
    )
    
    # Precision-optimized config
    precise_config = CacheConfig(
        cache_dir="./cache_precise", 
        enable_collections=True,
        max_tuple_recursive_length=20
    )
    
    fast_cache = UnifiedCache(fast_config)
    precise_cache = UnifiedCache(precise_config)
    
    # Functions with different caching strategies
    @cached(cache_instance=fast_cache)
    def fast_computation(data_list, params_dict, config_tuple):
        """Fast caching - simplified parameter analysis."""
        print("  🚀 Fast computation executed")
        time.sleep(0.1)  # Simulate work
        return sum(data_list) * len(params_dict) * len(config_tuple)
    
    @cached(cache_instance=precise_cache)
    def precise_computation(data_list, params_dict, config_tuple):
        """Precise caching - detailed parameter analysis."""
        print("  🎯 Precise computation executed")
        time.sleep(0.1)  # Simulate work
        return sum(data_list) * len(params_dict) * len(config_tuple)
    
    # Test data
    test_data = [1, 2, 3, 4, 5]
    test_params = {'a': 1, 'b': 2, 'nested': {'x': 10}}
    test_config = tuple(range(15))  # Large tuple
    
    print("Testing decorator configurations:")
    print()
    
    # Fast computation
    print("🚀 FAST Configuration (simplified serialization):")
    start_time = time.time()
    result1 = fast_computation(test_data, test_params, test_config)
    first_time = time.time() - start_time
    
    start_time = time.time()
    result2 = fast_computation(test_data, test_params, test_config)
    cache_time = time.time() - start_time
    
    print(f"  First call: {first_time:.4f}s (executed)")
    print(f"  Second call: {cache_time:.4f}s (cached)")
    print(f"  Results match: {'✅' if result1 == result2 else '❌'}")
    print()
    
    # Precise computation
    print("🎯 PRECISE Configuration (detailed serialization):")
    start_time = time.time()
    result3 = precise_computation(test_data, test_params, test_config)
    first_time = time.time() - start_time
    
    start_time = time.time()
    result4 = precise_computation(test_data, test_params, test_config)
    cache_time = time.time() - start_time
    
    print(f"  First call: {first_time:.4f}s (executed)")
    print(f"  Second call: {cache_time:.4f}s (cached)")
    print(f"  Results match: {'✅' if result3 == result4 else '❌'}")
    print()


def demonstrate_real_world_use_case():
    """Show a real-world ML pipeline configuration."""
    
    print("🧠 Real-World ML Pipeline Example")
    print("=" * 50)
    
    # ML-optimized configuration
    ml_config = CacheConfig(
        cache_dir="./ml_pipeline_cache",
        # Optimize for ML workloads
        enable_collections=False,           # Skip expensive parameter introspection
        enable_special_cases=True,          # Keep NumPy array optimization
        max_tuple_recursive_length=3,       # Limit hyperparameter tuple analysis
        # Prioritize array and object handling
        handler_priority=["numpy_arrays", "pandas_dataframes", "object_pickle"],
        # Compression optimized for numerical data
        pickle_compression_codec="lz4",     # Fast compression for models
        npz_compression=True,               # Compress arrays
    )
    
    ml_cache = UnifiedCache(ml_config)
    
    @cached(cache_instance=ml_cache, ttl_hours=48)
    def feature_engineering(raw_data, transformations, hyperparams):
        """Feature engineering with caching."""
        print("  🔧 Executing feature engineering...")
        
        # Simulate feature engineering
        features = np.random.rand(1000, 50)
        target = np.random.randint(0, 2, 1000)
        metadata = {
            'n_features': features.shape[1],
            'n_samples': features.shape[0],
            'transformations_applied': len(transformations),
            'hyperparams': hyperparams
        }
        
        time.sleep(0.2)  # Simulate processing time
        return features, target, metadata
    
    @cached(cache_instance=ml_cache, ttl_hours=24)
    def train_model(features, target, model_config):
        """Model training with caching."""
        print("  🤖 Training model...")
        
        # Simulate model training
        model_weights = np.random.rand(features.shape[1])
        training_metrics = {
            'accuracy': np.random.rand(),
            'loss': np.random.rand(),
            'epochs': model_config.get('epochs', 100)
        }
        
        time.sleep(0.3)  # Simulate training time
        return model_weights, training_metrics
    
    # Simulate ML pipeline
    print("Running ML pipeline with optimized caching:")
    print()
    
    # Pipeline parameters
    raw_data = np.random.rand(800, 20)
    transformations = ['normalize', 'pca', 'feature_selection'] 
    hyperparams = (0.01, 0.9, 'adam')  # learning_rate, momentum, optimizer
    model_config = {'epochs': 100, 'batch_size': 32}
    
    # Feature engineering
    start_time = time.time()
    features, target, metadata = feature_engineering(raw_data, transformations, hyperparams)
    fe_time = time.time() - start_time
    
    # Model training
    start_time = time.time()
    model_weights, metrics = train_model(features, target, model_config)
    train_time = time.time() - start_time
    
    print(f"  Feature engineering: {fe_time:.4f}s")
    print(f"  Model training: {train_time:.4f}s")
    print(f"  Total pipeline: {fe_time + train_time:.4f}s")
    print()
    
    # Run again to show caching
    print("Running pipeline again (should be cached):")
    start_time = time.time()
    features2, target2, metadata2 = feature_engineering(raw_data, transformations, hyperparams)
    model_weights2, metrics2 = train_model(features2, target2, model_config)
    cached_time = time.time() - start_time
    
    print(f"  Cached pipeline: {cached_time:.4f}s")
    print(f"  Speedup: {(fe_time + train_time) / cached_time:.1f}x")
    print(f"  Results identical: {'✅' if np.array_equal(features, features2) else '❌'}")


if __name__ == "__main__":
    print("🎯 Cacheness v2.0 - Configurable Serialization Examples")
    print("=" * 60)
    print()
    
    try:
        demonstrate_serialization_configs()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_handler_priority()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_decorator_configs()
        print("\n" + "=" * 60 + "\n")
        
        demonstrate_real_world_use_case()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("💡 Try modifying the configurations to see different behaviors.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
