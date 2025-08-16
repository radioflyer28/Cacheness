#!/usr/bin/env python3
"""
Dill Class Caching Demo - Security-Aware Implementation

Demonstrates caching complex initialized classes with large datasets using dill serialization.
This example includes comprehensive security warnings and validation patterns to show
how to safely use dill in real-world applications.

⚠️ SECURITY WARNING: This demo shows dill usage for educational purposes.
In production environments, consider disabling dill (enable_dill_fallback=False)
to eliminate code execution risks.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List

from cacheness import cacheness, CacheConfig


@dataclass
class ModelExperiment:
    """
    ML experiment results containing model weights, training history, 
    and embedded processing functions.
    
    Security features:
    - Version tracking for schema validation
    - Post-init validation of all data
    - Validation method for post-retrieval checks
    """
    experiment_id: str
    model_weights: np.ndarray
    training_history: Dict[str, List[float]]
    hyperparameters: Dict[str, Any]
    data_processor: Callable = field(init=False)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    _version: str = field(default="1.0", init=False)  # Security: version tracking
    
    def __post_init__(self):
        """Create processing function with captured hyperparameters."""
        # Security: validate version
        if self._version != "1.0":
            raise ValueError(f"Incompatible version: {self._version}")
            
        # Security: validate inputs
        self._validate_inputs()
        
        # Extract hyperparameters for closure
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        dropout_rate = self.hyperparameters.get('dropout_rate', 0.2)
        batch_norm_momentum = self.hyperparameters.get('batch_norm_momentum', 0.99)
        
        # Create processing function that captures experiment-specific parameters
        def process_batch(data):
            """Process data using experiment-specific hyperparameters."""
            # Normalize data
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Apply batch normalization-like transformation
            bn_normalized = normalized * batch_norm_momentum
            
            # Apply dropout simulation
            dropout_mask = np.random.random(data.shape) > dropout_rate
            dropout_applied = bn_normalized * dropout_mask
            
            # Scale by learning rate for gradient-like processing
            return dropout_applied * learning_rate
        
        self.data_processor = process_batch
    
    def _validate_inputs(self):
        """Security: validate all inputs."""
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("Invalid experiment_id")
        
        if not isinstance(self.model_weights, np.ndarray) or self.model_weights.size == 0:
            raise ValueError("Invalid model_weights")
        
        if not isinstance(self.training_history, dict):
            raise ValueError("Invalid training_history")
        
        if not isinstance(self.hyperparameters, dict):
            raise ValueError("Invalid hyperparameters")
    
    def validate_post_retrieval(self):
        """Security: validate object after cache retrieval."""
        try:
            # Check version compatibility
            if self._version != "1.0":
                raise ValueError(f"Version mismatch: {self._version}")
            
            # Validate data integrity
            self._validate_inputs()
            
            # Test that processor works
            if not hasattr(self, 'data_processor') or not callable(self.data_processor):
                raise ValueError("Data processor not properly initialized")
            
            # Test processor functionality
            test_data = np.random.randn(10, 5)
            result = self.data_processor(test_data)
            
            if not isinstance(result, np.ndarray) or result.shape != test_data.shape:
                raise ValueError("Data processor produces invalid output")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Post-retrieval validation failed: {e}")
            return False
    
    def evaluate_model(self, test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate model using cached weights and processor."""
        processed_data = self.data_processor(test_data)
        
        # Simulate model evaluation with cached weights
        predictions = np.dot(processed_data, self.model_weights[:processed_data.shape[1], :100])
        
        # Calculate mock metrics
        metrics = {
            'mean_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions)),
            'max_weight': float(np.max(self.model_weights)),
            'processing_factor': self.hyperparameters.get('learning_rate', 0.001)
        }
        
        return metrics


class ConfigurableDataProcessor:
    """
    Data processing pipeline with dynamic methods based on configuration.
    """
    
    def __init__(self, config: Dict[str, Any], large_lookup_table: np.ndarray):
        self.config = config
        self.lookup_table = large_lookup_table
        self.created_at = time.time()
        
        # Create dynamic processing methods based on configuration
        self._create_processing_methods()
    
    def _create_processing_methods(self):
        """Create processing methods based on configuration."""
        for processor_name, params in self.config.get('processors', {}).items():
            method = self._create_processor_method(processor_name, params)
            setattr(self, f"apply_{processor_name}", method)
    
    def _create_processor_method(self, name: str, params: Dict[str, Any]):
        """Create a closure-based processing method."""
        scale = params.get('scale', 1.0)
        offset = params.get('offset', 0.0)
        use_lookup = params.get('use_lookup_table', False)
        
        def processor_method(data):
            """Processing method with captured parameters."""
            # Basic transformation
            transformed = data * scale + offset
            
            # Optionally use lookup table
            if use_lookup and hasattr(self, 'lookup_table'):
                # Use lookup table for additional processing
                indices = np.clip(np.abs(transformed).astype(int), 0, len(self.lookup_table) - 1)
                lookup_values = self.lookup_table[indices]
                transformed = transformed + lookup_values * 0.1
            
            return transformed
        
        return processor_method
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about available processors."""
        return {
            'config': self.config,
            'lookup_table_size': self.lookup_table.shape,
            'created_at': self.created_at,
            'available_processors': [
                attr for attr in dir(self) 
                if attr.startswith('apply_') and callable(getattr(self, attr))
            ]
        }


def create_large_experiment() -> ModelExperiment:
    """Create a large ML experiment with substantial data."""
    print("🔄 Creating large ML experiment...")
    
    # Simulate large model weights (e.g., transformer model)
    model_weights = np.random.randn(5000, 2000).astype(np.float32)
    
    # Training history with detailed metrics
    training_history = {
        'loss': [2.5, 1.8, 1.2, 0.9, 0.6, 0.4, 0.3, 0.2],
        'accuracy': [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.94],
        'val_loss': [2.8, 2.0, 1.5, 1.1, 0.8, 0.6, 0.5, 0.4],
        'val_accuracy': [0.08, 0.25, 0.45, 0.65, 0.75, 0.8, 0.85, 0.89],
        'learning_rate': [0.001, 0.001, 0.0005, 0.0005, 0.0001, 0.0001, 0.00005, 0.00005]
    }
    
    # Complex hyperparameters
    hyperparameters = {
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'batch_norm_momentum': 0.99,
        'batch_size': 64,
        'epochs': 100,
        'optimizer': 'adam',
        'weight_decay': 1e-5,
        'scheduler': 'cosine_annealing',
        'architecture': {
            'layers': [2000, 1024, 512, 256, 128],
            'activation': 'relu',
            'use_batch_norm': True
        }
    }
    
    experiment = ModelExperiment(
        experiment_id="exp_transformer_v3",
        model_weights=model_weights,
        training_history=training_history,
        hyperparameters=hyperparameters
    )
    
    # Add evaluation metrics
    experiment.evaluation_metrics = {
        'test_accuracy': 0.92,
        'test_loss': 0.35,
        'inference_time_ms': 15.2,
        'model_size_mb': model_weights.nbytes / (1024**2)
    }
    
    print(f"✅ Created experiment with {model_weights.nbytes / (1024**2):.1f}MB of weights")
    return experiment


def create_data_processor() -> ConfigurableDataProcessor:
    """Create a configurable data processor with large lookup table."""
    print("🔄 Creating configurable data processor...")
    
    # Large lookup table for complex data transformations
    lookup_table = np.random.exponential(2.0, size=100000).astype(np.float32)
    
    # Complex processing configuration
    config = {
        'processors': {
            'normalize': {
                'scale': 0.5,
                'offset': -1.0,
                'use_lookup_table': False
            },
            'augment': {
                'scale': 1.2,
                'offset': 0.1,
                'use_lookup_table': True
            },
            'denoise': {
                'scale': 0.95,
                'offset': 0.05,
                'use_lookup_table': True
            }
        },
        'metadata': {
            'version': '2.1',
            'created_by': 'data_science_team',
            'purpose': 'production_preprocessing'
        }
    }
    
    processor = ConfigurableDataProcessor(config, lookup_table)
    
    print(f"✅ Created processor with {lookup_table.nbytes / (1024**2):.1f}MB lookup table")
    return processor


def safe_cache_retrieval(cache, experiment_id, **cache_keys):
    """
    Security-aware cache retrieval with validation.
    
    Returns None if validation fails, forcing object recreation.
    """
    try:
        # Attempt to retrieve cached object
        cached_obj = cache.get(experiment_id=experiment_id, **cache_keys)
        
        if cached_obj is None:
            return None
        
        # Validate type
        if not isinstance(cached_obj, ModelExperiment):
            print(f"⚠️ Type validation failed: expected ModelExperiment, got {type(cached_obj)}")
            return None
        
        # Validate experiment ID matches
        if cached_obj.experiment_id != experiment_id:
            print(f"⚠️ Experiment ID mismatch: cached={cached_obj.experiment_id}, expected={experiment_id}")
            return None
        
        # Run post-retrieval validation
        if not cached_obj.validate_post_retrieval():
            print("⚠️ Post-retrieval validation failed")
            return None
        
        print("✅ Cache retrieval and validation successful")
        return cached_obj
        
    except Exception as e:
        print(f"⚠️ Cache retrieval failed with exception: {e}")
        return None


def demonstrate_class_caching():
    """Demonstrate caching complex initialized classes with security awareness."""
    print("=" * 60)
    print("🚀 Dill Class Caching Demonstration (Security-Aware)")
    print("=" * 60)
    
    print("\n🚨 SECURITY WARNING:")
    print("This demo shows dill usage for educational purposes.")
    print("In production, consider disabling dill (enable_dill_fallback=False)")
    print("to eliminate code execution risks.")
    print("-" * 60)
    
    # Enable dill for complex object serialization WITH WARNING
    config = CacheConfig(
        cache_dir="./cache_dill_demo",
        enable_dill_fallback=True,  # ⚠️ Security risk in production
        default_ttl_hours=48
    )
    cache = cacheness(config)
    
    # === Example 1: Secure ML Experiment Caching ===
    print("\n📊 Example 1: Secure ML Experiment Results Caching")
    print("-" * 50)
    
    # Create and cache large experiment
    start_time = time.time()
    experiment = create_large_experiment()
    creation_time = time.time() - start_time
    
    print(f"⏱️  Experiment creation time: {creation_time:.2f}s")
    
    # Cache the entire experiment object
    cache_start = time.time()
    cache.put(experiment, 
              project="transformer_research",
              model="transformer_v3", 
              experiment_id=experiment.experiment_id)
    cache_time = time.time() - cache_start
    
    print(f"💾 Cache storage time: {cache_time:.2f}s")
    
    # SECURE RETRIEVAL with validation
    retrieve_start = time.time()
    cached_experiment = safe_cache_retrieval(
        cache,
        experiment.experiment_id,
        project="transformer_research",
        model="transformer_v3"
    )
    retrieve_time = time.time() - retrieve_start
    
    print(f"📥 Cache retrieval time: {retrieve_time:.2f}s")
    
    if cached_experiment is None:
        print("❌ Secure retrieval failed - would recreate object in production")
        return
    
    # Test that the cached object works correctly
    test_data = np.random.randn(100, 50)
    original_processed = experiment.data_processor(test_data)
    cached_processed = cached_experiment.data_processor(test_data)
    
    print(f"🔍 Processing functions identical: {np.allclose(original_processed, cached_processed)}")
    print(f"🎯 Model weights identical: {np.array_equal(experiment.model_weights, cached_experiment.model_weights)}")
    print(f"📈 Training history preserved: {experiment.training_history == cached_experiment.training_history}")
    
    # Evaluate model using cached experiment
    metrics = cached_experiment.evaluate_model(test_data)
    print(f"🧮 Evaluation metrics: {metrics}")
    
    # === Security Demonstration ===
    print("\n🔒 Security Validation Demonstration")
    print("-" * 50)
    
    # Show what happens when validation fails
    print("Testing validation failure scenarios:")
    
    # Test 1: Try to retrieve with wrong experiment ID
    print("1. Wrong experiment ID test:")
    wrong_id_result = safe_cache_retrieval(
        cache, "wrong_id", project="transformer_research", model="transformer_v3"
    )
    print(f"   Result: {'Failed as expected' if wrong_id_result is None else 'Unexpected success'}")
    
    # === Performance Summary ===
    print("\n📊 Performance Summary")
    print("-" * 50)
    
    cache_stats = cache.get_stats()
    print(f"📁 Total cache entries: {cache_stats['total_entries']}")
    print(f"💽 Total cache size: {cache_stats['total_size_mb']:.2f} MB")
    
    entries = cache.list_entries()
    for entry in entries:
        desc = entry.get('description', 'No description')
        size_mb = entry.get('size_mb', 0)
        metadata = entry.get('metadata', {})
        serializer = metadata.get('serializer', 'unknown')
        print(f"  - {desc}: {size_mb:.2f} MB (serializer: {serializer})")
    
    # Security audit
    print("\n🔍 Security Audit")
    print("-" * 50)
    audit_cache_security(cache)
    
    print("\n✨ Demonstration completed successfully!")
    print("🔑 Key Security Lessons:")
    print("   • Always validate cached objects after retrieval")
    print("   • Use version tracking for schema validation") 
    print("   • Implement post-retrieval functional testing")
    print("   • Consider disabling dill in production environments")
    print("   • Monitor cache for potential security issues")


def audit_cache_security(cache):
    """Simple security audit for cached objects."""
    entries = cache.list_entries()
    issues = []
    
    for entry in entries:
        metadata = entry.get('metadata', {})
        serializer = metadata.get('serializer', 'unknown')
        size_mb = entry.get('size_mb', 0)
        description = entry.get('description', '')
        
        if serializer == 'dill':
            if size_mb > 50:  # Large dill objects
                issues.append(f"Large dill object: {description} ({size_mb:.1f}MB)")
            
            if any(keyword in description.lower() for keyword in ['function', 'lambda', 'method']):
                issues.append(f"Cached function/method: {description}")
    
    if issues:
        print("⚠️ Potential security concerns found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n🛡️ Recommendations:")
        print("   - Validate all dill objects after retrieval")
        print("   - Consider moving to pickle-only for production")
        print("   - Implement cache integrity monitoring")
    else:
        print("✅ No obvious security issues detected")
    
    print(f"📈 Cache security score: {max(0, 100 - len(issues) * 20)}%")
    
    # === Example 2: Configurable Data Processor ===
    print("\n🔧 Example 2: Caching Configurable Data Processor")
    print("-" * 50)
    
    # Create and cache processor
    processor = create_data_processor()
    
    cache.put(processor,
              system="preprocessing_pipeline",
              version="v2.1",
              environment="production")
    
    # Retrieve processor
    cached_processor = cache.get(
        system="preprocessing_pipeline",
        version="v2.1", 
        environment="production"
    )
    
    # Verify we got the processor back
    assert cached_processor is not None, "Failed to retrieve cached processor"
    assert isinstance(cached_processor, ConfigurableDataProcessor), "Retrieved object is not a ConfigurableDataProcessor"
    
    # Test dynamic methods are preserved
    print("🧪 Testing dynamic processing methods:")
    for method_name in ['normalize', 'augment', 'denoise']:
        cached_method = getattr(cached_processor, f'apply_{method_name}')
        
        # Just test that methods exist and are callable
        method_preserved = callable(cached_method)
        print(f"  - apply_{method_name}: {'✅' if method_preserved else '❌'}")
    
    # Show processor info
    info = cached_processor.get_processor_info()
    print(f"📋 Processor info: {len(info['available_processors'])} methods available")
    print(f"🗂️  Lookup table size: {info['lookup_table_size']}")
    
    # === Performance Summary ===
    print("\n📊 Performance Summary")
    print("-" * 50)
    
    cache_stats = cache.get_stats()
    print(f"📁 Total cache entries: {cache_stats['total_entries']}")
    print(f"💽 Total cache size: {cache_stats['total_size_mb']:.2f} MB")
    
    entries = cache.list_entries()
    for entry in entries:
        desc = entry.get('description', 'No description')
        size_mb = entry.get('size_mb', 0)
        print(f"  - {desc}: {size_mb:.2f} MB")
    
    print("\n✨ Demonstration completed successfully!")
    print("🔑 Key Benefits:")
    print("   • Large objects with embedded logic cached seamlessly")
    print("   • Processing functions with closures preserved") 
    print("   • Complex configurations with dynamic methods work")
    print("   • Significant time savings on object recreation")


if __name__ == "__main__":
    try:
        demonstrate_class_caching()
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
