# Dill Integration Guide

Comprehensive guide to using dill serialization for advanced Python object caching.

## Overview

Cacheness includes integrated support for **dill**, a more powerful serialization library that extends beyond Python's standard `pickle` module. This enables caching of complex objects like functions, lambdas, closures, and other objects that standard pickle cannot handle.

> ⚠️ **CRITICAL SECURITY WARNING**: Dill serialization can execute arbitrary code during deserialization and may introduce serious bugs if class definitions change between cache storage and retrieval. **Never use dill with untrusted data sources.** See [Security Considerations](#security-considerations) for safe usage patterns.

## Key Benefits

- **Functions and Lambdas**: Cache function objects including closures with captured variables
- **Partial Functions**: Support for `functools.partial` and other functional programming constructs  
- **Complex Objects**: Handle objects with dynamic attributes, metaclasses, and other advanced features
- **Automatic Fallback**: Seamlessly falls back from pickle to dill when needed
- **Compressed Storage**: Maintains all compression benefits with Blosc2 integration

## ⚠️ Security Considerations

### Critical Risks

**🚨 Code Execution Risk:**
Dill can execute arbitrary code during deserialization. This means:
- Malicious cache files can compromise your entire application
- Modified cache files can inject code into your program
- Untrusted data sources can execute arbitrary commands

**🚨 Class Definition Changes:**
Cached objects may become incompatible if class definitions change:
- Method signature changes can cause crashes
- Removed attributes can cause AttributeError exceptions
- Changed logic can cause silent bugs with wrong results

**🚨 Environment Dependencies:**
Cached objects may not work across different environments:
- Different Python versions may cause compatibility issues
- Missing dependencies can cause import errors
- Different library versions may cause behavior changes

### Safe Usage Patterns

#### 1. Always Validate Cached Objects

```python
from typing import Optional, Type, TypeVar

T = TypeVar('T')

def safe_cache_retrieval(cache, expected_type: Type[T], **cache_keys) -> Optional[T]:
    """Safely retrieve and validate cached objects."""
    try:
        cached_obj = cache.get(**cache_keys)
        if cached_obj is None:
            return None
            
        # Type validation
        if not isinstance(cached_obj, expected_type):
            raise ValueError(f"Expected {expected_type.__name__}, got {type(cached_obj).__name__}")
        
        # Version validation (if your classes have versions)
        if hasattr(cached_obj, '_version') and hasattr(expected_type, '_version'):
            if cached_obj._version != expected_type._version:
                raise ValueError(f"Version mismatch: cached={cached_obj._version}, expected={expected_type._version}")
        
        # Functional validation - test critical methods
        if hasattr(cached_obj, 'validate'):
            cached_obj.validate()
        
        return cached_obj
        
    except Exception as e:
        print(f"⚠️ Cache validation failed: {e}")
        print("Falling back to object recreation...")
        return None

# Usage example
cached_model = safe_cache_retrieval(cache, ModelState, model="resnet", checkpoint="epoch_100")
if cached_model is None:
    # Recreate the object safely
    cached_model = create_model_state()
```

#### 2. Version Your Cached Classes

```python
@dataclass
class ModelState:
    _version: str = "2.1"  # Always include version
    _schema_hash: str = field(default="", init=False)
    weights: np.ndarray = None
    metadata: dict = None
    preprocessing_func: callable = None
    
    def __post_init__(self):
        # Validate version compatibility
        if self._version != "2.1":
            raise ValueError(f"Incompatible ModelState version: {self._version}")
        
        # Generate schema hash for additional validation
        import hashlib
        schema_str = f"{self._version}:{type(self.weights)}:{list(self.metadata.keys()) if self.metadata else []}"
        self._schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        
        # Initialize processing function
        scale_factor = self.metadata.get('scale_factor', 1.0)
        self.preprocessing_func = lambda x: x * scale_factor + np.random.normal(0, 0.01)
    
    def validate(self):
        """Validate object integrity after cache retrieval."""
        if self.weights is None:
            raise ValueError("Weights cannot be None")
        if not callable(self.preprocessing_func):
            raise ValueError("Preprocessing function is not callable")
        
        # Test function works
        test_input = np.array([[1.0, 2.0]])
        try:
            result = self.preprocessing_func(test_input)
            if result.shape != test_input.shape:
                raise ValueError("Preprocessing function returns wrong shape")
        except Exception as e:
            raise ValueError(f"Preprocessing function failed: {e}")
```

#### 3. Implement Cache Integrity Checks

```python
def cache_with_integrity_check(cache, obj, **cache_keys):
    """Cache object with integrity metadata."""
    import hashlib
    import pickle
    
    # Generate integrity hash
    obj_bytes = pickle.dumps(obj)
    integrity_hash = hashlib.sha256(obj_bytes).hexdigest()
    
    # Store object with metadata
    cache.put(obj, **cache_keys, _integrity_hash=integrity_hash)
    
    return integrity_hash

def retrieve_with_integrity_check(cache, **cache_keys):
    """Retrieve object and verify integrity."""
    import hashlib
    import pickle
    
    try:
        cached_obj = cache.get(**cache_keys)
        if cached_obj is None:
            return None
        
        # Get stored integrity hash
        stored_hash = cache_keys.get('_integrity_hash')
        if stored_hash:
            # Verify integrity
            obj_bytes = pickle.dumps(cached_obj)
            current_hash = hashlib.sha256(obj_bytes).hexdigest()
            
            if current_hash != stored_hash:
                raise ValueError("Cache integrity check failed - object may be corrupted")
        
        return cached_obj
        
    except Exception as e:
        print(f"⚠️ Integrity check failed: {e}")
        return None
```

#### 4. Production-Safe Configuration

```python
def create_production_safe_cache():
    """Create cache with security-focused configuration."""
    
    # For production: disable dill entirely for maximum security
    production_config = CacheConfig(
        enable_dill_fallback=False,  # Only allow standard pickle objects
        cache_dir="/secure/cache/path",
        metadata_backend="sqlite",
        verify_cache_integrity=True,
        max_cache_size_mb=1000  # Limit cache size
    )
    
    return cacheness(production_config)

def create_development_cache():
    """Create cache for development with dill enabled but with warnings."""
    
    config = CacheConfig(
        enable_dill_fallback=True,
        cache_dir="./dev_cache",
        metadata_backend="sqlite"
    )
    
    cache = cacheness(config)
    
    # Add warning for dill usage
    original_put = cache.put
    def warned_put(data, **kwargs):
        if hasattr(data, '__class__') and data.__class__.__module__ != 'builtins':
            print(f"⚠️ WARNING: Caching custom class {data.__class__.__name__} with dill")
            print("   Ensure class definition is stable and validate after retrieval")
        return original_put(data, **kwargs)
    
    cache.put = warned_put
    return cache
```

### Common Pitfalls and How to Avoid Them

#### Pitfall 1: Class Definition Changes

```python
# ❌ DANGEROUS: Changing class after caching
@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    # ... cache some instances ...

# Later, you change the class:
@dataclass 
class ModelConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32  # ❌ New field - cached instances won't have this!
    
# ✅ SAFE: Version your classes and handle migrations
@dataclass
class ModelConfig:
    _version: str = "2.0"
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    
    def __post_init__(self):
        if hasattr(self, '_version') and self._version == "1.0":
            # Handle migration from old version
            if not hasattr(self, 'batch_size'):
                self.batch_size = 32
            self._version = "2.0"
```

#### Pitfall 2: Environment Dependencies

```python
# ❌ DANGEROUS: Caching objects that depend on specific environments
class DataProcessor:
    def __init__(self):
        import some_optional_library  # ❌ May not exist in all environments
        self.processor = some_optional_library.create_processor()

# ✅ SAFE: Check dependencies and graceful fallbacks
class DataProcessor:
    def __init__(self):
        self._version = "1.0"
        try:
            import some_optional_library
            self.processor = some_optional_library.create_processor()
            self.has_advanced_processor = True
        except ImportError:
            print("⚠️ Optional library not available, using fallback")
            self.processor = self._create_fallback_processor()
            self.has_advanced_processor = False
    
    def validate(self):
        """Ensure processor is callable regardless of environment."""
        if not callable(self.processor):
            raise ValueError("Processor is not callable")
```

#### Pitfall 3: Mutable State in Closures

```python
# ❌ DANGEROUS: Mutable state captured in closures
def create_processor_bad():
    shared_state = {'counter': 0}  # ❌ Mutable state
    
    def process(data):
        shared_state['counter'] += 1  # ❌ State changes with each call
        return data * shared_state['counter']
    
    return process

# ✅ SAFE: Immutable state in closures
def create_processor_safe(multiplier=1.0):
    # Capture immutable values only
    def process(data):
        return data * multiplier  # ✅ No mutable state
    
    return process
```

## Configuration

### Basic Usage

```python
from cacheness import cacheness, CacheConfig

# Dill is enabled by default
cache = cacheness()

# Explicitly enable/disable dill fallback
config = CacheConfig(enable_dill_fallback=True)
cache = cacheness(config)
```

### Advanced Configuration

```python
from cacheness.config import HandlerConfig

config = CacheConfig(
    handlers=HandlerConfig(
        enable_dill_fallback=True,      # Enable dill for complex objects
        enable_object_pickle=True,      # Keep standard pickle for simple objects
    )
)
```

## Supported Object Types

### Classes with Complex Data (Most Common Use Case)

```python
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable

@dataclass 
class ExperimentResults:
    """ML experiment results with embedded processing logic."""
    model_weights: np.ndarray
    training_history: dict
    hyperparameters: dict
    data_processor: Callable
    
    def __post_init__(self):
        # Create processing function with captured hyperparameters
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        dropout_rate = self.hyperparameters.get('dropout_rate', 0.2)
        
        # This closure captures the hyperparameters
        def process_batch(data):
            # Apply learned preprocessing with experiment-specific parameters
            normalized = data / np.std(data) 
            return normalized * (1 - dropout_rate) + learning_rate
        
        self.data_processor = process_batch

# Create experiment with large dataset
experiment = ExperimentResults(
    model_weights=np.random.randn(10000, 1000),  # Large model weights
    training_history={'loss': [0.5, 0.3, 0.1], 'accuracy': [0.8, 0.9, 0.95]},
    hyperparameters={'learning_rate': 0.001, 'dropout_rate': 0.2, 'batch_size': 32},
    data_processor=None  # Will be set in __post_init__
)

# Cache entire experiment - weights, metadata, AND processing logic
cache.put(experiment, model="transformer", experiment="exp_42", date="2024-01-15")

# Retrieve months later - everything preserved including the closure
cached_experiment = cache.get(model="transformer", experiment="exp_42", date="2024-01-15")

# Use the cached processing function with original hyperparameters
test_data = np.random.randn(100, 50)
processed = cached_experiment.data_processor(test_data)
print(f"Weights shape: {cached_experiment.model_weights.shape}")
print(f"Final accuracy: {cached_experiment.training_history['accuracy'][-1]}")
```

### Configuration Objects with Dynamic Behavior

```python
class ModelConfig:
    """Configuration object with computed properties and methods."""
    
    def __init__(self, base_config: dict):
        self.base_config = base_config
        self._cached_transforms = {}
        
        # Create dynamic methods based on config
        for transform_name, params in base_config.get('transforms', {}).items():
            setattr(self, f"apply_{transform_name}", 
                   self._create_transform_method(params))
    
    def _create_transform_method(self, params):
        """Creates a closure-based transform method."""
        scale = params.get('scale', 1.0)
        offset = params.get('offset', 0.0)
        
        def transform(data):
            return data * scale + offset
        
        return transform

# Complex configuration with embedded logic
config_data = {
    'model_type': 'cnn',
    'transforms': {
        'normalize': {'scale': 0.5, 'offset': -1.0},
        'augment': {'scale': 1.2, 'offset': 0.1}
    },
    'training_params': {'epochs': 100, 'batch_size': 32}
}

model_config = ModelConfig(config_data)

# Cache the entire configuration object with all its dynamic methods
cache.put(model_config, project="image_classification", config_version="v2.1")

# Retrieve and use - all dynamic methods preserved
cached_config = cache.get(project="image_classification", config_version="v2.1")
normalized_data = cached_config.apply_normalize(raw_data)
augmented_data = cached_config.apply_augment(raw_data)
```

### Functions and Closures

While caching initialized classes is more common, dill also enables caching of functions:

```python
# Simple functions
def square(x):
    return x ** 2

cache.put(square, operation="square")
cached_func = cache.get(operation="square")
result = cached_func(5)  # Returns 25

# Closures with captured variables  
def create_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

multiplier = create_multiplier(2.5)
cache.put(multiplier, operation="multiply", factor=2.5)

# Lambda functions
processor = lambda x: x * 2 + 1
cache.put(processor, operation="lambda_process")
```

### Partial Functions

```python
from functools import partial
import operator

# Partial functions
multiply_by_10 = partial(operator.mul, 10)
add_5 = partial(operator.add, 5)

cache.put(multiply_by_10, operation="multiply", factor=10)
cache.put(add_5, operation="add", value=5)

# Retrieve and use
cached_multiply = cache.get(operation="multiply", factor=10)
result = cached_multiply(7)  # Returns 70
```

### Complex Function Decorators

```python
from cacheness import cached

# Cache function generators
@cached(ttl_hours=24)
def create_data_processor(mean=0, std=1):
    """Creates a data processing function with specific parameters."""
    import numpy as np
    
    def process_data(data):
        # Complex processing with captured parameters
        normalized = (data - mean) / std
        return normalized + np.random.normal(0, 0.01, data.shape)
    
    return process_data

# The returned function (with closure) is automatically cached
processor = create_data_processor(mean=5.0, std=2.0)
```

### Class Instances with Dynamic Attributes

```python
class ConfigurableProcessor:
    def __init__(self, config_dict):
        # Dynamically set attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def process(self, data):
        return data * self.multiplier + self.offset

# Complex configuration
config_data = {
    'multiplier': 2.5,
    'offset': 10,
    'debug': True,
    'processing_func': lambda x: x ** 2
}

processor = ConfigurableProcessor(config_data)
cache.put(processor, model="custom", version="v1")

# Retrieve maintains all dynamic attributes and embedded functions
cached_processor = cache.get(model="custom", version="v1")
```

## Performance Characteristics

### Serialization Performance

| Object Type | Pickle | Dill | Compression Ratio |
|-------------|---------|------|------------------|
| Simple functions | ❌ Fails | ✅ ~2ms | 60-70% |
| Lambdas | ❌ Fails | ✅ ~1ms | 50-60% |
| Closures | ❌ Fails | ✅ ~3ms | 65-75% |
| Partial functions | ❌ Fails | ✅ ~2ms | 55-65% |
| Simple objects | ✅ ~0.5ms | ✅ ~1ms | 70-80% |

### Storage Benefits

```python
# Dill objects still benefit from Blosc2 compression
import dill
import pickle

# Example function
def complex_func(x):
    import numpy as np
    return np.sin(x) * np.cos(x ** 2)

# Size comparison
dill_size = len(dill.dumps(complex_func))
compressed_size = cache.put(complex_func, key="function_test")

print(f"Original dill size: {dill_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compressed_size/dill_size:.1%}")
```

## Automatic Fallback Logic

Cacheness uses intelligent fallback logic:

1. **Try pickle first**: Standard objects use faster pickle serialization
2. **Fallback to dill**: Complex objects automatically use dill when pickle fails
3. **Preserve metadata**: Both methods maintain consistent metadata tracking

```python
# This happens automatically
cache.put("simple string", key="simple")     # Uses pickle
cache.put(lambda x: x**2, key="complex")     # Uses dill

# Metadata tracking shows which serializer was used
entries = cache.list_entries()
for entry in entries:
    metadata = entry.get('metadata', {})
    serializer = metadata.get('serializer', 'unknown')
    print(f"{entry.get('description')}: {serializer}")
```

## Error Handling

### When Dill is Disabled

```python
# Disable dill fallback
config = CacheConfig(enable_dill_fallback=False)
cache = cacheness(config)

# This will fail gracefully
try:
    cache.put(lambda x: x**2, key="lambda_test")
except ValueError as e:
    print(f"Error: {e}")  # "No handler available for data type: <class 'function'>"
```

### Debugging Serialization Issues

```python
from cacheness.compress_pickle import is_pickleable, is_dill_serializable

# Check what serialization method will be used
def test_function(x):
    return x * 2

print(f"Pickleable: {is_pickleable(test_function)}")           # False
print(f"Dill serializable: {is_dill_serializable(test_function)}")  # True

# For complex debugging
complex_object = SomeComplexClass()
if not is_pickleable(complex_object):
    if is_dill_serializable(complex_object):
        print("Will use dill serialization")
    else:
        print("Object cannot be serialized by either pickle or dill")
```

## Best Practices

### 1. Security First - Validate Everything

```python
# ✅ ALWAYS: Implement comprehensive validation
@dataclass
class SecureModelState:
    _version: str = "1.0"
    _schema_id: str = field(default="model_state_v1", init=False)
    weights: np.ndarray = None
    metadata: dict = None
    
    def __post_init__(self):
        self.validate_schema()
        self.validate_data()
    
    def validate_schema(self):
        """Validate class schema hasn't changed."""
        if self._version != "1.0":
            raise ValueError(f"Version mismatch: {self._version}")
        if self._schema_id != "model_state_v1":
            raise ValueError(f"Schema mismatch: {self._schema_id}")
    
    def validate_data(self):
        """Validate data integrity."""
        if self.weights is None or self.weights.size == 0:
            raise ValueError("Invalid weights")
        if not isinstance(self.metadata, dict):
            raise ValueError("Invalid metadata")

# ✅ ALWAYS: Use safe retrieval patterns
def get_model_safely(cache, **keys):
    try:
        model = cache.get(**keys)
        if model and isinstance(model, SecureModelState):
            model.validate_schema()
            model.validate_data()
            return model
    except Exception as e:
        print(f"⚠️ Cache retrieval failed: {e}")
    return None  # Force recreation
```

### 2. Prefer Class Caching Over Function Caching (With Caution)

```python
# ✅ BETTER: Cache data classes with validation, not raw functions
@dataclass
class ValidatedDataPipeline:
    _version: str = "1.0"
    training_data: np.ndarray = None
    model_config: dict = None
    
    def __post_init__(self):
        self._validate_and_create_processor()
    
    def _validate_and_create_processor(self):
        """Create processor with validation."""
        if not isinstance(self.model_config, dict):
            raise ValueError("Invalid model config")
        
        required_keys = ['scale', 'offset']
        if not all(key in self.model_config for key in required_keys):
            raise ValueError(f"Missing config keys: {required_keys}")
        
        # Create processor with validated parameters
        scale = float(self.model_config['scale'])
        offset = float(self.model_config['offset'])
        self.processor = lambda x: x * scale + offset
    
    def validate(self):
        """Post-retrieval validation."""
        if not hasattr(self, 'processor') or not callable(self.processor):
            raise ValueError("Processor not properly initialized")
        
        # Test processor works
        test_data = np.array([1.0])
        try:
            result = self.processor(test_data)
            if not isinstance(result, np.ndarray):
                raise ValueError("Processor returns wrong type")
        except Exception as e:
            raise ValueError(f"Processor validation failed: {e}")

# ❌ RISKY: Caching raw functions without validation
@cached()  # ❌ No validation, no version control
def risky_transform(x, scale, offset):
    return x * scale + offset  # What if this logic changes?
```

### 3. Be Mindful of Closure Scope and Security

```python
# Good: Minimal closure scope
def create_processor(factor):
    def process(data):
        return data * factor  # Only captures 'factor'
    return process

# Less ideal: Large closure scope
def create_processor_bad():
    large_data = load_huge_dataset()  # Will be captured in closure
    
    def process(data):
        return data * 2  # Doesn't even use large_data!
    return process
```

### 3. Validate Critical Functions

```python
# For critical cached functions, validate after retrieval
@cached(ttl_hours=48)
def create_model_predictor(model_params):
    # Complex model creation
    return trained_model.predict

predictor = create_model_predictor(params)

# Validate functionality
test_input = np.array([[1, 2, 3]])
try:
    result = predictor(test_input)
    print("✅ Cached predictor works correctly")
except Exception as e:
    print(f"❌ Cached predictor failed: {e}")
    # Fallback to recreating
```

### 4. Monitor Storage Usage

```python
# Check serialization method distribution
entries = cache.list_entries()
pickle_count = sum(1 for e in entries if e.get('metadata', {}).get('serializer') == 'pickle')
dill_count = sum(1 for e in entries if e.get('metadata', {}).get('serializer') == 'dill')

print(f"Pickle entries: {pickle_count}")
print(f"Dill entries: {dill_count}")
print(f"Dill usage ratio: {dill_count / (pickle_count + dill_count):.1%}")
```

### 5. Production Deployment Guidelines

#### Development vs Production Configuration

```python
# Development configuration - dill enabled with warnings
def create_dev_cache():
    config = CacheConfig(
        enable_dill_fallback=True,
        cache_dir="./dev_cache",
        metadata_backend="sqlite"
    )
    
    cache = cacheness(config)
    
    # Add dill usage warnings
    original_put = cache.put
    def warned_put(data, **kwargs):
        if not _is_safe_for_dill(data):
            print(f"⚠️ WARNING: Caching {type(data).__name__} with dill")
            print("   Ensure validation and error handling before production")
        return original_put(data, **kwargs)
    
    cache.put = warned_put
    return cache

# Production configuration - dill disabled or heavily restricted
def create_prod_cache():
    config = CacheConfig(
        enable_dill_fallback=False,  # ✅ Disable dill for security
        cache_dir="/secure/app/cache",
        metadata_backend="sqlite",
        verify_cache_integrity=True,
        max_cache_size_mb=2000
    )
    
    return cacheness(config)

def _is_safe_for_dill(obj):
    """Check if object is relatively safe for dill serialization."""
    # Allow basic types
    if isinstance(obj, (str, int, float, bool, list, dict, tuple)):
        return True
    
    # Allow numpy arrays
    if hasattr(obj, '__class__') and 'numpy' in str(type(obj)):
        return True
    
    # Warn for custom classes
    if hasattr(obj, '__class__') and obj.__class__.__module__ not in ['builtins', 'numpy']:
        return False
    
    return True
```

#### Cache Security Audit

```python
def audit_cache_security(cache):
    """Audit cache for potential security issues."""
    entries = cache.list_entries()
    issues = []
    
    for entry in entries:
        metadata = entry.get('metadata', {})
        serializer = metadata.get('serializer', 'unknown')
        
        if serializer == 'dill':
            # Check for potential security issues
            description = entry.get('description', '')
            size_mb = entry.get('size_mb', 0)
            
            if size_mb > 100:  # Large dill objects are riskier
                issues.append(f"Large dill object: {description} ({size_mb:.1f}MB)")
            
            if 'function' in description.lower() or 'lambda' in description.lower():
                issues.append(f"Cached function detected: {description}")
    
    if issues:
        print("🚨 Security audit found potential issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        print("  - Validate all cached objects after retrieval")
        print("  - Consider disabling dill in production")
        print("  - Implement cache integrity checks")
    else:
        print("✅ No obvious security issues found")
    
    return issues

# Run security audit
issues = audit_cache_security(cache)
```

## Limitations and Considerations

### 1. Platform Compatibility

- **Function serialization**: May not work across different Python versions
- **System dependencies**: Functions with system-specific imports may fail on different machines
- **Architecture**: Binary compatibility issues between different architectures

### 2. Security Considerations

```python
# Be careful with dill in production - it can execute arbitrary code
# Only cache functions from trusted sources

# For security-sensitive applications, consider disabling dill
production_config = CacheConfig(
    enable_dill_fallback=False,  # Only allow standard pickle objects
    # ... other secure settings
)
```

### 3. Performance Trade-offs

- **Startup time**: Dill import adds ~100ms to initial import
- **Serialization speed**: 2-3x slower than pickle for simple objects
- **Storage overhead**: Slightly larger serialized size for simple objects

## Troubleshooting

### Common Issues

**Functions not caching:**
```python
# Check if function is actually dill-serializable
from cacheness.compress_pickle import is_dill_serializable

def problematic_function():
    import some_module  # This might cause issues
    return some_module.complex_operation()

if not is_dill_serializable(problematic_function):
    print("Function cannot be serialized - check imports and dependencies")
```

**Cross-platform issues:**
```python
# For functions that need to work across platforms
def platform_safe_function(data):
    # Avoid system-specific operations
    import math  # Standard library only
    return math.sqrt(data)
```

**Memory usage with closures:**
```python
# Monitor memory usage of cached closures
import sys

def create_memory_efficient_closure(small_param):
    # Only capture what's needed
    def process(data):
        return data * small_param
    return process

# Check closure size
closure = create_memory_efficient_closure(2.5)
cache.put(closure, key="efficient_closure")

# Get file size from cache stats
entries = cache.list_entries()
for entry in entries:
    if "efficient_closure" in entry.get('description', ''):
        print(f"Closure cache size: {entry.get('size_mb', 0):.3f} MB")
```

## Migration Guide

### From Standard Pickle-Only Caching

```python
# Before: Only simple objects
old_cache = some_pickle_cache()
old_cache.put({"data": [1, 2, 3]}, key="simple")

# After: Same simple objects work, plus complex ones
new_cache = cacheness()  # Dill enabled by default
new_cache.put({"data": [1, 2, 3]}, key="simple")  # Still uses pickle
new_cache.put(lambda x: x**2, key="complex")       # Now works with dill
```

### Gradual Migration

```python
# Start with dill disabled to ensure compatibility
conservative_config = CacheConfig(enable_dill_fallback=False)
cache = cacheness(conservative_config)

# Test all existing functionality
run_existing_tests()

# Then enable dill for new use cases
enhanced_config = CacheConfig(enable_dill_fallback=True)
cache = cacheness(enhanced_config)
```

## Related Documentation

- **[Configuration Guide](CONFIGURATION.md)**: Complete configuration options
- **[Performance Guide](PERFORMANCE.md)**: Optimization strategies
- **[API Reference](API_REFERENCE.md)**: Full API documentation
