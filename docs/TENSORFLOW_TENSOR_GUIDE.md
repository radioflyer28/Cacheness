# TensorFlow Tensor Handling Guide

Comprehensive guide to caching TensorFlow tensors with optimized storage and performance.

## Overview

Cacheness provides native support for **TensorFlow tensors** with specialized handling that preserves tensor properties, optimizes storage through advanced compression, and maintains computational graph compatibility.

## Key Benefits

- **Native Tensor Support**: Direct tensor serialization without conversion overhead
- **Graph Compatibility**: Maintains tensor properties for seamless reintegration
- **Optimized Compression**: Blosc2 compression optimized for tensor data patterns
- **Memory Efficiency**: Avoids intermediate conversions and copies
- **Gradient Preservation**: Supports tensors with gradient information

## Configuration

### Enabling TensorFlow Support

```python
from cacheness import cacheness, CacheConfig
from cacheness.config import HandlerConfig

# TensorFlow support is disabled by default (due to import overhead)
config = CacheConfig(
    handlers=HandlerConfig(
        enable_tensorflow_tensors=True
    )
)

cache = cacheness(config)
```

### Import Overhead Considerations

```python
# TensorFlow imports add ~2-3 seconds to startup time
import time

start = time.time()
config = CacheConfig(enable_tensorflow_tensors=True)
cache = cacheness(config)
import_time = time.time() - start

print(f"Startup time with TensorFlow: {import_time:.2f}s")

# Consider lazy loading for production
class LazyTensorFlowCache:
    def __init__(self):
        self._cache = None
    
    @property
    def cache(self):
        if self._cache is None:
            config = CacheConfig(enable_tensorflow_tensors=True)
            self._cache = cacheness(config)
        return self._cache
```

## Supported Tensor Types

### Basic Tensors

```python
import tensorflow as tf

# Various tensor types
scalar_tensor = tf.constant(42.0)
vector_tensor = tf.constant([1.0, 2.0, 3.0])
matrix_tensor = tf.constant([[1, 2], [3, 4]])
high_dim_tensor = tf.random.normal((100, 50, 20))

# Cache all tensor types
cache.put(scalar_tensor, model="test", tensor="scalar")
cache.put(vector_tensor, model="test", tensor="vector")  
cache.put(matrix_tensor, model="test", tensor="matrix")
cache.put(high_dim_tensor, model="test", tensor="high_dim")

# Retrieve with preserved properties
cached_scalar = cache.get(model="test", tensor="scalar")
print(f"Original shape: {scalar_tensor.shape}")
print(f"Cached shape: {cached_scalar.shape}")
print(f"Types match: {type(scalar_tensor) == type(cached_scalar)}")
```

### Variables and Trainable Parameters

```python
# TensorFlow Variables (model parameters)
weight = tf.Variable(tf.random.normal((784, 128)), name="dense_weight")
bias = tf.Variable(tf.zeros((128,)), name="dense_bias")

# Cache model parameters
cache.put(weight, model="mlp", layer="dense1", param="weight")
cache.put(bias, model="mlp", layer="dense1", param="bias")

# Retrieve and verify trainability
cached_weight = cache.get(model="mlp", layer="dense1", param="weight")
cached_bias = cache.get(model="mlp", layer="dense1", param="bias")

print(f"Original trainable: {weight.trainable}")
print(f"Cached trainable: {cached_weight.trainable}")
```

### Different Data Types

```python
# Various tensor dtypes
float_tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
double_tensor = tf.constant([1.0, 2.0], dtype=tf.float64)
int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
bool_tensor = tf.constant([True, False, True], dtype=tf.bool)
complex_tensor = tf.constant([1+2j, 3+4j], dtype=tf.complex64)

# All dtypes are preserved
for i, tensor in enumerate([float_tensor, double_tensor, int_tensor, bool_tensor, complex_tensor]):
    cache.put(tensor, experiment="dtype_test", tensor_id=i)
    cached = cache.get(experiment="dtype_test", tensor_id=i)
    print(f"Dtype preserved: {tensor.dtype == cached.dtype}")
```

### Sparse Tensors

```python
# Sparse tensor support
indices = [[0, 0], [1, 2]]
values = [1.0, 2.0]
dense_shape = [3, 4]

sparse_tensor = tf.SparseTensor(
    indices=indices,
    values=values,
    dense_shape=dense_shape
)

cache.put(sparse_tensor, data="sparse", experiment="sparse_test")
cached_sparse = cache.get(data="sparse", experiment="sparse_test")

# Verify sparse properties preserved
print(f"Indices match: {tf.reduce_all(sparse_tensor.indices == cached_sparse.indices)}")
print(f"Values match: {tf.reduce_all(sparse_tensor.values == cached_sparse.values)}")
print(f"Shape match: {sparse_tensor.dense_shape == cached_sparse.dense_shape}")
```

## Model Component Caching

### Layer Weights

```python
# Cache individual layer weights
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Cache layer by layer
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'get_weights') and layer.get_weights():
        weights = layer.get_weights()
        for j, weight in enumerate(weights):
            weight_tensor = tf.constant(weight)
            cache.put(weight_tensor, model="mnist", layer=i, weight=j)

# Restore weights
def restore_model_weights(cache, model, model_name="mnist"):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = []
            for j in range(len(layer.get_weights())):
                try:
                    cached_weight = cache.get(model=model_name, layer=i, weight=j)
                    weights.append(cached_weight.numpy())
                except KeyError:
                    print(f"Warning: Weight not found for layer {i}, weight {j}")
                    break
            if len(weights) == len(layer.get_weights()):
                layer.set_weights(weights)
```

### Embeddings and Large Tensors

```python
# Large embedding matrices
vocab_size = 50000
embedding_dim = 300
embedding_matrix = tf.random.normal((vocab_size, embedding_dim))

# Check compression effectiveness
original_size = embedding_matrix.numpy().nbytes
cache.put(embedding_matrix, model="word2vec", component="embeddings")

# Get metadata to see compression ratio
entries = cache.list_entries()
for entry in entries:
    if "embeddings" in entry.get('description', ''):
        compressed_size = entry.get('size_mb', 0) * 1024 * 1024
        ratio = compressed_size / original_size
        print(f"Embedding compression ratio: {ratio:.1%}")
        print(f"Original: {original_size / 1024**2:.1f} MB")
        print(f"Compressed: {compressed_size / 1024**2:.1f} MB")
```

## Advanced Features

### Gradient Information

```python
# Tensors with gradient information
x = tf.Variable(tf.constant([1.0, 2.0, 3.0]))

with tf.GradientTape() as tape:
    y = tf.reduce_sum(x ** 2)

gradients = tape.gradient(y, x)

# Cache gradients
cache.put(gradients, computation="squared_sum", variable="x", step="grad")

# Retrieve and verify
cached_grad = cache.get(computation="squared_sum", variable="x", step="grad")
print(f"Gradients equal: {tf.reduce_all(gradients == cached_grad)}")
```

### Custom Tensor Operations

```python
# Complex tensor computations
@tf.function
def complex_computation(x):
    return tf.nn.softmax(tf.matmul(x, tf.transpose(x)))

input_tensor = tf.random.normal((100, 50))
result = complex_computation(input_tensor)

# Cache computation results
cache.put(result, computation="attention_weights", size="100x50")

# Verify computational properties
cached_result = cache.get(computation="attention_weights", size="100x50")
print(f"Result shape: {result.shape}")
print(f"Cached shape: {cached_result.shape}")
print(f"Sum to 1 (softmax): {tf.reduce_all(tf.abs(tf.reduce_sum(cached_result, axis=-1) - 1.0) < 1e-6)}")
```

## Performance Characteristics

### Compression Performance

| Tensor Type | Original Size | Compressed Size | Ratio | Compression Time |
|-------------|---------------|-----------------|-------|------------------|
| Dense Float32 | 100MB | 25-40MB | 25-40% | ~200ms |
| Dense Int32 | 100MB | 15-30MB | 15-30% | ~150ms |
| Sparse Tensors | 100MB | 5-15MB | 5-15% | ~100ms |
| Embedding Matrices | 500MB | 125-200MB | 25-40% | ~800ms |
| Gradient Tensors | 50MB | 20-35MB | 40-70% | ~100ms |

### Memory Usage Optimization

```python
# Memory-efficient tensor caching
def cache_large_tensor_efficiently(tensor, cache, **metadata):
    """Cache large tensors with memory optimization."""
    
    # Check if tensor is too large for memory
    tensor_size_mb = tensor.numpy().nbytes / (1024**2)
    
    if tensor_size_mb > 1000:  # 1GB threshold
        print(f"Warning: Large tensor ({tensor_size_mb:.1f}MB) - consider chunking")
        
        # Optional: Split into chunks
        if tensor.shape[0] > 1000:
            chunk_size = 1000
            for i in range(0, tensor.shape[0], chunk_size):
                chunk = tensor[i:i+chunk_size]
                chunk_metadata = {**metadata, 'chunk': i//chunk_size}
                cache.put(chunk, **chunk_metadata)
            return True
    
    # Regular caching for smaller tensors
    cache.put(tensor, **metadata)
    return True

# Usage
large_tensor = tf.random.normal((5000, 1000))
cache_large_tensor_efficiently(large_tensor, cache, model="large_model", layer="dense")
```

## Storage Optimization

### Compression Settings

```python
from cacheness.config import CompressionConfig

# Optimize compression for tensor data
tensor_optimized_config = CacheConfig(
    handlers=HandlerConfig(enable_tensorflow_tensors=True),
    compression=CompressionConfig(
        compression_threshold_bytes=1024*1024,  # 1MB threshold
        enable_parallel_compression=True,       # Use multiple cores
        compression_level=5                     # Balance speed vs ratio
    )
)

cache = cacheness(tensor_optimized_config)
```

### Data Type Considerations

```python
# Choose appropriate dtypes for storage efficiency
def optimize_tensor_dtype(tensor):
    """Optimize tensor dtype for storage."""
    
    if tensor.dtype == tf.float64:
        # Most ML applications don't need float64 precision
        return tf.cast(tensor, tf.float32)
    
    elif tensor.dtype == tf.int64 and tf.reduce_max(tensor) < 2**31:
        # Use int32 if values fit
        return tf.cast(tensor, tf.int32)
    
    return tensor

# Example usage
original_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
optimized_tensor = optimize_tensor_dtype(original_tensor)

cache.put(optimized_tensor, model="optimized", data="test")

# Check size difference
original_size = original_tensor.numpy().nbytes
optimized_size = optimized_tensor.numpy().nbytes
print(f"Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")
```

## Integration Patterns

### Training Loop Integration

```python
class CachedTrainingLoop:
    def __init__(self, cache, model):
        self.cache = cache
        self.model = model
        self.step = 0
    
    def train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            predictions = self.model(batch_x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Cache gradients and intermediate results
        if self.step % 100 == 0:  # Cache every 100 steps
            self.cache.put(predictions, epoch=self.step//1000, step=self.step, data="predictions")
            self.cache.put(loss, epoch=self.step//1000, step=self.step, data="loss")
            
            for i, grad in enumerate(gradients):
                if grad is not None:
                    self.cache.put(grad, epoch=self.step//1000, step=self.step, 
                                 layer=i, data="gradient")
        
        self.step += 1
        return loss

# Usage
trainer = CachedTrainingLoop(cache, model)
for batch in dataset:
    loss = trainer.train_step(batch[0], batch[1])
```

### Model Checkpoint Integration

```python
def save_model_tensors(model, cache, checkpoint_name):
    """Save all model tensors to cache."""
    
    saved_tensors = []
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights'):
            for weight_idx, weight in enumerate(layer.get_weights()):
                weight_tensor = tf.constant(weight)
                cache.put(weight_tensor, 
                         checkpoint=checkpoint_name,
                         layer=layer_idx, 
                         weight=weight_idx)
                
                saved_tensors.append({
                    'layer': layer_idx,
                    'weight': weight_idx,
                    'shape': weight.shape,
                    'dtype': str(weight.dtype)
                })
    
    # Save metadata about saved tensors
    import json
    metadata_path = f"/tmp/{checkpoint_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(saved_tensors, f, indent=2)
    
    return saved_tensors

def load_model_tensors(model, cache, checkpoint_name):
    """Load all model tensors from cache."""
    
    # Load metadata
    import json
    metadata_path = f"/tmp/{checkpoint_name}_metadata.json"
    with open(metadata_path, 'r') as f:
        saved_tensors = json.load(f)
    
    # Restore weights layer by layer
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'set_weights'):
            layer_weights = []
            
            # Find all weights for this layer
            layer_tensors = [t for t in saved_tensors if t['layer'] == layer_idx]
            layer_tensors.sort(key=lambda x: x['weight'])
            
            for tensor_info in layer_tensors:
                cached_tensor = cache.get(
                    checkpoint=checkpoint_name,
                    layer=layer_idx,
                    weight=tensor_info['weight']
                )
                layer_weights.append(cached_tensor.numpy())
            
            if layer_weights:
                layer.set_weights(layer_weights)

# Usage
save_model_tensors(model, cache, "best_model_epoch_50")
# ... later ...
load_model_tensors(new_model, cache, "best_model_epoch_50")
```

## Error Handling and Debugging

### Common Issues

**TensorFlow not available:**
```python
# Graceful degradation when TensorFlow isn't installed
try:
    config = CacheConfig(enable_tensorflow_tensors=True)
    cache = cacheness(config)
    print("✅ TensorFlow tensor support enabled")
except ImportError as e:
    print("⚠️  TensorFlow not available, using standard handlers")
    config = CacheConfig(enable_tensorflow_tensors=False)
    cache = cacheness(config)
```

**Memory issues with large tensors:**
```python
def safe_tensor_cache(tensor, cache, **metadata):
    """Safely cache tensors with memory checking."""
    
    try:
        # Check available memory
        import psutil
        available_memory = psutil.virtual_memory().available
        tensor_memory = tensor.numpy().nbytes
        
        if tensor_memory > available_memory * 0.5:  # Use max 50% of available memory
            raise MemoryError(f"Tensor too large: {tensor_memory/1024**3:.1f}GB")
        
        cache.put(tensor, **metadata)
        return True
        
    except MemoryError as e:
        print(f"Memory error caching tensor: {e}")
        return False
    except Exception as e:
        print(f"Error caching tensor: {e}")
        return False
```

**Version compatibility:**
```python
# Check TensorFlow version compatibility
import tensorflow as tf

def check_tf_compatibility():
    """Check if TensorFlow version is compatible."""
    
    version = tf.__version__
    major, minor = map(int, version.split('.')[:2])
    
    if major < 2:
        print("⚠️  TensorFlow 1.x detected - some features may not work")
        return False
    elif major == 2 and minor < 4:
        print("⚠️  TensorFlow 2.0-2.3 detected - consider upgrading")
        return True
    else:
        print("✅ TensorFlow version compatible")
        return True

# Check before enabling tensor support
if check_tf_compatibility():
    config = CacheConfig(enable_tensorflow_tensors=True)
```

## Best Practices

### 1. Selective Tensor Caching

```python
# Cache only expensive-to-compute tensors
def should_cache_tensor(tensor, computation_time_ms):
    """Decide whether a tensor should be cached."""
    
    size_mb = tensor.numpy().nbytes / (1024**2)
    
    # Cache if computation is expensive OR tensor is large
    if computation_time_ms > 1000 or size_mb > 100:
        return True
    
    # Don't cache small, quickly computed tensors
    return False

# Example usage
import time

start = time.time()
result = expensive_tensor_operation(input_data)
computation_time = (time.time() - start) * 1000

if should_cache_tensor(result, computation_time):
    cache.put(result, operation="expensive_op", input_hash=hash(str(input_data)))
```

### 2. Memory Management

```python
# Clean up tensors after caching
def cache_and_cleanup(tensor, cache, **metadata):
    """Cache tensor and clean up references."""
    
    cache.put(tensor, **metadata)
    
    # Clear the tensor reference to free memory
    del tensor
    
    # Force garbage collection for large tensors
    import gc
    gc.collect()

# Usage for large tensor processing
large_result = process_large_tensor(input_data)
cache_and_cleanup(large_result, cache, model="large_process", step=1)
```

### 3. Validation After Retrieval

```python
def validate_cached_tensor(original, cached, tolerance=1e-6):
    """Validate that cached tensor matches original."""
    
    # Shape check
    if original.shape != cached.shape:
        return False, f"Shape mismatch: {original.shape} vs {cached.shape}"
    
    # Dtype check
    if original.dtype != cached.dtype:
        return False, f"Dtype mismatch: {original.dtype} vs {cached.dtype}"
    
    # Value check
    if not tf.reduce_all(tf.abs(original - cached) < tolerance):
        max_diff = tf.reduce_max(tf.abs(original - cached))
        return False, f"Value mismatch: max difference {max_diff}"
    
    return True, "Tensors match"

# Usage
original_tensor = tf.random.normal((100, 100))
cache.put(original_tensor, test="validation")
cached_tensor = cache.get(test="validation")

valid, message = validate_cached_tensor(original_tensor, cached_tensor)
print(f"Validation: {message}")
```

## Migration Guide

### From NumPy Array Caching

```python
# Before: Converting tensors to numpy for caching
import numpy as np

tensor = tf.random.normal((100, 100))
numpy_array = tensor.numpy()  # Conversion overhead
old_cache.put(numpy_array, key="tensor_data")

# Retrieval required conversion back
cached_array = old_cache.get(key="tensor_data")
restored_tensor = tf.constant(cached_array)  # Another conversion

# After: Direct tensor caching
config = CacheConfig(enable_tensorflow_tensors=True)
new_cache = cacheness(config)

new_cache.put(tensor, key="tensor_data")  # No conversion
restored_tensor = new_cache.get(key="tensor_data")  # Direct tensor
```

### Gradual Adoption

```python
class HybridTensorCache:
    """Gradually migrate to tensor caching."""
    
    def __init__(self, enable_tensors=False):
        config = CacheConfig(enable_tensorflow_tensors=enable_tensors)
        self.cache = cacheness(config)
        self.enable_tensors = enable_tensors
    
    def put_tensor(self, tensor, **metadata):
        if self.enable_tensors:
            # New path: direct tensor caching
            self.cache.put(tensor, **metadata)
        else:
            # Old path: numpy conversion
            self.cache.put(tensor.numpy(), **metadata)
    
    def get_tensor(self, **metadata):
        result = self.cache.get(**metadata)
        
        if self.enable_tensors:
            return result  # Already a tensor
        else:
            return tf.constant(result)  # Convert from numpy

# Start conservative, then enable
hybrid_cache = HybridTensorCache(enable_tensors=False)
# ... test existing functionality ...
hybrid_cache = HybridTensorCache(enable_tensors=True)
```

## Related Documentation

- **[Configuration Guide](CONFIGURATION.md)**: TensorFlow handler configuration
- **[Performance Guide](PERFORMANCE.md)**: Tensor-specific optimizations  
- **[API Reference](API_REFERENCE.md)**: Complete tensor handling API
