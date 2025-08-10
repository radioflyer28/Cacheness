# cacheness

A high-performance disk caching library for Python with key-based hashing, automatic compression with Blosc2, and decorator support.  
It supports directly caching numpy arrays, Pandas and Polars DataFrames/Series via flexible backends. 
For everything else, will use pickle and Blosc compression.

Motivation... I wanted a very fast cache system to speed up ML model development when doing things like large API calls and persisting data intensive stages like building derived features.

It requires Python>=3.11 (due to numpy v2.x) and works with Python 3.13.

It's made for dataframes as well, however, that's totally optional. So, if all you want to do is cache random Python objects and/or numpy arrays then install as is. However, I do recommend installing at least **sqlalchemy** for the SQLite metadata backend which is highly beneficial for performance and reliability.

*Note: this library is 100% vibe-coded, so you know what that means... :)*

**Key Features:**
- **Key-based caching** using xxhash (XXH3_64) for fast, deterministic cache keys
- **Intelligent path hashing** with automatic content-based invalidation and parallel directory processing
- **Function decorators** for automatic caching with `@cached`
- **Multi-format storage** with automatic type detection and optimal format selection
- **Advanced compression** using Blosc2 and LZ4 for space efficiency
- **Multiple backends** with SQLite and JSON metadata support
- **Performance-optimized** parallel processing for large directories (4GB+ or 80+ files)

**Requirements:**
- Python >= 3.11
- **xxhash**: Fast XXH3_64 hashing for cache keys
- **numpy**: Array handling and storage
- **blosc2**: High-performance array compression

**Optional Dependencies:**
- **sqlalchemy**: SQLite metadata backend (10-500x faster than JSON for large caches)
- **orjson**: High-performance JSON serialization (1.5-5x faster than built-in json)
- **pandas/polars**: DataFrame and Series support with Parquet storage

## Quick Start

### Basic Usage with Key-Based Caching

```python
from cacheness import cacheness

# Create a cache instance
cache = cacheness()

# Store data using keyword arguments as cache keys
cache.put({"results": [1, 2, 3]}, 
          model="xgboost", 
          dataset="customer_data", 
          version="1.0")

# Retrieve data using the same key parameters
data = cache.get(model="xgboost", dataset="customer_data", version="1.0")
print(data)  # {"results": [1, 2, 3]}

# Store numpy arrays
import numpy as np
features = np.random.rand(1000, 50)
cache.put(features, 
          description="Feature embeddings",
          dataset="training", 
          preprocessing="standard_scaled")

# Retrieve arrays
cached_features = cache.get(dataset="training", preprocessing="standard_scaled")
```

### Function Decorators for Automatic Caching

```python
from cacheness import cached
import time

# Basic function caching
@cached()
def expensive_computation(n):
    """This function will be automatically cached."""
    time.sleep(2)  # Simulate expensive work
    return n ** 2

# First call takes 2 seconds
result1 = expensive_computation(5)  # Computed and cached

# Second call returns instantly from cache
result2 = expensive_computation(5)  # Retrieved from cache

# Custom TTL and key prefix
@cached(ttl_hours=6, key_prefix="ml_model_v2")
def train_model(data, learning_rate=0.01, epochs=100):
    """Train a model with caching."""
    # Expensive model training here
    return {"accuracy": 0.95, "model_params": {...}}

# Cache with different parameters
model1 = train_model(data, learning_rate=0.01)  # Cached
model2 = train_model(data, learning_rate=0.02)  # Different cache entry
model3 = train_model(data, learning_rate=0.01)  # Retrieved from cache

# Multiple return values (cached as single tuple)
@cached(ttl_hours=24)
def process_dataset(data_path):
    """Process dataset returning multiple objects."""
    df_train = load_and_clean(f"{data_path}/train.csv")
    df_test = load_and_clean(f"{data_path}/test.csv") 
    features = extract_features(df_train)
    metadata = {"processed_at": datetime.now(), "version": "1.0"}
    
    # Entire tuple cached as single unit (atomic consistency)
    return df_train, df_test, features, metadata

# Access cache management methods
print(train_model.cache_info())  # Cache configuration info
train_model.cache_clear()        # Clear function's cache entries
```

#### Design Note: Single Return Object

The `@cached` decorator caches the **entire function return** as a single object, even for multiple return values (tuples). This design provides:

- **Atomic consistency**: All components cached/retrieved together
- **Simplicity**: No need to specify individual object types
- **Performance**: Single cache operation, excellent compression
- **Reliability**: No partial cache misses or synchronization issues

For **individual object caching with specialized handlers** (DataFrame → Parquet, arrays → NPZ), use manual caching:

```python
# Manual approach for specialized storage formats
@cached()  # Cache the coordination/metadata
def process_dataset_manual(data_path):
    cache = cacheness()
    
    # Check if already processed
    metadata = cache.get(dataset=data_path, component="metadata", version="1.0")
    if metadata:
        # Load individual components with optimal formats
        df_train = cache.get(dataset=data_path, component="train", version="1.0")
        df_test = cache.get(dataset=data_path, component="test", version="1.0") 
        features = cache.get(dataset=data_path, component="features", version="1.0")
        return df_train, df_test, features, metadata
    
    # Process and cache individually
    df_train = expensive_processing(f"{data_path}/train.csv")
    df_test = expensive_processing(f"{data_path}/test.csv")
    features = extract_features(df_train) 
    metadata = {"processed_at": datetime.now(), "version": "1.0"}
    
    # Store each with specialized handlers
    cache.put(df_train, dataset=data_path, component="train", version="1.0")      # → Parquet
    cache.put(df_test, dataset=data_path, component="test", version="1.0")        # → Parquet  
    cache.put(features, dataset=data_path, component="features", version="1.0")   # → NPZ
    cache.put(metadata, dataset=data_path, component="metadata", version="1.0")   # → Pickle
    
    return df_train, df_test, features, metadata
```

## Installation

```bash
# Basic installation with core dependencies
pip install cacheness

# Recommended installation (includes SQLAlchemy + orjson performance optimizations)
pip install cacheness[recommended]

# With SQLite backend support (recommended)
pip install cacheness[sql]

# With DataFrame support (Pandas/Polars)
pip install cacheness[dataframes]

# Full installation with all optional dependencies
pip install cacheness[recommended,dataframes]
```

### Dependency Groups

- **`recommended`**: SQLAlchemy (SQLite backend) + orjson (JSON performance) + Pandas + PyArrow
- **`sql`**: SQLAlchemy for high-performance metadata backend
- **`dataframes`**: Pandas, Polars, and PyArrow for DataFrame/Series support
- **`dev`**: Development dependencies (pytest, ruff, etc.)

## Key-Based Caching System

### How Cache Keys Work

The library uses **xxhash (XXH3_64)** to generate deterministic cache keys from your parameters:

```python
# These are equivalent - parameter order doesn't matter
cache.put(data, model="xgboost", dataset="train", version="1.0")
cache.put(data, version="1.0", model="xgboost", dataset="train")  # Same cache key

# Different parameters = different cache entries
cache.put(data1, model="xgboost", dataset="train")  # Key: abc123...
cache.put(data2, model="lightgbm", dataset="train") # Key: def456...
cache.put(data3, model="xgboost", dataset="test")   # Key: ghi789...
```

### Parameter-Based Retrieval

```python
# Store with multiple identifying parameters
cache.put(
    processed_data,
    experiment="housing_prediction",
    model_type="random_forest",
    feature_set="v2",
    preprocessing="minmax_scaled",
    description="Final model features"
)

# Retrieve using the same parameters
features = cache.get(
    experiment="housing_prediction",
    model_type="random_forest", 
    feature_set="v2",
    preprocessing="minmax_scaled"
)
```

## Unified Serialization Strategy

### Quality-First Object Serialization

The library uses a sophisticated **unified serialization strategy** that prioritizes **semantic meaning and introspection quality** over raw performance, while still maintaining excellent speed through intelligent optimizations.

Both the `UnifiedCache` and `@cached` decorators use the same serialization approach, ensuring **100% consistency** across all caching methods.

#### Intelligent Ordering Strategy

The serialization follows a carefully optimized 6-tier hierarchy:

```python
# Serialization Priority Order (Quality-First Approach):
1. Basic immutable types     → Direct representation
2. Special cases            → Custom handling (NumPy arrays, etc.)  
3. Collections             → Recursive introspection
4. Objects with __dict__   → Full introspection
5. Hashable objects        → Performance fallback
6. String representation   → Last resort
```

#### Example Output Comparison

**Quality-First Approach (Current)**:
```python
# Small tuple - full introspection for better cache keys
(1, "hello", 3.0) → "tuple:[int:1,str:hello,float:3.0]"

# Custom object - meaningful introspection
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("Alice", 30)
# Result: "User:dict:[str:age:int:30,str:name:str:Alice]"

# NumPy array - content-aware hashing
array = np.array([1, 2, 3])
# Result: "array:(3,):int64:a9b2c3d4e5f6789a"
```

**Performance-First Alternative (Not Used)**:
```python
# Would sacrifice meaning for speed:
(1, "hello", 3.0) → "hashed:tuple:-9217304902224717415"
user = User("Alice", 30) → "hashed:User:42"
```

#### Detailed Serialization Behavior

**1. Basic Immutable Types** *(Ultra-fast: ~0.10-0.14 μs)*
```python
42 → "int:42"
"hello" → "str:hello" 
3.14 → "float:3.14"
True → "bool:True"
None → "None"
```

**2. Special Cases with Custom Handling** *(Optimized: ~1.67 μs)*
```python
# NumPy arrays - content-aware with shape/dtype info
np.array([[1, 2], [3, 4]]) → "array:(2, 2):int64:a1b2c3d4e5f67890"

# TODO: Future special cases
# pathlib.Path objects - content hashing
# pandas.DataFrame - schema + content hash
```

**3. Collections - Recursive Introspection** *(Good performance: ~0.59-0.89 μs)*
```python
# Lists - full recursive serialization
[1, 2, "three"] → "list:[int:1,int:2,str:three]"

# Dictionaries - sorted keys for determinism
{"b": 2, "a": 1} → "dict:[str:a:int:1,str:b:int:2]"

# Sets - sorted for deterministic ordering
{3, 1, 2} → "set:[int:1,int:2,int:3]"
```

**4. Objects with `__dict__` - Full Introspection** *(Quality-focused: ~0.81 μs)*
```python
class Config:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 100

config = Config()
# Result: "Config:dict:[str:epochs:int:100,str:learning_rate:float:0.01]"
```

**5. Smart Tuple Handling** *(Adaptive performance)*
```python
# Small tuples (≤10 elements) - full introspection
(1, 2, 3) → "tuple:[int:1,int:2,int:3]"

# Large tuples (>10 elements) - performance fallback
tuple(range(20)) → "hashed:tuple:-1234567890123456"
```

**6. Hashable Objects - Performance Fallback** *(Fast: ~0.31 μs)*
```python
# Objects without useful __dict__ (e.g., __slots__)
class HashableToken:
    __slots__ = ['value']
    def __init__(self, value):
        self.value = value
    def __hash__(self):
        return hash(self.value)

token = HashableToken("abc123")
# Result: "hashed:HashableToken:42"
```

**7. Final Fallback - String Representation**
```python
# Non-hashable objects without __dict__
class CustomObj:
    def __str__(self):
        return "CustomObj(data)"

obj = CustomObj()
# Result: "CustomObj:CustomObj(data)"
```

#### Performance Characteristics

| Object Type | Avg Time (μs) | Cache Key Quality | Use Case |
|-------------|---------------|-------------------|----------|
| Basic types | 0.10-0.14 | ⭐⭐⭐⭐⭐ | Primitives, fast path |
| Collections | 0.59-0.89 | ⭐⭐⭐⭐⭐ | Lists, dicts, sets |
| Objects w/ `__dict__` | 0.81 | ⭐⭐⭐⭐⭐ | Custom classes |
| Large tuples | 0.34 | ⭐⭐⭐ | Performance fallback |
| Hashable objects | 0.31 | ⭐⭐⭐ | `__slots__` objects |
| NumPy arrays | 1.67 | ⭐⭐⭐⭐⭐ | Scientific data |

#### Benefits of Quality-First Approach

**🎯 Better Cache Key Semantics**:
- **Meaningful keys**: `tuple:[int:1,str:data,float:3.14]` vs `hashed:tuple:42`
- **Debugging-friendly**: Clear understanding of what's cached
- **Introspection**: Full visibility into object structure

**🔍 Superior Cache Invalidation**:
- **Content-based**: Changes to object structure are detected
- **Precise**: Different objects with same hash don't collide
- **Deterministic**: Same content always produces same key

**⚡ Smart Performance Optimizations**:
- **Adaptive thresholds**: Large collections use hash fallback
- **Type-specific handling**: Each type gets optimal treatment  
- **Minimal overhead**: Quality improvements with <1 μs average cost

#### Real-World Impact Examples

**Machine Learning Pipeline**:
```python
# Model parameters get full introspection
params = {
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_estimators": 100
}

# Quality-first result (excellent debugging):
# "dict:[str:learning_rate:float:0.01,str:max_depth:int:6,str:n_estimators:int:100]"

# vs Performance-first (poor debugging):
# "hashed:dict:1234567890"

@cached()
def train_model(X, y, params):
    # Cache key includes full parameter details!
    return model
```

**Data Processing**:
```python
# Processing configuration
config = ProcessingConfig(
    normalize=True,
    remove_outliers="iqr",
    feature_selection=["age", "income", "score"]
)

# Quality-first approach captures full config:
# "ProcessingConfig:dict:[str:feature_selection:list:[str:age,str:income,str:score],
#  str:normalize:bool:True,str:remove_outliers:str:iqr]"

# Perfect for cache invalidation when config changes!
```

#### Migration and Consistency

**Before (Inconsistent)**:
- `UnifiedCache` used JSON + str fallback
- `@cached` decorators used custom `_serialize_for_hash()`
- Different objects could have different cache keys

**After (Unified)**:
- Single `serialize_for_cache_key()` function used everywhere
- Guaranteed consistency between caching methods
- Comprehensive test coverage ensures reliability

#### Configuration and Customization

The serialization strategy is optimized by default, but you can understand its behavior:

```python
from cacheness.serialization import serialize_for_cache_key

# Test how your objects will be serialized
test_object = {"model": "xgboost", "params": {"max_depth": 6}}
cache_key = serialize_for_cache_key(test_object)
print(f"Cache key: {cache_key}")
# Output: dict:[str:model:str:xgboost,str:params:dict:[str:max_depth:int:6]]

# The unified approach ensures this key is identical whether using:
cache.put(data, config=test_object)  # UnifiedCache
# or
@cached()
def process(config=test_object): pass  # Decorator
```

This unified serialization approach ensures your cache keys are **meaningful, consistent, and debuggable** while maintaining excellent performance across all usage patterns.

## Advanced Configuration

```python
from cacheness import cacheness, CacheConfig

# Custom configuration
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",          # "sqlite", "json", or "auto"
    default_ttl_hours=48,              # Default TTL for entries
    max_cache_size_mb=5000,            # Maximum cache size
    cleanup_on_init=True,              # Clean expired entries on startup
    use_blosc2_arrays=True,            # High-performance array compression
    pickle_compression_codec="zstd"     # Compression for Python objects
)

cache = cacheness(config)
```

## Path Content Hashing & Parallel Processing

The library features intelligent `pathlib.Path` handling with automatic content hashing and performance-optimized parallel processing for large directories.

### Automatic Path Content Hashing (Default: `hash_path_content=True`)

By default, the library hashes `Path` objects by their actual file content rather than just the filename, providing robust cache invalidation and content integrity.

```python
from pathlib import Path

config = CacheConfig(hash_path_content=True)  # Default behavior
cache = cacheness(config)

# Create files with identical content but different paths
file1 = Path("data/train.csv")
file2 = Path("backup/data/train.csv")  # Same content, different location

# Cache with first file
cache.put(processed_data, description="Training data", input_file=file1)

# Can retrieve with second file (same content)
result = cache.get(input_file=file2)  # ✓ Cache hit - same content hash
```

**Benefits:**
- **Content Integrity**: Cache automatically invalidates when file content changes
- **Location Independence**: Moving files doesn't break cache (same content = same cache key)
- **Robust Workflows**: Perfect for data pipelines where files may be reorganized

### Intelligent Parallel Directory Processing

For directory paths, the library automatically uses optimized parallel processing when beneficial, with smart thresholds based on empirical performance testing.

```python
from pathlib import Path

# Large directory with many files
large_dataset_dir = Path("./datasets/images/")  # 500+ files, 5GB+

# Automatically uses parallel processing for performance
cache.put(
    processed_features,
    dataset_path=large_dataset_dir,  # Parallel hashing automatically applied
    model="resnet50",
    preprocessing="augmented"
)

# Small directory 
small_config_dir = Path("./config/")  # 10 files, 50MB

# Automatically uses sequential processing (faster for small dirs)
cache.put(
    config_data,
    config_path=small_config_dir,  # Sequential hashing for efficiency
    version="v1.2"
)
```

**Performance Characteristics:**

| Directory Size | File Count | Processing Method | Typical Performance |
|---------------|------------|-------------------|-------------------|
| < 4GB | < 80 files | Sequential | 0.001-0.2s (very fast) |
| ≥ 4GB OR ≥ 80 files | Any | Parallel | 1.2x-1.6x speedup |

**Automatic Threshold Selection:**
- **Sequential**: Directories under 4GB total size AND under 80 files
- **Parallel**: Directories over 4GB total size OR over 80 files
- **Empirically Optimized**: Thresholds based on comprehensive benchmarking across different system configurations

### Directory Content Hashing Examples

```python
# ML Dataset with many files
dataset_path = Path("./training_data/")
# Contains: 1000 image files, 2GB total
# → Automatically uses parallel processing

cache.put(
    trained_model,
    training_data=dataset_path,  # Hashes all 1000 files in parallel
    model_architecture="cnn",
    epochs=50
)

# Configuration directory
config_path = Path("./app_config/")  
# Contains: 5 JSON files, 10MB total
# → Uses fast sequential processing

cache.put(
    application_state,
    config_dir=config_path,  # Hashes 5 files sequentially
    environment="production"
)

# Mixed content processing
project_path = Path("./ml_project/")
# Contains: code, data, models - 500 files, 8GB total
# → Automatically uses parallel processing for optimal performance

cache.put(
    experiment_results,
    project_directory=project_path,  # Parallel processing
    experiment_id="housing_prediction_v3"
)
```

### Content vs Filename Hashing Comparison

```python
# Content-based hashing (default - recommended)
config = CacheConfig(hash_path_content=True)
cache = cacheness(config)

file_v1 = Path("data/dataset.csv")
file_v2 = Path("data/dataset_v2.csv")  # Same content, different name

cache.put(results, input_data=file_v1)
# Later, file is renamed but content unchanged
cached_results = cache.get(input_data=file_v2)  # ✓ Cache hit

# Filename-based hashing (faster but less robust)
config = CacheConfig(hash_path_content=False)
cache = cacheness(config)

cache.put(results, input_data=file_v1)
cached_results = cache.get(input_data=file_v2)  # ✗ Cache miss (different names)
cached_results = cache.get(input_data=file_v1)  # ✓ Cache hit (same name)
```

### Real-World Performance Examples

```python
from pathlib import Path
import time

# Large image dataset processing
image_dir = Path("./datasets/imagenet_subset/")  # 10,000 images, 15GB
start = time.time()

cache.put(
    feature_embeddings,
    dataset=image_dir,  # Parallel hashing: ~2.1s
    model="efficientnet_b0",
    input_size=224
)

print(f"Large directory cached in {time.time() - start:.2f}s")
# Output: Large directory cached in 2.13s (with 1.3x speedup from parallel)

# Small configuration
config_dir = Path("./model_configs/")  # 15 YAML files, 2MB
start = time.time()

cache.put(
    model_settings,
    config_path=config_dir,  # Sequential hashing: ~0.003s
    version="v2.1"
)

print(f"Small directory cached in {time.time() - start:.3f}s")
# Output: Small directory cached in 0.003s
```

### Advanced Path Hashing Configuration

```python
# Custom configuration for specific use cases
config = CacheConfig(
    hash_path_content=True,           # Enable content hashing
    cache_dir="./path_aware_cache",
    default_ttl_hours=48,
    # Parallel processing automatically optimized
)

cache = cacheness(config)

# Monitor path hashing performance
import logging
logging.basicConfig(level=logging.DEBUG)

# Path operations will log performance details:
# DEBUG: Hashing directory /path/to/data with 150 files (750MB) using parallel method
# DEBUG: Directory hashing completed in 0.89s using 4 workers
```

### Use Cases & Best Practices

**Ideal for Content Hashing:**
- **Data Science Pipelines**: Datasets that may be moved or reorganized
- **Model Training**: Training data that changes content over time
- **Configuration Management**: Settings files that are version controlled
- **Document Processing**: Content that matters more than filename

**Performance Optimization:**
- **Large Datasets**: Automatically benefits from parallel processing (4GB+ or 80+ files)
- **Small Configs**: Fast sequential processing for efficiency
- **Mixed Workloads**: Intelligent automatic selection based on directory characteristics
- **Cache Invalidation**: Automatic when file content changes

**Configuration Recommendations:**
```python
# Production data pipelines (recommended)
config = CacheConfig(
    hash_path_content=True,    # Robust content-based caching
    default_ttl_hours=168,     # 1 week for stable datasets
    max_cache_size_mb=10000    # Large cache for big datasets
)

# Development/testing (faster)
config = CacheConfig(
    hash_path_content=False,   # Filename-based for speed
    default_ttl_hours=24,      # Shorter TTL for changing data
    cleanup_on_init=True       # Regular cleanup
)
```

## Decorator Advanced Usage

### Custom Cache Instances

```python
from cacheness import cached, CacheConfig, UnifiedCache

# Create a specialized cache for ML models
ml_config = CacheConfig(
    cache_dir="./ml_cache",
    default_ttl_hours=168,  # 1 week
    max_cache_size_mb=10000
)
ml_cache = UnifiedCache(ml_config)

@cached(cache_instance=ml_cache, key_prefix="production")
def train_production_model(data, hyperparams):
    # Expensive model training
    return trained_model

# Different cache for temporary experiments
@cached(ttl_hours=2, key_prefix="experiment")
def quick_experiment(data, params):
    return experiment_results
```

### Custom Key Generation

```python
@cached(key_func=lambda func, args, kwargs: f"user_{kwargs['user_id']}_model")
def generate_user_model(user_id, preferences):
    """Cache per-user models with custom keys."""
    return create_personalized_model(user_id, preferences)

# Context manager for temporary caching
from cacheness import CacheContext

with CacheContext(ttl_hours=1, cache_dir="./temp_cache") as temp_cache:
    @temp_cache.cached()
    def temporary_computation():
        return expensive_computation()
```

## Data Type Support & Examples

### Python Objects with Key-Based Storage

```python
# Store configuration objects
config = {
    "model_params": {"learning_rate": 0.01, "max_depth": 6},
    "data_params": {"test_size": 0.2, "random_state": 42}
}
cache.put(config, 
          experiment="housing_ml", 
          config_version="v1.2", 
          author="researcher")

# Retrieve configuration
saved_config = cache.get(experiment="housing_ml", config_version="v1.2", author="researcher")
```

### NumPy Arrays with Compression

```python
import numpy as np

# Single arrays
embeddings = np.random.rand(10000, 768)
cache.put(embeddings, 
          model="bert_base", 
          dataset="wikipedia", 
          layer="last_hidden")

# Dictionary of arrays (common in ML)
ml_data = {
    "X_train": np.random.rand(8000, 100),
    "X_test": np.random.rand(2000, 100), 
    "y_train": np.random.randint(0, 2, 8000),
    "y_test": np.random.randint(0, 2, 2000)
}
cache.put(ml_data, 
          dataset="fraud_detection", 
          split="80_20", 
          preprocessing="standard_scaled")

# Retrieve arrays
training_data = cache.get(dataset="fraud_detection", split="80_20", preprocessing="standard_scaled")
print(training_data.keys())  # ['X_train', 'X_test', 'y_train', 'y_test']
```

### DataFrames & Series (Pandas/Polars)

```python
# Pandas DataFrames (automatically stored as Parquet)
import pandas as pd
df = pd.DataFrame({
    "user_id": range(1000),
    "purchase_amount": np.random.exponential(50, 1000),
    "category": np.random.choice(["A", "B", "C"], 1000)
})

cache.put(df, 
          source="transactions", 
          date_range="2024_q1", 
          preprocessing="outliers_removed")

# Pandas Series (automatically stored as Parquet with index preservation)
series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], 
                   index=['a', 'b', 'c', 'd', 'e'], 
                   name="measurements")

cache.put(series, 
          sensor="temperature", 
          location="lab_1", 
          date="2024_01_15")

# Polars DataFrames and Series (if available, with automatic fallback)
try:
    import polars as pl
    
    # Polars DataFrame
    df_polars = pl.DataFrame({
        "id": range(1000),
        "value": np.random.rand(1000)
    })
    cache.put(df_polars, source="sensor_data", date="2024_01_15")
    
    # Polars Series  
    series_polars = pl.Series("measurements", [10.1, 20.2, 30.3])
    cache.put(series_polars, sensor="pressure", location="lab_2")
    
except ImportError:
    # Library gracefully handles missing Polars
    pass

# Complex index types are fully supported (MultiIndex, etc.)
multi_index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], 
                                        names=['letter', 'number'])
complex_series = pd.Series([100, 200, 300], index=multi_index, name="values")
cache.put(complex_series, experiment="multi_index_test")
```

## Real-World Examples

### Machine Learning Pipeline

```python
from cacheness import cached, cacheness
import pandas as pd
import numpy as np

cache = cacheness()

# Cache expensive data preprocessing
@cached(ttl_hours=24, key_prefix="preprocessing")
def load_and_preprocess_data(data_source, preprocessing_params):
    """Load and preprocess data with caching."""
    # Expensive data loading and preprocessing
    raw_data = pd.read_csv(data_source)
    processed_data = apply_preprocessing(raw_data, preprocessing_params)
    return processed_data

# Cache model training
@cached(ttl_hours=168, key_prefix="model_training")  # 1 week TTL
def train_model(data, model_type, hyperparams):
    """Train model with automatic caching."""
    if model_type == "xgboost":
        model = train_xgboost(data, hyperparams)
    elif model_type == "lightgbm":
        model = train_lightgbm(data, hyperparams)
    return model

# Cache expensive feature engineering
@cached(ttl_hours=72, key_prefix="features")
def engineer_features(raw_data, feature_config):
    """Feature engineering with caching."""
    return create_features(raw_data, feature_config)

# Usage - automatic caching based on parameters
data = load_and_preprocess_data(
    data_source="./data/customers.csv",
    preprocessing_params={"normalize": True, "handle_outliers": "clip"}
)

features = engineer_features(
    raw_data=data,
    feature_config={"include_interactions": True, "polynomial_degree": 2}
)

model = train_model(
    data=features,
    model_type="xgboost", 
    hyperparams={"learning_rate": 0.01, "max_depth": 6}
)
```

### Experiment Results Tracking

```python
# Cache experiment results with rich metadata
def run_experiment(model_name, dataset, hyperparams, fold_id):
    results = train_and_evaluate(model_name, dataset, hyperparams, fold_id)
    
    # Store results with comprehensive keys
    cache.put(
        results,
        experiment_id="housing_prediction_study",
        model=model_name,
        dataset=dataset,
        fold=fold_id,
        hyperparams_hash=hash(str(sorted(hyperparams.items()))),
        timestamp="2024_01_15",
        description=f"{model_name} results on {dataset} fold {fold_id}"
    )
    
    return results

# Retrieve specific experiment results
results = cache.get(
    experiment_id="housing_prediction_study",
    model="random_forest",
    dataset="boston_housing", 
    fold=3,
    hyperparams_hash=hash(str(sorted({"n_estimators": 100, "max_depth": 10}.items()))),
    timestamp="2024_01_15"
)
```

## Cache Management & Monitoring

### Statistics and Performance

```python
# Get comprehensive cache statistics
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Backend: {stats['backend_type']}")

# List all cache entries with metadata
entries = cache.list_entries()
for entry in entries:
    print(f"{entry['cache_key']}: {entry['description']} ({entry['size_mb']:.2f}MB)")
    print(f"  Created: {entry['created_at']}")
    print(f"  Type: {entry['data_type']}")
    print(f"  Expired: {entry.get('expired', False)}")
```

### Cache Cleanup and Management

```python
# Manual cleanup of specific entries
cache.invalidate(model="old_model", version="1.0")  # Remove specific entry

# Clear all cache entries
cache.clear_all()

# Automatic cleanup happens on initialization if cleanup_on_init=True
# Size-based cleanup happens automatically when max_cache_size_mb is exceeded
```

### TTL (Time-To-Live) Management

```python
# Default TTL from config
cache.put(data, experiment="test")  # Uses default_ttl_hours

# Custom TTL for specific data
cache.put(data, experiment="temporary", ttl_hours=2)   # Expires in 2 hours
cache.put(data, experiment="permanent", ttl_hours=None) # Never expires

# Check if entry exists and is not expired
result = cache.get(experiment="test", ttl_hours=48)  # Custom TTL override
```

## Architecture & Features

### Smart Storage Format Selection

The library automatically selects optimal storage formats:

| Data Type | Storage Format | Compression | Benefits |
|-----------|---------------|-------------|----------|
| NumPy arrays | NPZ + Blosc2 | LZ4/ZSTD | 60-80% size reduction, 4x faster I/O |
| DataFrames & Series | Parquet | LZ4 | 40-60% size reduction, columnar efficiency |
| Python objects | Pickle + Blosc | LZ4 | 30-50% size reduction, universal compatibility |

### Performance Optimizations

The library includes several automatic performance optimizations:

**JSON Serialization with orjson**:
- **orjson** (Rust-based): 1.5-5x faster JSON serialization, 1.5-3x faster deserialization
- Automatic fallback to built-in `json` if orjson not available
- Used for cache key generation and metadata storage
- Native datetime handling and UTF-8 optimization

**SQLAlchemy Metadata Backend**:
- **Direct SQLAlchemy**: Lightweight, fast database operations without ORM overhead
- **Connection pooling**: Efficient database connection management
- **Optimized queries**: Minimized database round trips for metadata operations

### Metadata Backend Performance

| Backend | Operation | Small Cache (<1k entries) | Large Cache (10k+ entries) |
|---------|-----------|---------------------------|---------------------------|
| **SQLite** | list_entries() | 0.5ms | **2.3ms** |
| **SQLite** | get_stats() | 0.8ms | **4.1ms** | 
| **SQLite** | cleanup_expired() | 1.2ms | **12ms** |
| **JSON** | list_entries() | 0.3ms | 1.2s |
| **JSON** | get_stats() | 0.5ms | 850ms |
| **JSON** | cleanup_expired() | 0.8ms | 1.5s |

*SQLite provides 10-500x performance improvement for large caches*

### Fallback Mechanisms

- **JSON Serialization**: orjson → built-in json (graceful performance degradation)
- **DataFrame/Series Libraries**: Polars → Pandas → Pickle (graceful degradation)
- **Parquet Compatibility**: DataFrames/Series with complex objects → Pickle fallback
- **Array Compression**: Blosc2 → NPZ → Standard Pickle
- **Metadata Backend**: SQLite → JSON (automatic selection based on availability)
- **Thread Safety**: Built-in locking for concurrent access

### Directory Structure

```
cache_dir/
├── cache_metadata.db          # SQLite metadata (if available)  
├── cache_metadata.json        # JSON metadata (fallback)
├── {hash}_object.pkl.lz4      # Compressed Python objects
├── {hash}_array.npz           # Compressed NumPy arrays
└── {hash}_dataframe.parquet   # DataFrame storage
```

## Configuration Reference

### CacheConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | str | `"./cache"` | Cache directory path |
| `default_ttl_hours` | int | `24` | Default time-to-live in hours |
| `max_cache_size_mb` | int | `2000` | Maximum cache size in MB |
| `metadata_backend` | str | `"auto"` | Backend: "auto", "sqlite", "json" |
| `hash_path_content` | bool | `True` | Hash Path objects by content vs filename |
| `cleanup_on_init` | bool | `True` | Clean expired entries on initialization |
| `use_blosc2_arrays` | bool | `True` | Use Blosc2 for array compression |
| `npz_compression` | bool | `True` | Enable NPZ compression |
| `parquet_compression` | str | `"lz4"` | DataFrame/Series compression algorithm |
| `pickle_compression_codec` | str | `"zstd"` | Pickle compression codec |
| `pickle_compression_level` | int | `5` | Pickle compression level (1-9) |
| `blosc2_array_codec` | str | `"lz4"` | Blosc2 compression codec |
| `enable_multithreading` | bool | `True` | Use multi-threading for compression |
| `auto_optimize_threads` | bool | `True` | Auto-optimize thread count based on data size |

### Compression Options

**Python Object Compression (Pickle)**:
- `"zstd"` (default): Excellent general-purpose compression with multi-threading support
- `"lz4"`: Fastest compression/decompression, best for numerical data
- `"lz4hc"`: Better compression than lz4 with similar speed
- `"zlib"`: Standard compression, good compatibility

**Array Compression**:
- `"lz4"` (default): Fastest decompression
- `"lz4hc"`: Better compression than lz4  
- `"zstd"`: Good balance of speed and compression
- `"zlib"`: Standard compression

**DataFrame & Series Compression**:
- `"lz4"` (default): Fast compression/decompression
- `"snappy"`: Best for compatibility, balance of speed and compression

### Advanced Compression Features

The library automatically optimizes compression parameters based on data characteristics:

- **Intelligent Threading**: Automatically adjusts thread count based on data size

## Configurable Serialization and Handler Priority

**NEW in v0.2.1**: Full control over cache key serialization methods and handler selection order.

### Serialization Configuration

Control how objects are serialized for cache key generation with fine-grained options:

```python
from cacheness import CacheConfig, UnifiedCache

# Performance-optimized config (speed over precision)
performance_config = CacheConfig(
    # Disable expensive introspection methods
    enable_collections=False,           # Skip recursive collection analysis
    enable_object_introspection=False,  # Skip __dict__ inspection
    max_tuple_recursive_length=2,       # Limit tuple recursion
    max_collection_depth=3              # Limit nesting depth
)

# Precision-optimized config (accuracy over speed)
precision_config = CacheConfig(
    enable_collections=True,            # Full collection introspection
    enable_object_introspection=True,   # Deep object analysis
    max_tuple_recursive_length=50,      # Allow deep tuple recursion
    max_collection_depth=20             # Allow deep nesting
)

# Custom serialization methods
custom_config = CacheConfig(
    enable_basic_types=True,            # str, int, float, bool, bytes
    enable_special_cases=True,          # NumPy arrays, custom handlers
    enable_collections=True,            # lists, dicts, sets with recursion
    enable_object_introspection=True,   # Objects with __dict__
    enable_hashable_fallback=True,      # Hashable objects (fast)
    enable_string_fallback=True         # String representation (last resort)
)
```

### Handler Priority Configuration

Control the order and availability of data format handlers:

```python
# Custom handler priority (DataFrames processed first)
config = CacheConfig(
    handler_priority=[
        "pandas_dataframes",    # Process pandas DataFrames first
        "polars_dataframes",    # Then polars DataFrames  
        "pandas_series",        # Then pandas Series
        "polars_series",        # Then polars Series
        "numpy_arrays",         # Then NumPy arrays
        "object_pickle"         # Finally, pickle everything else
    ]
)

# Disable specific handlers
minimal_config = CacheConfig(
    enable_pandas_dataframes=False,     # Disable pandas DataFrame caching
    enable_polars_series=False,         # Disable polars Series caching
    enable_numpy_arrays=True,           # Keep NumPy array support
    enable_object_pickle=True           # Keep general object support
)
```

### Serialization Method Priority

The default serialization follows a **Quality-First Approach**:

1. **Basic immutable types** → Direct representation (`str:hello`, `int:42`)
2. **Special cases** → Custom handling (NumPy arrays, Path objects)  
3. **Collections** → Recursive introspection (`list:[int:1,int:2,int:3]`)
4. **Objects with __dict__** → Full introspection
5. **Hashable objects** → Performance fallback (`hashed:tuple:12345`)
6. **String representation** → Last resort (`MyClass:custom_str_repr`)

### Real-World Use Cases

**ML Pipeline Optimization**:
```python
# Fast caching for rapid iteration
ml_config = CacheConfig(
    cache_dir="./ml_cache",
    enable_collections=False,           # Skip expensive list/dict analysis
    max_tuple_recursive_length=3,       # Limit parameter tuple analysis
    handler_priority=["numpy_arrays", "object_pickle"]  # Prioritize arrays
)

@cached(cache_instance=UnifiedCache(ml_config))
def train_model(X, y, params):
    # Model training logic
    return trained_model, metrics, feature_importance
```

**Data Processing Precision**:
```python
# Detailed caching for complex workflows
data_config = CacheConfig(
    cache_dir="./data_cache", 
    enable_collections=True,            # Full parameter introspection
    enable_object_introspection=True,   # Deep object analysis
    max_collection_depth=15,            # Allow complex nested structures
    handler_priority=["pandas_dataframes", "polars_dataframes", "numpy_arrays"]
)

@cached(cache_instance=UnifiedCache(data_config))
def process_datasets(raw_data, transformations, config_dict):
    # Complex data processing with nested configurations
    return processed_data, metadata, validation_results
```

**Configuration Benefits**:
- **Performance Tuning**: Optimize for your specific data patterns
- **Memory Efficiency**: Disable unnecessary introspection methods
- **Cache Key Precision**: Control how function parameters are analyzed
- **Handler Optimization**: Prioritize the data formats you use most
- **Compatibility Control**: Enable/disable specific libraries (pandas/polars)
- **Dynamic Compression Levels**: Optimizes compression level for data size and type
- **ZSTD Optimizations**: Leverages ZSTD's advanced features for general-purpose data

```python
# The library automatically optimizes these settings:
# - Small data (<1MB): Fewer threads, lower compression
# - Medium data (1-10MB): Moderate threading and compression  
# - Large data (>10MB): More threads, higher compression
```

## Best Practices

### 1. Use Descriptive Key Parameters

```python
# Good: Hierarchical and descriptive parameters
cache.put(model, 
          project="customer_segmentation",
          model_type="kmeans", 
          features="demographic_behavioral",
          k_clusters=5,
          version="v2.1")

# Avoid: Generic or ambiguous keys  
cache.put(model, id="model123", data="stuff")
```

### 2. Leverage Decorators for Functions

```python
# Cache expensive computations automatically
@cached(ttl_hours=24, key_prefix="data_processing")
def process_large_dataset(file_path, parameters):
    return expensive_processing(file_path, parameters)

# Cache with appropriate TTL for different use cases
@cached(ttl_hours=1)     # Short-lived: real-time data
@cached(ttl_hours=168)   # Long-lived: static features  
@cached(ttl_hours=None)  # Permanent: reference data
```

### 3. Monitor Cache Performance

```python
# Regular monitoring
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Total size: {stats['total_size_mb']:.1f}MB / {cache.config.max_cache_size_mb}MB")

# Log cache operations for debugging
import logging
logging.basicConfig(level=logging.INFO)
# Cache operations will now be logged
```

### 4. Handle Missing Dependencies Gracefully

```python
# The library automatically handles missing optional dependencies
try:
    import polars as pl
    # Use Polars if available
except ImportError:
    # Falls back to Pandas, then to Pickle
    pass

# Check backend selection
cache = cacheness()
print(f"Using {cache.actual_backend} metadata backend")
```

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test categories  
python -m pytest tests/test_core.py -v       # Core functionality
python -m pytest tests/test_handlers.py -v   # Data handlers
python -m pytest tests/test_metadata.py -v   # Metadata backends

# Run with coverage
python -m pytest --cov=cacheness --cov-report=html

# Test with minimal dependencies (JSON backend only)
python -m pytest -k "not sqlite"
```

## Error Handling & Troubleshooting

### Robust Error Handling

```python
try:
    cache.put(data, model="test", version="1.0")
    result = cache.get(model="test", version="1.0")
except ValueError as e:
    print(f"Invalid data or parameters: {e}")
except IOError as e:
    print(f"Storage error: {e}")
except ImportError as e:
    # Optional dependencies missing - fallbacks activated
    print(f"Fallback activated: {e}")
```

### Common Issues & Solutions

**Cache Miss for Expected Data**:
- Verify parameter names and values match exactly
- Check if TTL has expired: `cache.get(model="test", ttl_hours=48)`
- Review cache entries: `cache.list_entries()`

**Performance Issues**:
- Use SQLite backend for large caches: `metadata_backend="sqlite"`
- Enable compression: `use_blosc2_arrays=True`
- Monitor cache size: `cache.get_stats()`

**Missing Dependencies**:
```python
# Check what backend is actually being used
cache = cacheness()
print(f"Backend: {cache.actual_backend}")  # "sqlite" or "json"

# Optional dependencies are handled gracefully
# Library will use fallbacks automatically
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Cache operations will now log detailed information
cache = cacheness()
cache.put(data, debug="test")  # Will log storage details
result = cache.get(debug="test")  # Will log retrieval details

## Dependencies

### Required
- **xxhash**: XXH3_64 hashing for cache keys
- **numpy**: Array handling and storage
- **blosc2**: High-performance array compression (included in core)

### Optional (Graceful Fallbacks)
- **sqlalchemy**: SQLite metadata backend (10-500x faster than JSON)
- **orjson**: High-performance JSON serialization (1.5-5x faster than built-in json)
- **blosc2**: High-performance array compression  
- **pandas**: DataFrame and Series support
- **polars**: High-performance DataFrame and Series operations
- **pyarrow**: Parquet file support

## License

GPLv3 License - see LICENSE file for details.
