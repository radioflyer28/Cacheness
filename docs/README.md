# Cacheness Documentation

Welcome to the Cacheness documentation! This high-performance caching library provides intelligent data type handling, compression, and security features for Python applications.

## Getting Started

- **[README](../README.md)** - Quick start guide and basic usage
- **[Configuration Guide](CONFIGURATION.md)** - Comprehensive configuration options
- **[Backend Selection Guide](BACKEND_SELECTION.md)** - Choose JSON vs SQLite backend
- **[Security Guide](SECURITY.md)** - Cache entry signing and security features
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

## Platform Compatibility

- **[Cross-Platform Guide](CROSS_PLATFORM_GUIDE.md)** - Development guide for all platforms (Windows, Linux, macOS)
- **[Windows Compatibility](WINDOWS_COMPATIBILITY.md)** - Windows-specific considerations and troubleshooting
- **[Pandas Compatibility](PANDAS_COMPATIBILITY.md)** - pandas 2.0-3.x version compatibility guide

## Advanced Features

- **[Performance Guide](PERFORMANCE.md)** - Optimization strategies and benchmarks
- **[Custom Metadata](CUSTOM_METADATA.md)** - Working with SQLite backend and custom fields

## Specialized Handlers

- **[Dill Integration](DILL_INTEGRATION.md)** - Advanced object serialization (functions, lambdas)
- **[TensorFlow Tensor Guide](TENSORFLOW_TENSOR_GUIDE.md)** - Native TensorFlow tensor caching
- **[TensorFlow Handler Status](TENSORFLOW_HANDLER_STATUS.md)** - Handler implementation details

## Storage Layer

- **[BlobStore Guide](BLOB_STORE.md)** - Low-level storage API for non-caching use cases

`BlobStore` provides direct key-value storage without caching semantics (no TTL, no eviction). Useful for ML model versioning, artifact storage, and data pipeline checkpoints.

```python
from cacheness.storage import BlobStore

store = BlobStore(cache_dir="./models", backend="sqlite")
store.put(model, key="fraud_detector_v1", metadata={"accuracy": 0.95})
model = store.get("fraud_detector_v1")
```

## API Reference

- **[API Reference](API_REFERENCE.md)** - Complete API documentation

## Security

Cacheness provides cryptographic signing for cache integrity:

- **HMAC-SHA256 signatures** protect against tampering
- **Configurable security levels** (minimal, enhanced, paranoid)
- **In-memory keys** for enhanced security (no disk persistence)
- **Automatic cleanup** of entries with invalid signatures

See the [Security Guide](SECURITY.md) for detailed configuration and best practices.

## Key Features

### Intelligent Data Type Handling
- **DataFrames**: pandas, polars with optimized parquet storage
- **Arrays**: NumPy with blosc2 compression
- **Objects**: Advanced serialization with dill fallback
- **Tensors**: Native TensorFlow tensor support

### Core Caching Backends

### Core Caching Backends
- **SQLite**: Production-ready with full concurrency support (recommended for 200+ entries)
- **JSON**: Fast for small caches and development (< 200 entries, single-process only)
- **In-Memory SQLite**: Maximum performance for temporary data

**Important**: JSON backend is **not safe** for multiple processes. Use SQLite backend for production deployments or when multiple processes access the same cache.

See the [Backend Selection Guide](BACKEND_SELECTION.md) for detailed comparison and recommendations.

### Advanced Compression
- **Multi-codec support**: LZ4, ZSTD, LZ4HC for different performance profiles
- **Array-specific compression**: Blosc2 optimization for numerical data
- **Configurable thresholds**: Compress only when beneficial

### Robust Caching Features
- **TTL management**: Automatic expiration with configurable defaults
- **Cache size limits**: Automatic cleanup when limits exceeded
- **Integrity verification**: Optional file hash checking
- **Flexible key generation**: Deep parameter introspection

## Use Cases

- **Machine Learning**: Cache datasets, models, and experiment results
- **Data Processing**: Store intermediate pipeline results
- **API Responses**: Fast response caching with automatic expiration
- **Scientific Computing**: Preserve expensive computation results
- **Development**: Speed up development with intelligent caching

## Installation

```bash
pip install cacheness
```

## Quick Example

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Basic usage
cache = cacheness()
result = cache.get(experiment="ml_training", dataset="cifar10")
if result is None:
    result = expensive_training_function()
    cache.put(result, experiment="ml_training", dataset="cifar10")

# Secure configuration
secure_cache = cacheness(CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        use_in_memory_key=True,
        security_level="enhanced"
    )
))
```

## Contributing

For development setup and contribution guidelines, see the project repository.

---

**Next Steps:**
- Start with the [Configuration Guide](CONFIGURATION.md) for detailed setup
- Review [Security Guide](SECURITY.md) for production deployments  
- Check [Performance Guide](PERFORMANCE.md) for optimization tips
