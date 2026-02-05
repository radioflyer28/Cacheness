# Custom Metadata Guide

Enhance your caching with rich metadata for better organization, debugging, and advanced use cases.

## Overview

Cacheness supports custom metadata alongside the built-in metadata (creation time, size, etc.). This enables powerful filtering, searching, and management capabilities for your cache.

## Basic Custom Metadata

### Setting Metadata

```python
from cacheness import cacheness

cache = cacheness()

# Add metadata when storing
cache.put(
    data=trained_model,
    
    # Standard cache key parameters
    project="fraud_detection",
    model_type="xgboost",
    version="v2.1",
    
    # Custom metadata (preserved as-is)
    metadata={
        "accuracy": 0.94,
        "training_time_minutes": 45,
        "dataset_size": 100000,
        "features": ["amount", "merchant", "location", "time"],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        "author": "data_team",
        "environment": "production",
        "notes": "Best performing model for Q4 data"
    }
)
```

### Retrieving with Metadata

```python
# Get data with metadata
result = cache.get_with_metadata(
    project="fraud_detection",
    model_type="xgboost",
    version="v2.1"
)

if result:
    data, metadata = result
    print(f"Model accuracy: {metadata['accuracy']}")
    print(f"Training time: {metadata['training_time_minutes']} minutes")
    print(f"Author: {metadata['author']}")
```

## Advanced Metadata Patterns

### Experiment Tracking

```python
def train_and_cache_model(experiment_config):
    """Cache ML experiments with detailed metadata for tracking."""
    
    # Train model
    model = train_model(experiment_config)
    
    # Evaluate model
    metrics = evaluate_model(model, test_data)
    
    # Store with comprehensive metadata
    cache.put(
        model,
        experiment_id=experiment_config["experiment_id"],
        model_type=experiment_config["model_type"],
        
        metadata={
            # Experiment configuration
            "config": experiment_config,
            
            # Performance metrics
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics["auc_roc"],
            
            # Training metadata
            "training_duration": metrics["training_time"],
            "dataset_info": {
                "train_size": len(train_data),
                "test_size": len(test_data),
                "features": list(train_data.columns),
                "target_distribution": get_target_distribution(train_data)
            },
            
            # Infrastructure metadata
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "hardware": {
                "cpu_count": os.cpu_count(),
                "memory_gb": psutil.virtual_memory().total // (1024**3)
            },
            
            # Git metadata (if available)
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            
            # Timestamps
            "created_at": datetime.now().isoformat(),
            "created_by": os.getenv("USER", "unknown")
        }
    )

# Usage
experiment_config = {
    "experiment_id": "exp_2024_01_15_001",
    "model_type": "random_forest",
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

train_and_cache_model(experiment_config)
```

### Data Pipeline Metadata

```python
def cache_processed_data(raw_data_path, processing_version):
    """Cache processed data with lineage and quality metadata."""
    
    # Process data
    processed_data = process_data(raw_data_path)
    
    # Calculate data quality metrics
    quality_metrics = calculate_quality_metrics(processed_data)
    
    cache.put(
        processed_data,
        data_path=raw_data_path,
        processing_version=processing_version,
        
        metadata={
            # Data lineage
            "source_file": raw_data_path,
            "source_size_mb": os.path.getsize(raw_data_path) / (1024**2),
            "source_modified": datetime.fromtimestamp(
                os.path.getmtime(raw_data_path)
            ).isoformat(),
            
            # Processing metadata
            "processing_version": processing_version,
            "processing_steps": [
                "remove_duplicates",
                "handle_missing_values",
                "normalize_categories",
                "feature_engineering"
            ],
            "processing_time": quality_metrics["processing_time"],
            
            # Data quality metrics
            "row_count": len(processed_data),
            "column_count": len(processed_data.columns),
            "missing_value_percentage": quality_metrics["missing_pct"],
            "duplicate_row_count": quality_metrics["duplicates"],
            "data_types": processed_data.dtypes.to_dict(),
            
            # Column statistics
            "column_stats": {
                col: {
                    "unique_values": processed_data[col].nunique(),
                    "null_count": processed_data[col].isnull().sum(),
                    "mean": processed_data[col].mean() if processed_data[col].dtype in ['int64', 'float64'] else None
                }
                for col in processed_data.columns
            },
            
            # Validation results
            "validation_passed": quality_metrics["validation_passed"],
            "validation_errors": quality_metrics["validation_errors"]
        }
    )
```

### API Response Metadata

```python
def cache_api_response(url, params):
    """Cache API responses with detailed request/response metadata."""
    
    start_time = time.time()
    response = requests.get(url, params=params)
    response_time = time.time() - start_time
    
    cache.put(
        response.json(),
        api_endpoint=url,
        params_hash=hash_params(params),
        
        metadata={
            # Request metadata
            "url": url,
            "method": "GET",
            "params": params,
            "request_headers": dict(response.request.headers),
            "request_time": datetime.now().isoformat(),
            "response_time_seconds": response_time,
            
            # Response metadata
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "content_length": len(response.content),
            "content_type": response.headers.get("content-type"),
            
            # Rate limiting info (if available)
            "rate_limit_remaining": response.headers.get("x-ratelimit-remaining"),
            "rate_limit_reset": response.headers.get("x-ratelimit-reset"),
            
            # Cache metadata
            "cache_source": "api_request",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    )
```

## Metadata Querying and Filtering

### Finding Entries by Metadata

```python
# Get all entries with metadata
all_entries = cache.list_entries(include_metadata=True)

# Filter by custom metadata criteria
def find_high_accuracy_models(entries, min_accuracy=0.9):
    """Find models with accuracy above threshold."""
    high_accuracy_models = []
    
    for entry in entries:
        metadata = entry.get('metadata', {})
        accuracy = metadata.get('accuracy')
        
        if accuracy and accuracy >= min_accuracy:
            high_accuracy_models.append({
                'cache_key': entry['cache_key'],
                'accuracy': accuracy,
                'model_type': metadata.get('config', {}).get('model_type'),
                'created_at': entry['created_at']
            })
    
    return sorted(high_accuracy_models, key=lambda x: x['accuracy'], reverse=True)

# Find best models
best_models = find_high_accuracy_models(all_entries, min_accuracy=0.92)
for model in best_models:
    print(f"Model: {model['model_type']}, Accuracy: {model['accuracy']:.3f}")
```

### Metadata-Based Cleanup

```python
def cleanup_by_metadata(cache, criteria):
    """Clean cache entries based on metadata criteria."""
    
    entries = cache.list_entries(include_metadata=True)
    to_remove = []
    
    for entry in entries:
        metadata = entry.get('metadata', {})
        
        # Remove low-performing models
        if metadata.get('accuracy', 1.0) < criteria.get('min_accuracy', 0.0):
            to_remove.append(entry['cache_key'])
            
        # Remove old experiments from specific user
        if (metadata.get('created_by') == criteria.get('author') and 
            entry['created_at'] < criteria.get('before_date')):
            to_remove.append(entry['cache_key'])
            
        # Remove large failed validations
        if (not metadata.get('validation_passed', True) and 
            entry['size_mb'] > criteria.get('max_size_mb', float('inf'))):
            to_remove.append(entry['cache_key'])
    
    # Remove identified entries
    for cache_key in to_remove:
        cache.invalidate_by_cache_key(cache_key)
        print(f"Removed cache entry: {cache_key}")
    
    return len(to_remove)

# Example cleanup
cleanup_criteria = {
    'min_accuracy': 0.85,
    'author': 'test_user',
    'before_date': '2024-01-01',
    'max_size_mb': 100
}

removed_count = cleanup_by_metadata(cache, cleanup_criteria)
print(f"Removed {removed_count} entries based on metadata criteria")
```

## Metadata Best Practices

### 1. Consistent Schema

```python
# Define metadata schemas for different use cases
ML_EXPERIMENT_SCHEMA = {
    "accuracy": float,
    "training_time_minutes": float,
    "dataset_size": int,
    "features": list,
    "hyperparameters": dict,
    "author": str,
    "environment": str
}

DATA_PROCESSING_SCHEMA = {
    "source_file": str,
    "processing_version": str,
    "row_count": int,
    "column_count": int,
    "processing_time": float,
    "validation_passed": bool
}

def validate_metadata(metadata, schema):
    """Validate metadata against schema."""
    for key, expected_type in schema.items():
        if key in metadata:
            if not isinstance(metadata[key], expected_type):
                raise ValueError(f"Metadata '{key}' should be {expected_type.__name__}")
```

### 2. Metadata Versioning

```python
def add_versioned_metadata(data, base_metadata):
    """Add versioned metadata with schema version."""
    
    versioned_metadata = {
        "schema_version": "1.2",
        "metadata_created_at": datetime.now().isoformat(),
        **base_metadata
    }
    
    # Migration logic for different schema versions
    if "schema_version" not in base_metadata:
        # Migrate from unversioned schema
        versioned_metadata = migrate_v0_to_v1(versioned_metadata)
    
    return versioned_metadata
```

### 3. Searchable Metadata

```python
def create_searchable_metadata(base_metadata):
    """Create metadata optimized for searching."""
    
    # Add searchable tags
    tags = []
    
    # Auto-generate tags from metadata
    if base_metadata.get('model_type'):
        tags.append(f"model:{base_metadata['model_type']}")
    
    if base_metadata.get('accuracy', 0) > 0.9:
        tags.append("high_accuracy")
    
    if base_metadata.get('environment'):
        tags.append(f"env:{base_metadata['environment']}")
    
    # Add searchable text
    searchable_text = " ".join([
        base_metadata.get('notes', ''),
        base_metadata.get('author', ''),
        " ".join(tags)
    ])
    
    return {
        **base_metadata,
        "tags": tags,
        "searchable_text": searchable_text.lower()
    }

# Usage
metadata = create_searchable_metadata({
    "model_type": "xgboost",
    "accuracy": 0.94,
    "author": "data_team",
    "environment": "production",
    "notes": "Best model for fraud detection"
})

# Later search by tags
def search_by_tag(entries, tag):
    return [e for e in entries 
            if tag in e.get('metadata', {}).get('tags', [])]

production_models = search_by_tag(cache.list_entries(include_metadata=True), 
                                 "env:production")
```

## Integration with Monitoring

### Metadata-Based Alerting

```python
import logging

def check_cache_health(cache):
    """Monitor cache health using metadata."""
    
    entries = cache.list_entries(include_metadata=True)
    issues = []
    
    # Check for failed validations
    failed_validations = [
        e for e in entries 
        if not e.get('metadata', {}).get('validation_passed', True)
    ]
    
    if failed_validations:
        issues.append(f"Found {len(failed_validations)} failed validations")
    
    # Check for low-quality data
    low_quality = [
        e for e in entries
        if e.get('metadata', {}).get('missing_value_percentage', 0) > 50
    ]
    
    if low_quality:
        issues.append(f"Found {len(low_quality)} high missing value datasets")
    
    # Check for old experiments
    week_ago = datetime.now() - timedelta(days=7)
    old_experiments = [
        e for e in entries
        if (datetime.fromisoformat(e['created_at']) < week_ago and
            e.get('metadata', {}).get('environment') == 'development')
    ]
    
    if old_experiments:
        issues.append(f"Found {len(old_experiments)} old development experiments")
    
    # Log issues
    if issues:
        logging.warning(f"Cache health issues: {'; '.join(issues)}")
    else:
        logging.info("Cache health check passed")
    
    return issues

# Run health check
health_issues = check_cache_health(cache)
```

### Metadata Reporting

```python
def generate_cache_report(cache):
    """Generate comprehensive cache report using metadata."""
    
    entries = cache.list_entries(include_metadata=True)
    
    report = {
        "summary": {
            "total_entries": len(entries),
            "total_size_mb": sum(e['size_mb'] for e in entries),
            "date_range": {
                "oldest": min(e['created_at'] for e in entries),
                "newest": max(e['created_at'] for e in entries)
            }
        },
        "by_type": {},
        "by_author": {},
        "quality_metrics": {},
        "performance_metrics": {}
    }
    
    # Analyze by type
    for entry in entries:
        metadata = entry.get('metadata', {})
        
        # Group by model type
        model_type = metadata.get('model_type', 'unknown')
        if model_type not in report["by_type"]:
            report["by_type"][model_type] = {"count": 0, "total_size_mb": 0}
        
        report["by_type"][model_type]["count"] += 1
        report["by_type"][model_type]["total_size_mb"] += entry['size_mb']
        
        # Group by author
        author = metadata.get('author', 'unknown')
        if author not in report["by_author"]:
            report["by_author"][author] = {"count": 0, "avg_accuracy": []}
        
        report["by_author"][author]["count"] += 1
        if 'accuracy' in metadata:
            report["by_author"][author]["avg_accuracy"].append(metadata['accuracy'])
    
    # Calculate averages
    for author_data in report["by_author"].values():
        if author_data["avg_accuracy"]:
            author_data["avg_accuracy"] = sum(author_data["avg_accuracy"]) / len(author_data["avg_accuracy"])
        else:
            author_data["avg_accuracy"] = None
    
    return report

# Generate and display report
report = generate_cache_report(cache)
print(f"Total entries: {report['summary']['total_entries']}")
print(f"Total size: {report['summary']['total_size_mb']:.2f} MB")

for model_type, stats in report['by_type'].items():
    print(f"{model_type}: {stats['count']} entries, {stats['total_size_mb']:.2f} MB")
```

## Metadata Storage and Performance

### Efficient Metadata Storage

```python
# Configure metadata storage
from cacheness import CacheConfig
from cacheness.config import CacheMetadataConfig

config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",                    # Better for complex queries
        store_cache_key_params=True,        # Enable parameter storage
        verify_cache_integrity=True         # Enable integrity checks
    )
)
```

### Metadata Size Considerations

```python
def optimize_metadata_size(metadata):
    """Optimize metadata for storage efficiency."""
    
    optimized = {}
    
    for key, value in metadata.items():
        # Truncate long strings
        if isinstance(value, str) and len(value) > 1000:
            optimized[key] = value[:1000] + "... (truncated)"
        
        # Round floating point numbers
        elif isinstance(value, float):
            optimized[key] = round(value, 6)
        
        # Limit list/array sizes
        elif isinstance(value, list) and len(value) > 100:
            optimized[key] = value[:100] + ["... (truncated)"]
        
        # Simplify large dictionaries
        elif isinstance(value, dict) and len(value) > 50:
            # Keep only important keys
            important_keys = ['accuracy', 'model_type', 'version', 'environment']
            optimized[key] = {k: v for k, v in value.items() if k in important_keys}
            optimized[key]['_truncated'] = True
        
        else:
            optimized[key] = value
    
    return optimized

# Use optimized metadata
optimized_metadata = optimize_metadata_size(large_metadata)
cache.put(data, key="example", metadata=optimized_metadata)
```

## PostgreSQL Backend Compatibility

Custom metadata works seamlessly with both SQLite and PostgreSQL metadata backends. This enables you to use structured custom metadata with production-grade PostgreSQL deployments.

### Using Custom Metadata with PostgreSQL

```python
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig

# Configure cache with PostgreSQL metadata backend
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",
        connection_url="postgresql://user:password@localhost:5432/cacheness_db",
        pool_size=10,
        max_overflow=20
    )
)

cache = cacheness(config=config)

# Use custom metadata exactly the same as with SQLite
cache.put(
    data=model,
    model_name="fraud_detector",
    version="v1.0",
    metadata={
        "accuracy": 0.94,
        "training_time_minutes": 45,
        "author": "data_team"
    }
)

# Retrieve with metadata
result = cache.get_with_metadata(model_name="fraud_detector", version="v1.0")
if result:
    data, metadata = result
    print(f"Accuracy: {metadata['accuracy']}")
```

### Custom SQLAlchemy Models with PostgreSQL

For advanced use cases, you can define custom SQLAlchemy ORM models that work with both SQLite and PostgreSQL:

```python
from sqlalchemy import Column, Integer, String, Float, DateTime
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base

@custom_metadata_model("experiments")
class MLExperiment(Base, CustomMetadataBase):
    """Custom metadata table for ML experiments."""
    __tablename__ = "custom_ml_experiments"
    
    # cache_key FK is inherited from CustomMetadataBase
    experiment_name = Column(String, nullable=False, index=True)
    accuracy = Column(Float, index=True)
    f1_score = Column(Float, index=True)
    model_type = Column(String, index=True)
    created_at = Column(DateTime)

# The custom model works with both SQLite and PostgreSQL backends
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig

# With PostgreSQL
pg_config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",
        connection_url="postgresql://user:password@localhost:5432/cacheness_db"
    )
)
pg_cache = cacheness(config=pg_config)

# Or with SQLite (default)
sqlite_cache = cacheness()

# Both work identically with custom metadata models
```

### Migrating Custom Metadata Tables

When using custom SQLAlchemy models, you can migrate the tables to your database:

```python
from cacheness.custom_metadata import migrate_custom_metadata_tables
from sqlalchemy import create_engine

# Create engine for your database
engine = create_engine("postgresql://user:password@localhost:5432/cacheness_db")

# Migrate custom metadata tables
migrate_custom_metadata_tables(engine)
```

> **Note:** The migration function only creates custom metadata tables. Infrastructure tables (`cache_entries`, `cache_stats`) are managed separately by each backend.

### Querying Custom Metadata with PostgreSQL

Use `cache.query_custom_session()` to perform advanced queries on your custom metadata:

```python
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig

config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",
        connection_url="postgresql://user:password@localhost:5432/cacheness_db"
    )
)
cache = cacheness(config=config)

with cache.query_custom_session("experiments") as query:
    # Query high-accuracy experiments
    high_accuracy = query.filter(
        MLExperiment.accuracy > 0.9
    ).all()
    
    for exp in high_accuracy:
        print(f"Experiment: {exp.experiment_name}, Accuracy: {exp.accuracy}")
```
```

### Backend Architecture Notes

When using custom metadata with different backends:

| Backend | Infrastructure Tables | Custom Metadata Tables |
|---------|----------------------|------------------------|
| SQLite | `cache_entries`, `cache_stats` | Custom models with direct FK to `cache_entries.cache_key` |
| PostgreSQL | `cache_entries`, `cache_stats` | Custom models with direct FK to `cache_entries.cache_key` |

**Architecture:**
- Custom metadata models inherit from `CustomMetadataBase` which provides the `cache_key` foreign key
- Each custom metadata record belongs to **exactly one** cache entry (one-to-many)
- Direct foreign key provides clear ownership and automatic cascade deletion
- Both SQLite and PostgreSQL backends work identically

### Automatic Cascade Deletion

**Important Behavior:** When cache entries are deleted (via `invalidate()` or `clear_all()`), **custom metadata records are automatically cascade-deleted** via the foreign key constraint. This ensures clean cache isolation:

- Each cache entry "owns" its custom metadata records
- Deleting a cache entry automatically removes all associated custom metadata
- No orphaned metadata records to clean up
- Better cache isolation - metadata cannot be shared across cache entries

**What happens when cache entry is deleted:**
```
Before:                          After cache.invalidate():
┌─────────────────────┐          ┌─────────────────────┐
│ cache_entries       │          │ cache_entries       │
│  cache_key=abc123   │ ╌╌╌╌╌>   │  (entry deleted)    │
└─────────────────────┘          └─────────────────────┘
         ↑                                ↑
         │ FK (CASCADE)                   │ (cascade triggered)
         │                                ↓
┌─────────────────────┐          ┌─────────────────────┐
│ custom_experiments  │          │ custom_experiments  │
│  cache_key=abc123   │ ╌╌╌╌╌>   │  (auto-deleted)     │
│  accuracy=0.95      │          │                     │
└─────────────────────┘          └─────────────────────┘
```

**Querying after deletion:**
```python
# Store with custom metadata
experiment = MLExperiment(
    experiment_name="exp_001",
    accuracy=0.95
)
cache.put(model, experiment="exp_001", custom_metadata=experiment)

# Verify metadata exists
meta = cache.get_custom_metadata_for_entry(experiment="exp_001")
print(meta)  # {'experiments': <MLExperiment object>}

# Delete cache entry
cache.invalidate(experiment="exp_001")

# Custom metadata automatically deleted
meta = cache.get_custom_metadata_for_entry(experiment="exp_001")
print(meta)  # {} (empty - no entry, no metadata)
```

**Benefits of automatic cascade:**
- Simpler code - no manual cleanup needed
- Clearer semantics - metadata lifecycle tied to cache entry
- Better isolation - each cache entry has independent metadata
- No orphaned records - database stays clean automatically

## Best Practices for Custom Metadata Tables

### Table Naming Convention

Always prefix custom metadata tables with `custom_` to distinguish them from infrastructure tables:

```python
@custom_metadata_model("experiments")
class MLExperiment(Base, CustomMetadataBase):
    __tablename__ = "custom_ml_experiments"  # ✅ Good
    # __tablename__ = "ml_experiments"       # ⚠️  Warning from validator
```

### Add Indexes to Frequently Queried Columns

Improve query performance by indexing columns you'll filter or sort by:

```python
@custom_metadata_model("experiments")
class MLExperiment(Base, CustomMetadataBase):
    __tablename__ = "custom_ml_experiments"
    
    experiment_name = Column(String, nullable=False, unique=True, index=True)  # ✅
    accuracy = Column(Float, index=True)        # ✅ Frequently filtered
    model_type = Column(String, index=True)     # ✅ Frequently grouped
    notes = Column(String)                      # No index - rarely queried
```

### Use Appropriate Data Types

Choose the right SQLAlchemy column types for your data:

```python
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text

@custom_metadata_model("experiments")
class MLExperiment(Base, CustomMetadataBase):
    __tablename__ = "custom_ml_experiments"
    
    # Use specific types for better query performance
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    accuracy = Column(Float, index=True)           # Not String!
    epochs = Column(Integer, index=True)           # Not String!
    is_production = Column(Boolean, default=False) # Not String!
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Use Text for long descriptions (no length limit)
    description = Column(Text, nullable=True)
```

### Register Model Name for Easy Querying

The decorator's model name parameter makes querying easier:

```python
# Register with descriptive name
@custom_metadata_model("ml_experiments")  # ✅ Easy to remember
class MLExperiment(Base, CustomMetadataBase):
    ...

# Query using the registered name
with cache.query_custom_session("ml_experiments") as query:
    results = query.filter(MLExperiment.accuracy > 0.9).all()
```

### Store Multiple Models for Different Concerns

Separate different types of metadata into different models:

```python
@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    """ML experiment configuration and results."""
    __tablename__ = "custom_experiments"
    experiment_id = Column(String(100), index=True)
    accuracy = Column(Float, index=True)
    model_type = Column(String(50), index=True)

@custom_metadata_model("performance")
class PerformanceMetadata(Base, CustomMetadataBase):
    """System performance metrics during training."""
    __tablename__ = "custom_performance"
    run_id = Column(String(100), index=True)
    training_time_seconds = Column(Float, index=True)
    memory_usage_mb = Column(Float, index=True)
    gpu_utilization = Column(Float, index=True)

# Store both with one cache entry
cache.put(
    model,
    experiment="exp_001",
    custom_metadata=[experiment, performance]  # Both models
)

# Query each independently
with cache.query_custom_session("experiments") as query:
    high_accuracy = query.filter(ExperimentMetadata.accuracy >= 0.9).all()

with cache.query_custom_session("performance") as query:
    fast_runs = query.filter(PerformanceMetadata.training_time_seconds < 60).all()
```

### Join Across Custom Tables via cache_key

Correlate data across custom metadata tables using the `cache_key` foreign key:

```python
# Find high-accuracy experiments
with cache.query_custom_session("experiments") as query:
    high_accuracy = query.filter(
        ExperimentMetadata.accuracy >= 0.92
    ).all()
    cache_keys = {exp.cache_key for exp in high_accuracy}

# Get corresponding performance metrics
with cache.query_custom_session("performance") as query:
    perf_metrics = query.filter(
        PerformanceMetadata.cache_key.in_(cache_keys)
    ).all()
    
    for perf in perf_metrics:
        print(f"Run: {perf.run_id}, Time: {perf.training_time_seconds}s")
```

### Validation Warnings

Cacheness validates custom metadata models and warns about potential issues:

```python
# This will generate warnings:
@custom_metadata_model("test")
class BadExample(Base, CustomMetadataBase):
    __tablename__ = "experiments"  # ⚠️  Should be "custom_experiments"
    accuracy = Column(Float)       # ⚠️  No index on frequently queried column
```

Fix warnings by following the best practices above for optimal performance and maintainability.

Custom metadata transforms cacheness from a simple cache into a powerful data management and experiment tracking system. Use it to organize your cache, track data lineage, monitor quality, and enable sophisticated analysis of your cached artifacts.
