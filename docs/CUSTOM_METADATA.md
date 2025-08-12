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

Custom metadata transforms cacheness from a simple cache into a powerful data management and experiment tracking system. Use it to organize your cache, track data lineage, monitor quality, and enable sophisticated analysis of your cached artifacts.
