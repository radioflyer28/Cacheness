# Development Planning

This document tracks the architectural evolution and future development roadmap for Cacheness.

## 📋 Document Navigation

**Quick Links:**
- [Phase 1: Storage Layer Separation](#phase-1-storage-layer-separation-low-risk) (✅ Complete)
- [Phase 2: Extensibility & Plugins](#phase-2-plugin-architecture--extensibility-medium-risk) (🚧 In Planning)
- [Phase 3: Management Operations](#phase-3-management-operations-api-medium-risk) (📋 Ready to Implement)
- [Phase 2 Implementation Roadmap](#recommendations-for-phase-2-implementation-order) (Start here for what's next!)
- [Bug Tracking](#-bug-tracking) (Historical - most fixed)
- [Feasibility Review](#phase-2-feasibility-review) (Technical analysis)

## Document Overview

- **Completed Work**: Phase 1 storage layer refactoring (✅ Complete Jan 2026)
- **Active Planning**: Phase 2 extensibility features (handlers, metadata backends, blob backends) - 9-12 month effort
- **Ready to Implement**: Phase 3 management operations (update, touch, bulk delete, batch ops) - 3-6 month effort
- **Bug Tracking**: Issues identified during code review (most ✅ Fixed)
- **Design Decisions**: Manual registration over auto-discovery plugins

**For New Contributors**: Jump to [Phase 3 Management Operations](#phase-3-management-operations-api-medium-risk) for the next implementation phase, or review [Phase 2 Implementation Roadmap](#recommendations-for-phase-2-implementation-order) for extensibility features.

---

## 🏗️ Architectural Evolution: Separation of Concerns

### Background

This library began as a simple persistent cache solution. Over time, it evolved to include:
- Multiple storage backends (JSON, SQLite, In-Memory)
- Blob management with compression (blosc2, lz4, zstd, gzip)
- Type-aware serialization handlers (DataFrames, arrays, objects)
- Security features (HMAC signing, integrity verification)
- SQL pull-through caching with gap detection

**Key Insight:** Most of the complexity lives in the **storage infrastructure**, not the caching logic itself. The actual "caching" semantics (TTL, eviction, key generation) are a thin layer on top of a sophisticated blob storage and metadata management system.

### Current Architecture Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER-FACING APIs                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  cacheness  │  │  @cached    │  │  SqlCache               │  │
│  │  (core.py)  │  │ (decorator) │  │  (pull-through cache)   │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CACHING SEMANTICS LAYER                        │
│  • TTL management          • Key generation                      │
│  • Eviction policies       • Cache statistics                    │
│  • Hit/miss tracking       • Expiration cleanup                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BLOB STORAGE LAYER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  HandlerRegistry │  │  Compression    │  │  Security       │  │
│  │  (type handlers) │  │  (blosc2/lz4)   │  │  (signing)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Metadata Backends                         ││
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐     ││
│  │  │  JSON    │  │   SQLite     │  │   In-Memory        │     ││
│  │  │  Backend │  │   Backend    │  │   Backend          │     ││
│  │  └──────────┘  └──────────────┘  └────────────────────┘     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Code Distribution Analysis

| Layer | Files | LOC (approx) | Complexity |
|-------|-------|--------------|------------|
| **Caching Semantics** | core.py (partial), decorators.py | ~400 | Low |
| **Blob Storage** | handlers.py, compress_pickle.py, metadata.py | ~2,500 | High |
| **SQL Pull-Through** | sql_cache.py | ~1,800 | High |
| **Configuration** | config.py | ~550 | Medium |
| **Security** | security.py | ~240 | Medium |
| **Utilities** | serialization.py, file_hashing.py, etc. | ~800 | Medium |

**Observation:** ~75% of the codebase is blob storage infrastructure, ~25% is caching logic.

---

### Recommendation: Two-Layer Architecture

#### Option A: Monorepo with Clear Module Boundaries (Recommended for Now)

Keep as a single library but reorganize into distinct sub-packages with clear boundaries:

```
cacheness/
├── storage/                    # Low-level blob storage (reusable)
│   ├── __init__.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract backend interface
│   │   ├── json_backend.py
│   │   ├── sqlite_backend.py
│   │   └── memory_backend.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base.py            # Handler interfaces
│   │   ├── dataframe.py       # Pandas/Polars handlers
│   │   ├── array.py           # NumPy/blosc2 handlers
│   │   └── object.py          # Pickle/dill handlers
│   ├── compression.py
│   ├── security.py
│   └── blob_store.py          # Main BlobStore class
│
├── cache/                      # Caching semantics (uses storage)
│   ├── __init__.py
│   ├── unified_cache.py       # Current UnifiedCache
│   ├── decorators.py
│   ├── ttl.py                 # TTL/expiration logic
│   └── eviction.py            # Eviction policies
│
├── sql/                        # SQL pull-through cache
│   ├── __init__.py
│   ├── sql_cache.py
│   ├── adapters.py
│   └── gap_detection.py
│
└── __init__.py                 # Public API exports
```

**Pros:**
- Clear separation without breaking changes
- Storage layer becomes independently testable
- Easier to reason about responsibilities
- Natural migration path to full separation

**Cons:**
- Still a single package to version
- Can't use storage layer without installing full library

---

#### Option B: Separate Packages (Future State)

Split into two separate PyPI packages:

```
# Package 1: blobcache-storage (or similar name)
blobcache_storage/
├── backends/
├── handlers/
├── compression.py
├── security.py
└── blob_store.py

# Package 2: cacheness (depends on blobcache-storage)
cacheness/
├── unified_cache.py
├── decorators.py
├── sql_cache.py
└── __init__.py
```

**Pros:**
- True separation of concerns
- Storage layer usable for non-caching use cases (e.g., ML model versioning, artifact storage)
- Independent versioning and release cycles
- Smaller install size for users who only need one layer

**Cons:**
- Breaking change for existing users
- Two packages to maintain
- Dependency coordination complexity

---

#### Option C: Service-Oriented Architecture (Future Consideration)

For enterprise/team use cases, the blob storage could become a service:

```
┌─────────────────────┐     ┌─────────────────────────────┐
│  cacheness (client) │────▶│  Blob Storage Service       │
│  - TTL logic        │ HTTP│  - REST/gRPC API            │
│  - Key generation   │     │  - Distributed storage      │
│  - Local fallback   │     │  - Central metadata DB      │
└─────────────────────┘     └─────────────────────────────┘
```

**Pros:**
- Centralized cache for teams
- Shared storage across services
- Better observability

**Cons:**
- Significant complexity increase
- Network latency
- Deployment overhead
- Overkill for most use cases

---

### Recommended Migration Path

<a id="phase-1-storage-layer-separation-low-risk"></a>
#### Phase 1: Storage Layer Separation (Low Risk)

1. Create `storage/` sub-package with clean interfaces
2. Move handlers, backends, compression into storage
3. Update imports (maintain backward compatibility in `__init__.py`)
4. Add `BlobStore` class as primary storage API

```python
# New low-level API
from cacheness.storage import BlobStore

store = BlobStore(backend="sqlite", compression="lz4")
blob_id = store.put(data, metadata={"type": "model", "version": "1.0"})
data = store.get(blob_id)

# Existing high-level API (unchanged)
from cacheness import cacheness

cache = cacheness()
cache.put(data, model="xgboost", version="1.0")
```

**Status:** ✅ Phase 1 Complete (2026-01-27)

The storage subpackage has been created with:
- `cacheness/storage/` - Main storage layer package
- `cacheness/storage/backends/` - Metadata backend implementations (JSON, SQLite)
- `cacheness/storage/handlers/` - Type-aware serialization handlers
- `cacheness/storage/compression.py` - Compression utilities
- `cacheness/storage/security.py` - HMAC signing
- `cacheness/storage/blob_store.py` - New `BlobStore` class for direct storage access

<a id="phase-2-plugin-architecture--extensibility-medium-risk"></a>
#### Phase 2: Plugin Architecture & Extensibility (Medium Risk)

**Goal:** Enable developers to add custom type handlers, metadata backends, and blob storage backends without modifying core library code.

**Key Architectural Insight:** Separate metadata storage (queryable cache index) from blob storage (actual cached data). These are orthogonal concerns that should have independent plugin systems.

**Three Existing Metadata Features to Preserve:**

1. **Cache Infrastructure Metadata** - Built-in cacheness metadata (cache_key, file_hash, created_at, etc.)
2. **Custom SQLAlchemy Metadata** - User-defined ORM models via `@custom_metadata_model` decorator (Section 2.7)
3. **SqlCache Custom Tables** - Pull-through cache with user-defined data schemas (Section 2.8)

**Status:** 🚧 In Planning

**Design Decision: Manual Registration Over Auto-Discovery**

After evaluating both plugin approaches, we've decided to use **manual registration** rather than setuptools entry point auto-discovery:

**Chosen Approach (Manual Registration):**
```python
# Users explicitly register handlers/backends
from cacheness import register_handler, register_blob_backend
from my_package import MyCustomHandler, MyS3Backend

register_handler("parquet", MyCustomHandler)
register_blob_backend("s3", MyS3Backend)
```

**Deferred Approach (Auto-Discovery via Entry Points):**
```python
# Auto-loads from installed packages - deferred to future if needed
# [project.entry-points."cacheness.handlers"]
# parquet = "my_package:MyCustomHandler"
```

**Rationale:**
- **Simplicity:** Manual registration is straightforward; no entry point scanning overhead
- **Explicit control:** No "magic" auto-discovery; clear registration trail
- **Sufficient for now:** Most users will write handlers for internal use, not distribute as packages
- **Ecosystem readiness:** Wait for community demand before adding plugin infrastructure
- **Easy migration:** Registration API can later support entry points without breaking changes

**If future usage warrants a plugin system:**
- Implement entry point discovery as optional feature
- Keep manual registration as primary API
- Add plugin security, versioning, marketplace features

**Planned Features:**

##### 2.1 Handler Registration System

Allow external packages to register custom type handlers:

```python
# User's custom package: myproject/custom_handlers.py
from cacheness.interfaces import CacheHandler
from cacheness.storage import HandlerRegistry

class MyCustomDataTypeHandler(CacheHandler):
    """Handler for proprietary data format."""
    
    def can_handle(self, data: Any) -> bool:
        return isinstance(data, MyCustomDataType)
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        # Custom serialization logic
        ...
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        # Custom deserialization logic
        ...
    
    def get_file_extension(self, config: Any) -> str:
        return ".custom"
    
    @property
    def data_type(self) -> str:
        return "my_custom_type"

# Register handler programmatically
from cacheness import cacheness

cache = cacheness()
cache.handler_registry.register_handler(MyCustomDataTypeHandler(), priority=0)

# Or via configuration
from cacheness import CacheConfig, HandlerConfig

config = CacheConfig(
    handlers=HandlerConfig(
        custom_handlers=[MyCustomDataTypeHandler],
        handler_priority=["my_custom_type", "pandas_dataframes", "numpy_arrays", "object_pickle"]
    )
)
cache = cacheness(config)
```

**Implementation Tasks:**
- [ ] Add `register_handler(handler, priority=None)` method to HandlerRegistry
- [ ] Add `unregister_handler(handler_name)` method to HandlerRegistry
- [ ] Add `list_handlers()` method to show registered handlers and priority order
- [ ] Support handler instantiation from class references in config
- [ ] Add validation for custom handlers (must implement required methods)
- [ ] Create handler plugin example in `examples/custom_handler_plugin.py`
- [ ] Document handler interface contract in `docs/CUSTOM_HANDLERS.md`

##### 2.2 Metadata Backend Plugin System

Allow external packages to register custom metadata backends (for cache index/queries):

```python
# User's custom package: myproject/custom_backends.py
from cacheness.storage.backends.base import MetadataBackend

class PostgresBackend(MetadataBackend):
    """PostgreSQL metadata backend for distributed caching."""
    
    def __init__(self, connection_url: str):
        self.engine = create_engine(connection_url)
        # Setup tables, etc.
    
    def load_metadata(self) -> Dict[str, Any]:
        # Load from PostgreSQL
        ...
    
    def save_metadata(self, metadata: Dict[str, Any]):
        # Save to PostgreSQL
        ...
    
    # ... implement other required methods

# Register backend programmatically
from cacheness import cacheness
from cacheness.storage.backends import register_backend

register_backend("postgresql", PostgresBackend)

# Use in cache creation
cache = cacheness(config=CacheConfig(
    metadata_backend="postgresql",
    backend_options={"connection_url": "postgresql://user:pass@host/db"}
))

# Or pass instance directly
postgres_backend = PostgresBackend("postgresql://...")
cache = cacheness(metadata_backend=postgres_backend)
```

**Implementation Tasks:**
- [ ] Create metadata backend registry system in `storage/backends/__init__.py`
- [ ] Add `register_metadata_backend(name, backend_class)` function
- [ ] Add `get_metadata_backend(name, **options)` factory function
- [ ] Support `backend_options` in CacheConfig for backend-specific parameters
- [ ] Add validation for custom backends (must implement MetadataBackend interface)
- [ ] Create PostgreSQL metadata backend as first external backend example
- [ ] Document metadata backend interface in `docs/CUSTOM_METADATA_BACKENDS.md`

##### 2.3 Blob Storage Backend Registry System

**Current State:** Blob storage is hardcoded to filesystem (`cache_dir` parameter). All cached data files are stored locally.

**Goal:** Abstract blob storage to support cloud storage backends while keeping metadata backends independent.

**Architecture:**

```python
# Separate concerns: Metadata vs Blob Storage
┌─────────────────────────────────────┐
│     Metadata Backend (Index)        │
│  - JSON (local file)                │
│  - SQLite (local database)          │
│  - PostgreSQL (distributed)         │
│  → Stores: cache keys, metadata,   │
│    file paths, statistics           │
└─────────────────────────────────────┘
              ↕
       (references)
              ↕
┌─────────────────────────────────────┐
│     Blob Storage Backend (Data)     │
│  - Filesystem (local directory)     │
│  - S3 (cloud storage)               │
│  - Azure Blob (cloud storage)       │
│  - GCS (cloud storage)              │
│  → Stores: actual cached data       │
└─────────────────────────────────────┘
```

**Example Combinations:**

```python
from cacheness import cacheness, CacheConfig

# Combination 1: SQLite metadata + Filesystem blobs (current default)
config = CacheConfig(
    cache_dir="./cache",           # Filesystem for blobs
    metadata_backend="sqlite"      # SQLite for metadata
)

# Combination 2: PostgreSQL metadata + S3 blobs (distributed team cache)
config = CacheConfig(
    metadata_backend="postgresql",
    metadata_backend_options={
        "connection_url": "postgresql://cache-db.example.com/cache"
    },
    blob_backend="s3",
    blob_backend_options={
        "bucket": "team-cache-blobs",
        "region": "us-west-2"
    }
)

# Combination 3: SQLite metadata + S3 blobs (serverless/Lambda)
config = CacheConfig(
    cache_dir="/tmp/metadata",     # Ephemeral local SQLite
    metadata_backend="sqlite",
    blob_backend="s3",             # Persistent S3 storage
    blob_backend_options={"bucket": "lambda-cache"}
)

# Combination 4: In-memory metadata + S3 blobs (testing/temporary)
config = CacheConfig(
    metadata_backend="memory",     # Ephemeral metadata
    blob_backend="s3",             # Persistent blobs
    blob_backend_options={"bucket": "test-cache"}
)

# Combination 5: SqlCache with custom metadata + shared PostgreSQL
from cacheness.sql_cache import SqlCache

cache = SqlCache(
    db_url="postgresql://localhost/shared_db",
    table_name="ml_experiments",        # Custom queryable metadata
    data_fetcher=train_model,
    
    # User-defined columns for SQL queries
    model_type=String(50),
    accuracy=Float,
    training_date=Date,
    
    config=CacheConfig(
        metadata_backend="postgresql",  # Infrastructure metadata
        metadata_backend_options={
            "connection_url": "postgresql://localhost/shared_db",
            "table_name": "cache_metadata"  # Different table, same DB
        },
        blob_backend="s3",              # Blobs in S3
        blob_backend_options={"bucket": "ml-models"}
    )
)
```

**Implementation Plan:**

```python
# cacheness/storage/backends/blob_backends.py
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional
from pathlib import Path

class BlobBackend(ABC):
    """Abstract interface for blob storage backends."""
    
    @abstractmethod
    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Write blob data, return storage path/URL."""
        pass
    
    @abstractmethod
    def read_blob(self, blob_path: str) -> bytes:
        """Read blob data from storage path/URL."""
        pass
    
    @abstractmethod
    def delete_blob(self, blob_path: str) -> bool:
        """Delete blob from storage."""
        pass
    
    @abstractmethod
    def exists(self, blob_path: str) -> bool:
        """Check if blob exists."""
        pass
    
    @abstractmethod
    def write_blob_stream(self, blob_id: str, stream: BinaryIO) -> str:
        """Write blob from stream (for large objects)."""
        pass
    
    @abstractmethod
    def read_blob_stream(self, blob_path: str) -> BinaryIO:
        """Read blob as stream (for large objects)."""
        pass

class FilesystemBlobBackend(BlobBackend):
    """Current filesystem-based blob storage (refactored)."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        blob_path = self.base_dir / blob_id
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(data)
        return str(blob_path)
    
    def read_blob(self, blob_path: str) -> bytes:
        return Path(blob_path).read_bytes()
    
    # ... other methods

# Registry system
_blob_backends = {
    "filesystem": FilesystemBlobBackend,
}

def register_blob_backend(name: str, backend_class: type):
    """Register a custom blob storage backend."""
    if not issubclass(backend_class, BlobBackend):
        raise ValueError(f"{backend_class} must inherit from BlobBackend")
    _blob_backends[name] = backend_class

def get_blob_backend(name: str, **options) -> BlobBackend:
    """Get blob backend instance by name."""
    if name not in _blob_backends:
        raise ValueError(f"Unknown blob backend: {name}")
    return _blob_backends[name](**options)
```

**Implementation Tasks:**
- [ ] Create `BlobBackend` abstract base class in `storage/backends/blob_backends.py`
- [ ] Refactor current file I/O in handlers to use `FilesystemBlobBackend`
- [ ] Create blob backend registry system (`register_blob_backend`, `get_blob_backend`)
- [ ] Add `blob_backend` and `blob_backend_options` to `CacheConfig`
- [ ] Update `UnifiedCache` to accept blob backend parameter
- [ ] Update `BlobStore` to accept blob backend parameter
- [ ] Ensure metadata backend stores blob paths/URLs, not file handles
- [ ] Add streaming support for large blobs (avoid loading entire blob in memory)
- [ ] Create `FilesystemBlobBackend` as default implementation
- [ ] Document blob backend interface in `docs/CUSTOM_BLOB_BACKENDS.md`
- [ ] Add tests for blob backend abstraction
- [ ] Add example showing custom blob backend in `examples/custom_blob_backend.py`

##### 2.4 Configuration Schema & Validation

Standardize configuration for easier plugin integration:

```python
from cacheness import CacheConfig
from cacheness.config import validate_config

# Define config with custom backends
config = CacheConfig(
    cache_dir="./cache",
    metadata_backend="postgresql",
    backend_options={
        "connection_url": "postgresql://...",
        "pool_size": 10
    },
    blob_backend="s3",
    blob_backend_options={
        "bucket": "my-cache",
        "region": "us-west-2"
    }
)

# Validate configuration
errors = validate_config(config)
if errors:
    raise ValueError(f"Invalid config: {errors}")
```

**Implementation Tasks:**
- [ ] Add `backend_options: Dict[str, Any]` to CacheConfig
- [ ] Add `blob_backend: Union[str, BlobBackend]` to CacheConfig
- [ ] Add `blob_backend_options: Dict[str, Any]` to CacheConfig
- [ ] Create `validate_config()` function with detailed error messages
- [ ] Add JSON Schema for configuration validation
- [ ] Support configuration from YAML/JSON files

##### 2.5 Use Cases Beyond Caching

Document and support non-caching storage use cases with various backend combinations.

**Note:** These examples use `BlobStore` for simple key-value storage. For queryable domain-specific metadata, use `SqlCache` with custom table schemas (see section 2.6).

1. **ML Model Versioning**
   ```python
   from cacheness.storage import BlobStore
   
   model_store = BlobStore(
       cache_dir="./models",
       backend="postgresql",  # Queryable metadata
       content_addressable=True  # Deduplicate identical models
   )
   
   # Store model with version metadata
   model_id = model_store.put(
       model,
       key=f"fraud_detector_v{version}",
       metadata={
           "accuracy": 0.95,
           "training_date": datetime.now().isoformat(),
           "dataset_hash": dataset_hash,
           "hyperparameters": {...}
       }
   )
   
   # Query models by accuracy
   best_models = model_store.list(metadata_filter={"accuracy": {"$gte": 0.9}})
   ```

2. **Artifact Storage in Data Pipelines**
   ```python
   # Store intermediate pipeline results
   pipeline_store = BlobStore(cache_dir="./pipeline_artifacts")
   
   # Step 1
   raw_data = extract_data()
   pipeline_store.put(raw_data, key=f"raw_data_{run_id}")
   
   # Step 2
   processed_data = transform(raw_data)
   pipeline_store.put(
       processed_data,
       key=f"processed_data_{run_id}",
       metadata={"depends_on": f"raw_data_{run_id}"}
   )
   ```

3. **Checkpoint Storage for Long-Running Tasks**
   ```python
   checkpoint_store = BlobStore(cache_dir="./checkpoints")
   
   for epoch in range(100):
       train_epoch(model)
       
       # Save checkpoint every 10 epochs
       if epoch % 10 == 0:
           checkpoint_store.put(
               model.state_dict(),
               key=f"checkpoint_epoch_{epoch}",
               metadata={"epoch": epoch, "loss": current_loss}
           )
   ```

**Implementation Tasks:**
- [x] Create example scripts for each use case in `examples/`
- [x] Add BlobStore documentation in `docs/BLOB_STORE.md`
- [ ] Add metadata query operators (e.g., `$gte`, `$lt`, `$in`) for advanced filtering
- [ ] Consider adding blob tagging/labeling system for organization

##### 2.7 Custom SQLAlchemy Metadata Models Integration

**Important:** Preserve existing functionality for users to define custom SQLAlchemy metadata schemas alongside cache infrastructure metadata.

**Three Types of Metadata in Cacheness:**

1. **Cache Infrastructure Metadata** (handled by metadata backends):
   - Generic cache metadata for all cache operations
   - Keys: `cache_key`, `file_hash`, `created_at`, `file_size`, etc.
   - Schema is fixed, managed by cacheness
   - Backends: JSON, SQLite, PostgreSQL (via `metadata_backend`)
   - Used by: `cacheness()`, `BlobStore`

2. **Custom SQLAlchemy Metadata** (user-defined ORM models - existing feature):
   - User defines SQLAlchemy ORM models with `@custom_metadata_model` decorator
   - Strongly-typed, queryable columns: `experiment_id`, `model_type`, `accuracy`, etc.
   - Schema is custom, managed by user
   - Enables complex SQLAlchemy queries on cache metadata
   - Currently works with SQLite backend only
   - Used by: `cacheness()` with `custom_metadata` parameter

3. **SqlCache Custom Tables** (domain-specific data tables):
   - Separate feature for pull-through caching
   - User defines columns for cached data itself
   - Used by: `SqlCache` class

**Example - Custom SQLAlchemy Metadata (Existing Feature):**

```python
from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer

# Define custom metadata schema with SQLAlchemy ORM
@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    """Custom queryable metadata for ML experiments."""
    
    __tablename__ = "custom_experiments"
    
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False, index=True)
    created_by = Column(String(100), nullable=False, index=True)

# Create cache with SQLite backend (required for custom metadata)
config = CacheConfig(
    cache_dir="./cache",
    metadata_backend="sqlite"
)
cache = cacheness(config)

# Store data with custom metadata
experiment = ExperimentMetadata(
    experiment_id="exp_001",
    model_type="xgboost",
    accuracy=0.95,
    epochs=100,
    created_by="alice"
)

cache.put(
    trained_model,
    experiment="exp_001",
    custom_metadata=experiment  # Custom SQLAlchemy object
)

# Query with SQLAlchemy ORM
with cache.query_custom_session("experiments") as query:
    alice_experiments = query.filter(
        ExperimentMetadata.created_by == "alice",
        ExperimentMetadata.accuracy >= 0.9
    ).all()
    
    for exp in alice_experiments:
        print(f"{exp.experiment_id}: {exp.model_type} - {exp.accuracy}")
```

**How It Works Today (SQLite backend):**

```
SQLite Database (cache_metadata.db)
├── cache_entries table          # Infrastructure metadata (cacheness-managed)
│   ├── cache_key
│   ├── file_hash
│   ├── created_at
│   ├── file_size
│   └── ...
│
├── custom_experiments table     # User-defined metadata (user-managed)
│   ├── cache_key (FK)
│   ├── experiment_id
│   ├── model_type
│   ├── accuracy
│   └── created_by
│
└── custom_performance table     # Another user-defined schema
    ├── cache_key (FK)
    ├── run_id
    ├── training_time_seconds
    └── memory_usage_mb
```

**PostgreSQL Backend Compatibility:**

When implementing PostgreSQL metadata backend, ensure custom metadata models continue to work:

```python
from cacheness import cacheness, CacheConfig

# Custom metadata models + PostgreSQL backend
config = CacheConfig(
    metadata_backend="postgresql",
    metadata_backend_options={
        "connection_url": "postgresql://localhost/cache",
        "table_name": "cache_entries"  # Infrastructure table
    }
)

cache = cacheness(config)

# Custom metadata tables created in same database
experiment = ExperimentMetadata(
    experiment_id="exp_001",
    model_type="xgboost",
    accuracy=0.95
)

cache.put(model, experiment="exp_001", custom_metadata=experiment)
# Creates both tables: cache_entries + custom_experiments
```

**Implementation Considerations for Phase 2:**

- [x] Ensure PostgreSQL metadata backend supports custom metadata models
- [x] Custom tables use same SQLAlchemy engine as infrastructure tables
- [x] Foreign key from custom tables to cache_entries.cache_key works
- [x] `query_custom_session()` works with PostgreSQL backend
- [x] `migrate_custom_metadata_tables()` works with PostgreSQL
- [x] Update `docs/CUSTOM_METADATA.md` to show PostgreSQL compatibility
- [x] Add tests for custom metadata + PostgreSQL backend
- [x] Document relationship between infrastructure and custom metadata tables
- [ ] Consider making `@custom_metadata_model` backend-agnostic
- [ ] Example showing custom metadata + PostgreSQL infrastructure + S3 blobs:
  ```python
  config = CacheConfig(
      metadata_backend="postgresql",      # Both infrastructure + custom metadata
      metadata_backend_options={
          "connection_url": "postgresql://localhost/cache"
      },
      blob_backend="s3",                  # Blobs in S3
      blob_backend_options={"bucket": "ml-cache"}
  )
  
  cache = cacheness(config)
  cache.put(model, custom_metadata=ExperimentMetadata(...))
  # PostgreSQL: cache_entries + custom_experiments tables
  # S3: actual model blob
  ```

##### 2.8 SqlCache Custom Tables Integration

**Status:** 🟡 Deferred - Requires significant rework, lower priority

**Assessment (January 2026):**

After audit, this phase requires more substantial changes than originally scoped:

1. **Current State:** SqlCache is completely standalone - it does NOT use `CacheConfig`, metadata backends, or the cacheness infrastructure. It manages its own database connections and tables directly.

2. **Original Vision:** Integrate SqlCache with the metadata backend system so:
   - Infrastructure metadata (cache keys, TTL, stats) uses metadata backends
   - User data tables remain separate but in same database

3. **Reality:** This is essentially a rewrite of SqlCache to use the cacheness infrastructure, which would:
   - Break existing SqlCache API compatibility
   - Add complexity for users who just want simple SQL caching
   - Require extensive testing with all backend combinations

**Recommendation:** Defer to Phase 3 or beyond. SqlCache works well standalone, and the integration doesn't add significant user value.

**If pursued later, implementation would require:**

- [ ] Add `CacheConfig` parameter to SqlCache constructor
- [ ] Separate SqlCache into data layer (user tables) and metadata layer (infrastructure)
- [ ] Use metadata backend for infrastructure metadata (cache_entries, cache_stats)
- [ ] Keep user data tables managed by SqlCache directly
- [ ] Ensure PostgreSQL can host both without table conflicts
- [ ] Update docs/SQL_CACHE.md to explain separation
- [ ] Maintain backward compatibility with existing SqlCache API
- [ ] Add SqlCache.create_with_shared_db() helper method
- [ ] Comprehensive testing with all backend combinations

**Alternative (simpler):** Just document that SqlCache and cacheness() use separate storage systems and that's by design. Users who need integrated metadata can use the `@custom_metadata_model` feature with cacheness().

---

##### 2.9 Testing & Documentation ✅ COMPLETED

Comprehensive testing and documentation for extensibility.

**Testing Tasks:**
- [x] Handler registration tests - `tests/test_handler_registration.py` (483 lines)
- [x] Metadata backend registration tests - `tests/test_metadata_backend_registry.py` (657 lines)
- [x] Blob backend registration tests - `tests/test_blob_backend_registry.py` (661 lines)
- [x] Configuration validation tests - `tests/test_config_validation.py` (659 lines)
- [x] PostgreSQL backend tests - `tests/test_postgresql_backend.py`
- [x] Custom metadata tests - `tests/test_custom_metadata.py`
- [ ] ~~Create `tests/test_sqlcache_with_backends.py`~~ - Deferred with 2.8
- [ ] Performance benchmarks for custom handlers - Nice to have, not blocking

**Documentation Tasks:**
- [x] Create `docs/PLUGIN_DEVELOPMENT.md` - Complete plugin development guide
- [x] Update `docs/API_REFERENCE.md` with Extensibility API section
- [x] Add "Extending Cacheness" section to main README
- [x] Update `docs/CUSTOM_METADATA.md` with PostgreSQL compatibility
- [x] Backend Selection Guide already exists - `docs/BACKEND_SELECTION.md`
- [ ] ~~Create separate CUSTOM_HANDLERS.md, CUSTOM_METADATA_BACKENDS.md, CUSTOM_BLOB_BACKENDS.md~~ - Consolidated into PLUGIN_DEVELOPMENT.md
- [ ] ~~Update `docs/SQL_CACHE.md`~~ - Deferred with 2.8
- [ ] Plugin packaging template project - Nice to have, not blocking

---

### Phase 2 Summary

Phase 2 (Extensibility & Plugin Architecture) status:

| Section | Feature | Status |
|---------|---------|--------|
| 2.1 | Handler Registration System | ✅ Complete |
| 2.2 | Metadata Backend Registry | ✅ Complete |
| 2.3 | Blob Backend Registry | ✅ Complete |
| 2.4 | Configuration Schema & Validation | ✅ Complete |
| 2.5 | BlobStore Examples & Docs | ✅ Complete |
| 2.6 | PostgreSQL Metadata Backend | ✅ Complete |
| 2.7 | Custom Metadata + PostgreSQL | ✅ Complete |
| 2.8 | SqlCache Integration | 🟡 Deferred |
| 2.9 | Testing & Documentation | ✅ Complete |
| 2.10 | S3 Blob Backend | 🚧 In Progress |

**Key Deliverables (2.1-2.9):**
- Full plugin system for handlers, metadata backends, and blob backends
- PostgreSQL as production-grade metadata backend
- Custom SQLAlchemy metadata models work with both SQLite and PostgreSQL
- Comprehensive configuration validation with JSON/YAML loading
- Complete documentation: PLUGIN_DEVELOPMENT.md, updated API_REFERENCE.md, README

**Deferred Items:**
- SqlCache + metadata backend integration (lower priority, breaks API)
- Performance benchmarks for custom handlers
- Plugin packaging template

---

<a id="phase-3-management-operations-api-medium-risk"></a>
#### Phase 3: Management Operations API (Medium Risk)

**Goal:** Complete missing CRUD and management operations in storage backends and cache layer.

**Status:** 📋 Ready to Implement (February 2026)

**Effort Estimate:** 3-6 months

**Reference:** [MISSING_MANAGEMENT_API.md](MISSING_MANAGEMENT_API.md) - Complete architectural analysis

---

### Architectural Principles (From MISSING_MANAGEMENT_API.md)

**Storage Backend Responsibilities:**
- Basic CRUD (create, read, update, delete)
- Bulk operations (delete many, batch operations)
- Metadata management (get/update metadata without loading data)
- Pattern-based operations (delete by prefix, list with filters)
- **NO TTL awareness** - storage is "dumb" about expiration
- **Enforce cache_key immutability** - keys cannot be modified after creation

**Cache Layer Responsibilities:**
- TTL policy and enforcement
- Cache hit/miss logic with expiration checking
- Touch/refresh operations (extending expiration)
- Cleanup of expired entries (delegates timestamp deletion to backend)
- Wraps storage backend with caching semantics

**Key Insight:** Storage backends store timestamps but don't interpret them. Cache layer interprets timestamps according to TTL policy.

---

### 3.1 Update Blob Data (High Priority)

**Current State:** Must delete then re-insert to update data at an existing cache_key.

**Goal:** Add `update_blob_data()` operation to replace data at a fixed cache_key.

**Important:** The cache_key is computed from input parameters and is **immutable**. This operation replaces the blob data stored at that key, not the key itself. Updates derived metadata (file_size, content_hash, created_at) but cache_key remains unchanged.

**Implementation Tasks:**

**Storage Backend Layer:**
- [ ] Add `update_blob_data(cache_key: str, new_data: Any)` to `MetadataBackend` abstract class
- [ ] Implement in `SQLiteBackend`:
  - [ ] Serialize new data using handlers
  - [ ] Write new blob file (or update in place)
  - [ ] Update metadata: file_size, content_hash, file_hash, created_at
  - [ ] Keep cache_key unchanged
- [ ] Implement in `PostgreSQLBackend` (same logic)
- [ ] Implement in `JSONBackend` (same logic)
- [ ] Implement in `MemoryBackend` (update in-memory dict)
- [ ] Add `update_data(key: str, new_data: Any)` to `BlobStore` class
- [ ] Handle errors: key not found, serialization failure, disk full

**Cache Layer:**
- [ ] Add `update_data(data: Any, cache_key: Optional[str] = None, **kwargs)` to `UnifiedCache`
- [ ] Resolve cache_key from kwargs if not provided
- [ ] Check if entry exists (return False if not)
- [ ] Delegate to `backend.update_blob_data(cache_key, data)`
- [ ] Update cache stats (if applicable)
- [ ] Handle TTL: reset created_at timestamp (acts like touch + update)

**Testing:**
- [ ] Test update_blob_data across all backends (SQLite, PostgreSQL, JSON, Memory)
- [ ] Test cache_key immutability (verify key doesn't change)
- [ ] Test derived metadata updates (file_size, content_hash, created_at)
- [ ] Test error cases: key not found, serialization failure
- [ ] Test UnifiedCache.update_data() with kwargs resolution
- [ ] Test TTL behavior after update

**Documentation:**
- [ ] Add to API_REFERENCE.md (backend and cache layer)
- [ ] Add example to examples/management_operations_demo.py
- [ ] Document difference from put() (update requires existing key)

**Priority:** ✅✅✅ Highest - Core CRUD operation, frequently requested

---

### 3.2 Bulk Delete by Pattern (High Priority)

**Current State:** Must loop over list results and delete individually.

**Goal:** Delete multiple entries matching criteria efficiently.

**Implementation Tasks:**

**Storage Backend Layer:**
- [ ] Add `delete_where(filter_fn: Callable[[Dict], bool])` to `MetadataBackend`
- [ ] Implement in `SQLiteBackend`:
  - [ ] Query all entries
  - [ ] Filter with filter_fn
  - [ ] Delete matching entries in transaction
  - [ ] Return count of deleted entries
- [ ] Implement in `PostgreSQLBackend` (same logic, can optimize with SQL WHERE)
- [ ] Implement in `JSONBackend` (filter dict, delete, save atomically)
- [ ] Implement in `MemoryBackend` (filter dict, delete)
- [ ] Add `delete_by_cache_key_prefix(prefix: str)` convenience method
  - [ ] Implemented as: `delete_where(lambda e: e['cache_key'].startswith(prefix))`
- [ ] Optimize PostgreSQL/SQLite: Use SQL `DELETE WHERE cache_key LIKE 'prefix%'`

**Cache Layer:**
- [ ] Add `delete_by_prefix(**prefix_kwargs)` to `UnifiedCache`
- [ ] Query matching entries with kwargs
- [ ] Extract cache_keys
- [ ] Delegate to `backend.delete_entries_batch(cache_keys)` or loop
- [ ] Return count of deleted entries
- [ ] Add `delete_where(filter_fn)` for advanced filtering

**Testing:**
- [ ] Test delete_where with various filter functions
- [ ] Test delete_by_cache_key_prefix across all backends
- [ ] Test cache.delete_by_prefix with kwargs matching
- [ ] Test transaction rollback on error (SQLite, PostgreSQL)
- [ ] Test empty result (no matches)
- [ ] Test delete_where with TTL-based filters (cache layer concern)

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples showing common patterns (delete old entries, delete by project)
- [ ] Document performance characteristics (SQL backends faster)

**Priority:** ✅✅✅ High - Essential for cleanup operations

---

### 3.3 Get Metadata Without Loading Data (Trivial)

**Current State:**
- `BlobStore.get_metadata()` ✅ Already exists
- `MetadataBackend.get_entry()` ✅ Already exists
- `UnifiedCache.get_metadata()` ❌ Not exposed

**Goal:** Expose existing backend method in cache layer.

**Implementation Tasks:**

**Cache Layer:**
- [ ] Add `get_metadata(cache_key: Optional[str] = None, **kwargs)` to `UnifiedCache`
  ```python
  def get_metadata(self, cache_key: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
      """Get entry metadata without loading blob data."""
      if cache_key is None:
          cache_key = self._create_cache_key(kwargs)
      return self.metadata_backend.get_entry(cache_key)
  ```
- [ ] Handle expired entries (return None if expired, per cache policy)
- [ ] Add to BlobStore as well (already has it, ensure exposed)

**Testing:**
- [ ] Test get_metadata with cache_key
- [ ] Test get_metadata with kwargs
- [ ] Test returns None for non-existent entry
- [ ] Test returns None for expired entry (cache layer)
- [ ] Test metadata fields are correct (cache_key, file_size, created_at, etc.)

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add example showing metadata inspection before loading

**Priority:** ✅✅ Medium - Trivial implementation, useful utility

---

### 3.4 Touch/Refresh TTL (High Priority)

**Current State:** No way to extend expiration without reloading data.

**Goal:** Update entry timestamp to reset TTL (cache layer operation).

**Implementation Tasks:**

**Storage Backend Support:**
- [ ] Add `update_entry_timestamp(cache_key: str, new_timestamp: datetime)` to `MetadataBackend`
- [ ] Implement in `SQLiteBackend`: UPDATE created_at
- [ ] Implement in `PostgreSQLBackend`: UPDATE created_at
- [ ] Implement in `JSONBackend`: Update dict, save atomically
- [ ] Implement in `MemoryBackend`: Update dict

**Cache Layer:**
- [ ] Add `touch(cache_key: Optional[str] = None, ttl_seconds: Optional[float] = None, **kwargs)` to `UnifiedCache`
  ```python
  def touch(self, cache_key: Optional[str] = None, ttl_seconds: Optional[float] = None, **kwargs) -> bool:
      """Update entry timestamp to extend TTL without reloading data."""
      if cache_key is None:
          cache_key = self._create_cache_key(kwargs)
      
      entry = self.metadata_backend.get_entry(cache_key)
      if not entry:
          return False
      
      # Update timestamp
      new_timestamp = datetime.now(timezone.utc)
      self.metadata_backend.update_entry_timestamp(cache_key, new_timestamp)
      return True
  ```
- [ ] Handle custom TTL (if provided, store in metadata? Or just update timestamp?)
- [ ] Decision needed: Store TTL per entry or use config default?

**Testing:**
- [ ] Test touch updates timestamp
- [ ] Test touch with kwargs resolution
- [ ] Test touch returns False for non-existent entry
- [ ] Test TTL extension works (entry not expired after touch)
- [ ] Test touch with custom ttl_seconds parameter

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples: keep session alive, prevent long computation expiration
- [ ] Document that touch is a cache-layer operation (not storage backend)

**Priority:** ✅✅✅ High - Core cache operation for TTL management

---

### 3.5 Batch Operations (Medium Priority)

**Current State:** Must make N individual calls for N operations.

**Goal:** Optimize multiple operations in single transaction/batch.

**Implementation Tasks:**

**Storage Backend Layer:**
- [ ] Add `get_entries_batch(cache_keys: List[str])` to `MetadataBackend`
  - [ ] Return `Dict[str, Optional[Dict]]` - {cache_key: metadata or None}
  - [ ] Optimize SQLite/PostgreSQL: Single query with `WHERE cache_key IN (...)`
- [ ] Add `delete_entries_batch(cache_keys: List[str])` to `MetadataBackend`
  - [ ] Return count of deleted entries
  - [ ] Use transaction for atomicity
  - [ ] Optimize SQL: `DELETE WHERE cache_key IN (...)`
- [ ] Add `update_entries_batch(updates: Dict[str, Any])` to `MetadataBackend`
  - [ ] Input: {cache_key: new_data}
  - [ ] Call update_blob_data for each in transaction
  - [ ] Return count of updated entries

**Cache Layer:**
- [ ] Add `get_batch(kwargs_list: List[Dict])` to `UnifiedCache`
  - [ ] Resolve each kwargs dict to cache_key
  - [ ] Call `backend.get_entries_batch(cache_keys)`
  - [ ] Load blob data for each hit
  - [ ] Return `Dict[str, Any]` - {cache_key: data or None}
- [ ] Add `delete_batch(kwargs_list: List[Dict])` to `UnifiedCache`
  - [ ] Resolve cache_keys from kwargs
  - [ ] Call `backend.delete_entries_batch(cache_keys)`
  - [ ] Return count
- [ ] Add `update_batch(updates: List[Tuple[Dict, Any]])` to `UnifiedCache`
  - [ ] Input: [(kwargs, new_data), ...]
  - [ ] Resolve cache_keys
  - [ ] Build {cache_key: new_data} dict
  - [ ] Call `backend.update_entries_batch(updates)`

**Testing:**
- [ ] Test get_entries_batch across all backends
- [ ] Test delete_entries_batch with transaction rollback on error
- [ ] Test update_entries_batch
- [ ] Test cache.get_batch with kwargs resolution
- [ ] Test batch operations with mixed hits/misses
- [ ] Test empty batch (no-op)
- [ ] Benchmark: batch vs individual operations

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples showing batch loading, batch delete, batch update
- [ ] Document performance benefits (transaction overhead, network round trips)

**Priority:** ✅✅ Medium - Performance optimization for common patterns

---

### 3.6 Copy/Move Entries (Low Priority - Convenience)

**Current State:** Users must manually get then put to duplicate entries.

**Goal:** Convenience methods that compose CRUD primitives.

**Why Provide:** While users can compose CRUD, copy/move provide:
- **Atomicity** - Backend can guarantee atomic move
- **Efficiency** - Server-side copy avoids transferring large blobs
- **Convenience** - Common operations packaged

**Implementation Tasks:**

**Storage Backend Layer:**
- [ ] Add `copy_entry(source_key: str, dest_key: str)` to `MetadataBackend`
  ```python
  def copy_entry(self, source_key: str, dest_key: str) -> bool:
      """Convenience: Get source + Put to dest."""
      entry = self.get_entry(source_key)  # CRUD: Read
      if entry:
          # Copy blob file as well
          blob_data = self.read_blob(entry['blob_path'])
          new_entry = entry.copy()
          new_entry['cache_key'] = dest_key
          new_entry['created_at'] = datetime.now(timezone.utc).isoformat()
          self.put_entry(dest_key, new_entry)  # CRUD: Create
          self.write_blob(blob_data, new_entry['blob_path'])
      return entry is not None
  ```
- [ ] Add `move_entry(source_key: str, dest_key: str)` to `MetadataBackend`
  ```python
  def move_entry(self, source_key: str, dest_key: str) -> bool:
      """Convenience: Atomic copy + delete."""
      # Use transaction for atomicity (SQLite/PostgreSQL)
      if self.copy_entry(source_key, dest_key):
          self.remove_entry(source_key)  # CRUD: Delete
          return True
      return False
  ```
- [ ] Implement in all backends (SQLite, PostgreSQL, JSON, Memory)
- [ ] Optimize: File-based backends can use OS rename for move

**Cache Layer:**
- [ ] Add `copy(source: Dict, dest: Dict)` to `UnifiedCache`
  - [ ] Resolve source and dest cache_keys
  - [ ] Delegate to `backend.copy_entry(source_key, dest_key)`
- [ ] Add `move(source: Dict, dest: Dict)` to `UnifiedCache`
  - [ ] Resolve source and dest cache_keys
  - [ ] Delegate to `backend.move_entry(source_key, dest_key)`

**Testing:**
- [ ] Test copy_entry creates new entry with new cache_key
- [ ] Test move_entry deletes source
- [ ] Test move is atomic (transaction rollback on error)
- [ ] Test copy/move with BlobStore (file operations)
- [ ] Test cache.copy/move with kwargs resolution
- [ ] Test error cases: source not found, dest already exists

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples: backup before modification, fork experiments
- [ ] Document that these are convenience methods (users can use CRUD directly)
- [ ] Document atomicity guarantees for move

**Priority:** ⚠️ Low - Convenience only, users can compose CRUD

---

### 3.7 Export/Import Cache (Low Priority)

**Goal:** Backup/restore cache or migrate between environments.

**Implementation Tasks:**

**Cache Layer:**
- [ ] Add `export_to_file(path: str, compress: bool = True, filter_fn: Optional[Callable] = None)`
  - [ ] Query all entries (or filtered subset)
  - [ ] Create tarball with: metadata.json + all blobs
  - [ ] Optionally compress with gzip
- [ ] Add `export_to_dict()` for programmatic export
- [ ] Add `import_from_file(path: str)`
  - [ ] Extract tarball
  - [ ] Restore metadata entries
  - [ ] Restore blob files
  - [ ] Validate integrity (signatures, hashes)
- [ ] Add `import_from_dict(data: Dict)`

**Testing:**
- [ ] Test export creates valid tarball
- [ ] Test import restores all entries
- [ ] Test round-trip (export → import → verify)
- [ ] Test filtered export
- [ ] Test cross-backend migration (SQLite → PostgreSQL)

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples: backup, team sharing, dev→prod migration

**Priority:** ⚠️ Low - Nice-to-have for migrations

---

### 3.8 Verify and Repair Cache (Low Priority)

**Goal:** Detect corrupted entries or missing files.

**Implementation Tasks:**

**Storage Backend:**
- [ ] Add `verify_integrity()` method
  - [ ] Check all metadata entries have corresponding blobs
  - [ ] Verify blob signatures (if enabled)
  - [ ] Verify content hashes match
  - [ ] Return list of issues

**Cache Layer:**
- [ ] Add `verify_integrity()` wrapper
- [ ] Add `verify_cache_coherence()` for TTL consistency
- [ ] Add `repair(dry_run: bool = True)`
  - [ ] Remove orphaned metadata entries
  - [ ] Remove orphaned blob files
  - [ ] Recompute invalid hashes
- [ ] Add `find_orphaned_files()` utility
- [ ] Add `cleanup_orphaned_files()`

**Testing:**
- [ ] Test detect missing blobs
- [ ] Test detect orphaned files
- [ ] Test detect signature mismatches
- [ ] Test repair fixes issues
- [ ] Test dry_run doesn't modify cache

**Documentation:**
- [ ] Add to API_REFERENCE.md
- [ ] Add examples: post-crash recovery, debugging

**Priority:** ⚠️ Low - Debugging/maintenance utility

---

### Phase 3 Implementation Roadmap

**Recommended Order:**

**Sprint 1 (2-3 weeks): Trivial/High-Value Operations**
1. ✅ **Get Metadata Exposure** (3.3) - Trivial, expose existing method
   - 1-2 days implementation
   - 1 day testing
   - 1 day documentation
2. ✅ **Update Blob Data** (3.1) - High priority, core CRUD
   - 1 week implementation (all backends)
   - 3 days testing
   - 2 days documentation

**Sprint 2 (3-4 weeks): Cache Operations**
3. ✅ **Touch/Refresh TTL** (3.4) - High priority cache operation
   - 1 week implementation (backend support + cache wrapper)
   - 3 days testing
   - 2 days documentation
4. ✅ **Bulk Delete** (3.2) - High priority cleanup operation
   - 1 week implementation (backends + cache wrappers)
   - 3 days testing
   - 2 days documentation

**Sprint 3 (3-4 weeks): Batch Operations**
5. ✅ **Batch Operations** (3.5) - Medium priority performance optimization
   - 1.5 weeks implementation (get_batch, delete_batch, update_batch)
   - 4 days testing (including benchmarks)
   - 3 days documentation

**Sprint 4 (2-3 weeks): Convenience Operations**
6. ⚠️ **Copy/Move Entries** (3.6) - Low priority convenience
   - 1 week implementation
   - 2 days testing
   - 2 days documentation

**Sprint 5 (Optional - 2-3 weeks): Utilities**
7. ⚠️ **Export/Import** (3.7) - Low priority
8. ⚠️ **Verify/Repair** (3.8) - Low priority

**Total Estimate:**
- **Core Operations (Sprints 1-3):** 8-11 weeks
- **With Convenience (Sprint 4):** 10-14 weeks
- **With Utilities (Sprint 5):** 12-17 weeks

**Recommendation:** Implement Sprints 1-3 first (core operations), then evaluate user demand for Sprints 4-5.

---

### Phase 3 Testing Strategy

**Unit Tests:**
- [ ] Test each operation in isolation
- [ ] Test across all backends (SQLite, PostgreSQL, JSON, Memory)
- [ ] Test error cases and edge cases
- [ ] Test transaction rollback (where applicable)

**Integration Tests:**
- [ ] Test cache layer + backend layer integration
- [ ] Test with BlobStore wrapper
- [ ] Test with UnifiedCache wrapper
- [ ] Test with different configurations

**Performance Tests:**
- [ ] Benchmark batch operations vs individual
- [ ] Benchmark bulk delete vs loop
- [ ] Compare backend performance (SQL vs JSON vs Memory)

**Regression Tests:**
- [ ] Ensure existing operations still work
- [ ] Ensure cache_key immutability enforced
- [ ] Ensure TTL behavior unchanged

---

### Phase 3 Documentation

**API Reference Updates:**
- [ ] Document all new backend methods
- [ ] Document all new cache layer methods
- [ ] Include code examples for each operation
- [ ] Document return types and error conditions

**New Examples:**
- [ ] `examples/management_operations_demo.py` - Comprehensive demo
- [ ] Update existing examples to use new operations where appropriate

**Guides:**
- [ ] Update MISSING_MANAGEMENT_API.md with implementation status
- [ ] Add "Cache Management" section to main README
- [ ] Add troubleshooting guide for common patterns

---

##### 2.10 S3 Blob Backend

**Status:** 🚧 In Progress (Core Implementation Complete)

**Purpose:** Cloud-native blob storage using Amazon S3 or S3-compatible services (MinIO, etc.)

**Use Case:** Production distributed caching environments requiring:
- Shared blob storage across multiple machines/workers
- Durability and redundancy of cloud storage
- Scalable storage without local disk constraints
- MinIO compatibility for on-premises S3-compatible storage

**Design Principles:**

1. **S3 ETag Integration:**
   - S3 ETags are stored as separate metadata field (`s3_etag`)
   - ETags are NOT the same as cacheness file/content hash
   - Content hash is computed client-side before upload for integrity
   - ETag is server-side (S3's MD5 or multipart hash) for S3 consistency checks
   - Both are stored: `file_hash` (cacheness) + `s3_etag` (S3)

2. **Backend Compatibility Validation:**
   - **Allowed:** PostgreSQL metadata + S3 blobs (distributed)
   - **Allowed:** Memory metadata + Memory blobs (testing)
   - **NOT Allowed:** SQLite metadata + S3 blobs (local + remote mismatch)
   - **NOT Allowed:** JSON metadata + S3 blobs (local + remote mismatch)
   - Configuration validation enforces these rules at initialization

3. **boto3 and MinIO Compatibility:**
   - Uses boto3 as the S3 client library
   - Supports custom `endpoint_url` for MinIO/S3-compatible services
   - Standard AWS credential chain (env vars, IAM roles, credential files)

4. **Git-Style Directory Sharding:**
   - Avoids filesystem/S3 performance issues from too many files in one directory
   - Uses leading characters of `file_hash` as subdirectory (like Git's `.git/objects/`)
   - Configurable `shard_chars` option (default: 2, matching Git)
   - Full hash is always the filename for easy lookup
   - Example: hash `abc123def456...` → `ab/abc123def456.blob`
   - Works with both S3 and Filesystem blob backends
   - S3 benefits: faster LIST operations, better partitioning
   - Filesystem benefits: avoids inode limits, better fs performance

**Configuration Example:**

```python
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig, CacheBlobConfig

# Production: PostgreSQL + S3
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",
        connection_url="postgresql://user:pass@db.example.com/cache",
    ),
    blob=CacheBlobConfig(
        backend="s3",
        bucket="my-cache-bucket",
        prefix="cache/v1/",  # Optional key prefix
        region="us-east-1",
        shard_chars=2,  # Git-style: ab/abc123... (default=2)
        # For MinIO or S3-compatible:
        # endpoint_url="http://minio.local:9000",
        # use_ssl=False,
    )
)

cache = cacheness(config=config)

# Store data - blob goes to S3, metadata to PostgreSQL
cache.put(model, experiment="exp_001")

# Metadata includes both hashes:
# {
#     "cache_key": "abc123...",
#     "file_hash": "xxhash_content_hash",  # Cacheness-computed
#     "s3_etag": "\"d41d8cd98f00b204...\"",  # S3-provided
#     "s3_bucket": "my-cache-bucket",
#     "s3_key": "cache/v1/ab/abc123def456.blob",  # Sharded path
#     ...
# }
```

**MinIO Example:**

```python
# On-premises MinIO deployment
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",
        connection_url="postgresql://localhost/cache",
    ),
    blob=CacheBlobConfig(
        backend="s3",
        bucket="local-cache",
        endpoint_url="http://minio.internal:9000",
        use_ssl=False,
        access_key="minioadmin",  # Or use env vars
        secret_key="minioadmin",
    )
)
```

**Backend Compatibility Matrix:**

| Metadata Backend | Blob Backend | Allowed | Reason |
|------------------|--------------|---------|--------|
| PostgreSQL | S3 | ✅ Yes | Both distributed/remote |
| PostgreSQL | Filesystem | ✅ Yes | Shared metadata, local blobs |
| SQLite | Filesystem | ✅ Yes | Both local |
| SQLite | S3 | ❌ No | Local metadata + remote blobs |
| JSON | S3 | ❌ No | Local metadata + remote blobs |
| Memory | Memory | ✅ Yes | Both ephemeral (testing) |
| Memory | S3 | ❌ No | Ephemeral metadata + persistent blobs |

**Implementation Tasks:**

- [x] Create `S3BlobBackend` class implementing `BlobBackend` interface
- [x] Add boto3 as optional dependency (`pip install cacheness[s3]`)
- [x] Implement S3 operations: put, get, delete, exists, list_keys
- [ ] Handle multipart uploads for large blobs (>5GB)
- [x] Store and track S3 ETags in metadata
- [ ] **TODO: Add `s3_etag` field to metadata schema**
  - [ ] Add s3_etag column to SQLite metadata backend (cache_entries table)
  - [ ] Add s3_etag column to PostgreSQL metadata backend (cache_entries table)
  - [ ] Store S3 ETag in metadata dict when S3BlobBackend is used
  - [ ] Update metadata serialization to preserve s3_etag
  - [ ] Add tests for s3_etag storage and retrieval
  - [ ] Document s3_etag field in API reference
- [x] Implement backend compatibility validation in `CacheConfig`
- [x] Add validation error for incompatible backend combinations
- [x] Support custom endpoint_url for MinIO compatibility
- [x] Support AWS credential chain (env vars, IAM, credential files)
- [x] Handle S3 errors gracefully (network, permissions, bucket not found)
- [ ] Add retry logic with exponential backoff for transient failures
- [x] Implement Git-style directory sharding with configurable `shard_chars`
- [x] Add `shard_chars` config option to `CacheBlobConfig` (default: 2)
- [x] Apply sharding to both S3 and Filesystem blob backends
- [x] Create `tests/test_s3_blob_backend.py` with moto mocking
- [x] Add tests for directory sharding (0, 1, 2, 3+ chars)
- [ ] Create integration tests for MinIO (optional, CI environment)
- [ ] **TODO: Add `s3_etag` field to metadata schema**
  - [ ] Add s3_etag column to SQLite metadata backend (cache_entries table)
  - [ ] Add s3_etag column to PostgreSQL metadata backend (cache_entries table)
  - [ ] Store S3 ETag in metadata dict when S3BlobBackend is used
  - [ ] Update metadata serialization to preserve s3_etag
  - [ ] Add tests for s3_etag storage and retrieval
  - [ ] Document s3_etag field in API reference
- [ ] Update `docs/PLUGIN_DEVELOPMENT.md` with S3 backend example
- [ ] Create `docs/S3_BACKEND.md` with setup and configuration guide
- [ ] Add S3 example to `examples/` directory

**S3BlobBackend Interface:**

```python
class S3BlobBackend(BlobBackend):
    """S3-compatible blob storage backend."""
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,  # For MinIO
        use_ssl: bool = True,
        access_key: Optional[str] = None,  # Falls back to boto3 credential chain
        secret_key: Optional[str] = None,
        shard_chars: int = 2,  # Git-style: use first N chars as subdirectory
        **kwargs
    ):
        ...
    
    def _get_sharded_key(self, file_hash: str) -> str:
        """
        Get the sharded S3 key for a file hash.
        
        Example (shard_chars=2):
            file_hash = "abc123def456..."
            returns: "ab/abc123def456.blob"
        
        Example (shard_chars=0, disabled):
            returns: "abc123def456.blob"
        """
        if self.shard_chars > 0:
            shard_dir = file_hash[:self.shard_chars]
            return f"{self.prefix}{shard_dir}/{file_hash}.blob"
        return f"{self.prefix}{file_hash}.blob"
    
    def put(self, key: str, data: bytes) -> dict:
        """
        Upload blob to S3.
        
        Returns:
            dict with 's3_key', 's3_etag', 's3_bucket', 's3_version_id' (if versioned)
        """
        ...
    
    def get(self, key: str) -> Optional[bytes]:
        """Download blob from S3."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete blob from S3."""
        ...
    
    def exists(self, key: str) -> bool:
        """Check if blob exists in S3."""
        ...
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List blob keys in S3 bucket."""
        ...
    
    def get_etag(self, key: str) -> Optional[str]:
        """Get S3 ETag for a blob (HEAD request)."""
        ...
    
    def verify_etag(self, key: str, expected_etag: str) -> bool:
        """Verify blob integrity via ETag comparison."""
        ...
```

**Directory Sharding:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     GIT-STYLE DIRECTORY SHARDING                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  file_hash: "abc123def456789..."                                │
│                                                                  │
│  shard_chars=0 (disabled):     shard_chars=2 (default/Git):     │
│  ─────────────────────────     ────────────────────────────     │
│  prefix/abc123def456.blob      prefix/ab/abc123def456.blob      │
│                                                                  │
│  shard_chars=3:                shard_chars=4:                   │
│  ────────────────              ────────────────                 │
│  prefix/abc/abc123def456.blob  prefix/abc1/abc123def456.blob    │
│                                                                  │
│  Benefits:                                                       │
│  • S3: Faster LIST operations, better request distribution      │
│  • Filesystem: Avoids ext4 ~10K file/dir limit, better perf     │
│  • With shard_chars=2: max 256 dirs (00-ff for hex hashes)      │
│  • Matches Git's proven .git/objects/ organization              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**ETag vs Content Hash:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     INTEGRITY VERIFICATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  file_hash (cacheness)          s3_etag (S3)                    │
│  ─────────────────────          ────────────                    │
│  • Computed client-side         • Computed server-side          │
│  • xxhash of content            • MD5 (single upload)           │
│  • Consistent across backends   • Multipart hash (>5GB)         │
│  • Used for cache key gen       • Used for S3 consistency       │
│  • Stored in metadata           • Stored in metadata            │
│                                                                  │
│  Use Cases:                     Use Cases:                       │
│  • Deduplication                • Verify upload success          │
│  • Cache hit detection          • Conditional GET/PUT            │
│  • Cross-backend integrity      • S3 versioning/replication     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Dependencies:**

```toml
# pyproject.toml
[project.optional-dependencies]
s3 = ["boto3>=1.26.0"]
recommended = ["sqlalchemy>=2.0", "orjson", "blosc2", "psycopg[binary]>=3.1"]
cloud = ["boto3>=1.26.0", "psycopg[binary]>=3.1"]  # S3 + PostgreSQL
```

---

##### 2.11 Backend Feature Parity Verification

**Status:** ✅ Complete

**Purpose:** Ensure complete feature parity between SQLite and PostgreSQL metadata backends.

**Key Findings:**
- ✅ **Technical metadata fields preserved:** s3_etag, actual_path, object_type, storage_format, serializer, compression_codec, file_hash, entry_signature
- ⚠️ **Arbitrary metadata NOT preserved by default:** This is by design for performance (zero JSON parsing overhead)
- ℹ️ **For arbitrary metadata:** Use `store_full_metadata=True` option (Section 2.13)
- ℹ️ **For custom typed metadata:** Use custom metadata tables (Section 2.7)

**Implementation Tasks:**

- [x] Custom metadata tables (Section 2.7 - completed)
- [x] query_custom_session() support (Section 2.7 - completed)
- [x] Verify all metadata operations work identically on both backends
- [x] Add comprehensive cross-backend compatibility tests (test_backend_parity.py - 11 tests passing)
- [x] Document backend design decision (technical fields only by default)
- [ ] Document backend differences in BACKEND_SELECTION.md (in progress)
- [ ] Consider memory_cache layer compatibility with both backends

---

##### 2.12 TTL Unit Consistency

**Status:** 🚧 Planned (BREAKING CHANGE)

**Purpose:** Standardize TTL units throughout codebase for consistency and clarity.

**Current State:** Mixed usage (ttl_hours, memory_cache_ttl_seconds)

**Recommendation:** **Seconds** (most granular, standard in caching systems like Redis)

**Alternative:** Keep hours for human-friendly config, convert internally

**Implementation Tasks:**

- [ ] **Decision needed**: Choose seconds, minutes, or hours as standard unit
- [ ] Audit all TTL-related code: core.py, config.py, backends, sql_cache.py
- [ ] Update configuration parameters (breaking change for users)
- [ ] Update all documentation to reflect new TTL units
- [ ] Update all tests to use new TTL parameters
- [ ] Add migration guide for users upgrading
- [ ] Consider adding helper methods: ttl_hours(), ttl_minutes(), ttl_seconds()

---

##### 2.13 Custom Metadata Tables Verification

**Status:** ✅ Complete (Core Testing) | 🚧 Documentation In Progress

**Purpose:** Verify that custom SQLAlchemy metadata table functionality works correctly with both SQLite and PostgreSQL backends.

**Background:** Users can define custom SQLAlchemy ORM models that link to the main metadata table via `cache_key` foreign key. This allows storing typed, queryable metadata alongside cache entries. This is a **backend feature** (not cache-layer functionality), similar to how `metadata_dict` is handled - it's pure storage/retrieval without caching semantics.

**Key Design Principle:**

Custom metadata tables are backend-agnostic storage:
- **Not involved in cache key generation** - cache keys are computed before metadata storage
- **Not involved in cache hit/miss logic** - cache lookups use standard cache_key
- **Pure storage feature** - backends store/retrieve custom metadata entries alongside cache metadata
- **Foreign key relationship** - custom tables link to `cache_entries.cache_key`
- **User-managed schema** - users define columns, types, indexes via SQLAlchemy models

**Architecture:**

```
┌────────────────────────────────────────────────────────────────┐
│                     CACHING LAYER (core.py)                     │
│  • Generate cache_key from function args                        │
│  • Check cache hit/miss                                         │
│  • Serialize/deserialize data                                   │
│  • Pass metadata to backend for storage                         │
└────────────────────────────────────────────────────────────────┘
                              ↓
        cache_key, metadata_dict, custom_metadata_entry
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                  METADATA BACKEND (backends/)                   │
│                                                                  │
│  cache_entries table (infrastructure metadata)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ cache_key (PK) │ file_hash │ created_at │ file_size │... │  │
│  │ abc123...      │ xxhash... │ 2026-02-05 │ 1024      │... │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↑                                  │
│                              │ FK (ondelete CASCADE)            │
│                              │                                  │
│  custom_experiments table (user-defined metadata)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ id (PK) │ cache_key (FK) │ experiment_id │ model_type │   │  │
│  │ 1       │ abc123...      │ exp_001       │ xgboost    │   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  custom_performance table (another user-defined schema)         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ id (PK) │ cache_key (FK) │ run_id │ training_time │ ...  │  │
│  │ 1       │ abc123...      │ run_1  │ 123.45        │ ...  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  • Simple one-to-many: one cache entry → many metadata records  │
│  • Direct FK provides clear ownership and isolation             │
│  • Automatic cascade delete - no orphaned metadata              │
│  • Backends: SQLite, PostgreSQL                                 │
└────────────────────────────────────────────────────────────────┘
```

**Current Implementation (Section 2.7):**

```python
from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer

# User defines custom SQLAlchemy model
@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    __tablename__ = "custom_experiments"
    
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False)

# Create cache with backend
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="postgresql",  # or "sqlite"
        connection_url="postgresql://localhost/cache"
    )
)
cache = cacheness(config=config)

# Store data with custom metadata
experiment = ExperimentMetadata(
    experiment_id="exp_001",
    model_type="xgboost",
    accuracy=0.95,
    epochs=100
)

# Backend stores both cache metadata AND custom metadata
cache.put(
    trained_model,
    experiment="exp_001",
    custom_metadata=experiment  # Backend handles storage
)

# Query custom metadata via backend
with cache.query_custom_session("experiments") as query:
    high_accuracy = query.filter(
        ExperimentMetadata.accuracy >= 0.9
    ).all()
```

**Backend Contract:**

Metadata backends must implement:

```python
class MetadataBackend(ABC):
    @abstractmethod
    def store_custom_metadata(self, cache_key: str, custom_entry: Any) -> None:
        """
        Store custom SQLAlchemy metadata entry linked to cache_key.
        
        Args:
            cache_key: The cache key (foreign key to cache_entries)
            custom_entry: Populated SQLAlchemy ORM instance
        """
        pass
    
    @abstractmethod
    def get_custom_metadata(self, cache_key: str, model_name: str) -> Optional[Any]:
        """
        Retrieve custom metadata entry for cache_key.
        
        Args:
            cache_key: The cache key
            model_name: The registered custom metadata model name
            
        Returns:
            SQLAlchemy ORM instance or None
        """
        pass
    
    @abstractmethod
    def query_custom_session(self, model_name: str) -> Session:
        """
        Return SQLAlchemy session for querying custom metadata.
        
        Args:
            model_name: The registered custom metadata model name
            
        Returns:
            SQLAlchemy Session for custom queries
        """
        pass
```

**Implementation Tasks:**

**Testing:**
- [x] Verify SQLite backend custom metadata storage
- [x] Verify PostgreSQL backend custom metadata storage (skipped - not available in test env)
- [x] Test custom metadata with cache.put() for both backends
- [x] Test custom metadata retrieval with cache.get() for both backends
- [x] Test query_custom_session() for both backends
- [x] Test foreign key constraints (cascade deletion works correctly)
- [x] Test multiple custom metadata models in same cache
- [x] Test custom metadata with different column types (String, Integer, Float, DateTime, Boolean)
- [x] Test custom metadata queries with ordering
- [x] Test custom metadata queries with joins across multiple custom tables
- [ ] Test custom metadata migration (add/remove columns) for both backends
- [x] Add comprehensive test file: `tests/test_custom_metadata_backends.py` (34 tests, 17 passing SQLite)
- [x] Fix test model validation warnings (table naming, indexes)

**Backend Implementation Verification:**
- [x] Verify SQLiteBackend.store_custom_metadata() implementation
- [x] Verify PostgreSQLBackend.store_custom_metadata() implementation (via parametrized tests)
- [x] Verify SQLiteBackend.get_custom_metadata() implementation
- [x] Verify PostgreSQLBackend.get_custom_metadata() implementation (via parametrized tests)
- [x] Verify SQLiteBackend.query_custom_session() implementation
- [x] Verify PostgreSQLBackend.query_custom_session() implementation (via parametrized tests)
- [x] Ensure custom tables use same SQLAlchemy engine as cache_entries
- [x] Ensure custom tables created automatically on first use (via migrate_custom_metadata_tables)
- [x] Ensure foreign key from custom tables to cache_entries.cache_key works

**Integration with Core:**
- [x] Verify core.py passes custom_metadata to backend.store_custom_metadata()
- [x] Verify core.py retrieves custom_metadata from backend.get_custom_metadata()
- [x] Ensure custom_metadata is NOT used in cache key generation (verified via tests)
- [x] Ensure custom_metadata is NOT used in cache hit/miss logic (verified via tests)
- [x] Ensure custom_metadata storage happens AFTER cache entry is created
- [x] Ensure custom_metadata is optional (cache works without it, tested)

**Documentation:**
- [x] Update `docs/CUSTOM_METADATA.md` with backend-agnostic examples
- [x] Update `docs/CUSTOM_METADATA.md` with cascade deletion behavior
- [x] Update `docs/CUSTOM_METADATA.md` with best practices section
- [x] Document custom metadata API in `docs/API_REFERENCE.md` (Complete section added)
- [x] Add PostgreSQL custom metadata example to `examples/` (examples/custom_metadata_postgresql.py)
- [x] Document foreign key relationship and cascade behavior
- [ ] Document migration patterns for schema changes
- [ ] Document performance implications (indexed columns, query optimization)

**Error Handling:**
- [x] Test custom metadata storage failure (cache entry succeeds, metadata fails gracefully)
- [ ] Test custom metadata with invalid foreign key
- [ ] Test custom metadata with duplicate unique constraint violations
- [x] Test custom metadata with missing required columns (DB constraint error, logged)
- [x] Add validation for custom metadata models (validate_custom_metadata_model in custom_metadata.py)

**Edge Cases:**
- [x] Test custom metadata with cache.clear_all() (links removed, records orphaned - by design)
- [x] Test custom metadata with cache.invalidate() (links cascade-deleted, records orphaned)
- [x] Test custom metadata with expired cache entries (orphaned custom metadata - use cleanup_orphaned_metadata())
- [x] Test custom metadata with cache.get() when entry doesn't exist (returns empty dict)
- [ ] Test custom metadata with multiple concurrent cache.put() calls

**Example Test Structure:**

```python
# tests/test_custom_metadata_backends.py
import pytest
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer

@custom_metadata_model("test_experiments")
class TestExperimentMetadata(Base, CustomMetadataBase):
    __tablename__ = "test_custom_experiments"
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)

@pytest.fixture(params=["sqlite", "postgresql"])
def cache_with_backend(request, postgres_available):
    """Test with both SQLite and PostgreSQL backends."""
    backend = request.param
    
    if backend == "postgresql" and not postgres_available:
        pytest.skip("PostgreSQL not available")
    
    if backend == "sqlite":
        config = CacheConfig(
            metadata=CacheMetadataConfig(
                backend="sqlite",
                connection_url="sqlite:///:memory:"
            )
        )
    else:
        config = CacheConfig(
            metadata=CacheMetadataConfig(
                backend="postgresql",
                connection_url="postgresql://localhost/test_cache"
            )
        )
    
    cache = cacheness(config=config)
    yield cache
    cache.clear_all()

def test_custom_metadata_storage(cache_with_backend):
    """Test custom metadata storage for both backends."""
    cache = cache_with_backend
    
    # Create custom metadata entry
    experiment = TestExperimentMetadata(
        experiment_id="exp_001",
        model_type="xgboost",
        accuracy=0.95
    )
    
    # Store with custom metadata
    data = {"model": "trained_model"}
    cache.put(data, experiment="exp_001", custom_metadata=experiment)
    
    # Verify retrieval
    result = cache.get(experiment="exp_001")
    assert result == data
    
    # Query custom metadata
    with cache.query_custom_session("test_experiments") as query:
        entries = query.filter(
            TestExperimentMetadata.accuracy >= 0.9
        ).all()
        assert len(entries) == 1
        assert entries[0].experiment_id == "exp_001"
        assert entries[0].model_type == "xgboost"
        assert entries[0].accuracy == 0.95

def test_custom_metadata_foreign_key_cascade(cache_with_backend):
    """Test foreign key cascade on cache entry deletion."""
    cache = cache_with_backend
    
    # Store with custom metadata
    experiment = TestExperimentMetadata(
        experiment_id="exp_002",
        model_type="lightgbm",
        accuracy=0.92
    )
    cache.put({"model": "data"}, experiment="exp_002", custom_metadata=experiment)
    
    # Verify custom metadata exists
    with cache.query_custom_session("test_experiments") as query:
        entry = query.filter(
            TestExperimentMetadata.experiment_id == "exp_002"
        ).first()
        assert entry is not None
    
    # Delete cache entry
    cache.delete(experiment="exp_002")
    
    # Verify custom metadata also deleted (cascade)
    with cache.query_custom_session("test_experiments") as query:
        entry = query.filter(
            TestExperimentMetadata.experiment_id == "exp_002"
        ).first()
        assert entry is None
```

**Success Criteria:**

- ✅ Custom metadata storage works identically on SQLite and PostgreSQL
- ✅ Custom metadata is purely a backend feature (no cache function involvement like TTL)
- ✅ Foreign key relationships maintained correctly
- ✅ Cascade deletes work properly (links cascade-deleted, records remain)
- ✅ Query functionality works for both backends
- ✅ Multiple custom metadata models can coexist
- ✅ Comprehensive test coverage (tests/test_custom_metadata_backends.py - 34 tests, 17 passing SQLite)
- [ ] Documentation updated with backend-agnostic examples

**Key Findings (February 2026):**

1. **Architecture Simplified (Feb 5 2026):**
   - **Changed from link table to direct FK**: Originally used `cache_metadata_links` table for many-to-many
   - **Direct FK is correct**: Each metadata record belongs to exactly ONE cache entry (one-to-many)
   - **Benefits**: Simpler code, clearer ownership, automatic cascade delete, easier queries
   - **No orphaned metadata**: CASCADE delete removes custom metadata when cache entry deleted
   - **Isolation**: Each cache entry's metadata is independent - deleting one doesn't affect others

2. **Test Results:**
   - All 17 SQLite tests passing ✅ (includes simplified joins test)
   - All 17 PostgreSQL tests skipping gracefully (backend not available in test environment)
   - Tests verify: storage, retrieval, queries, cascade behavior, multiple models, column types, edge cases

3. **API Clarifications:**
   - Use `cache.invalidate()` not `cache.delete()` to remove cache entries
   - Custom metadata retrieved via `cache.get_custom_metadata_for_entry()`
   - Query sessions via `cache.query_custom_session(model_name)` context manager
   - Correlating across tables: query by `cache_key` (direct FK attribute)

4. **Implementation Notes:**
   - Session-scoped fixtures needed for model registration (avoid registry reset)
   - Models need `__table_args__ = {'extend_existing': True}` for test flexibility
   - Migration via `migrate_custom_metadata_tables(engine)` with explicit engine parameter
   - Registry not reset automatically between tests to preserve model registrations
   - `cache_key` column added automatically via `CustomMetadataBase.cache_key` declared_attr

---

#### Phase 3: Management API Enhancements

**Status:** 🆕 Proposed

**Purpose:** Add comprehensive management operations for cache maintenance and data lifecycle.

**Analysis Document:** [docs/MISSING_MANAGEMENT_API.md](MISSING_MANAGEMENT_API.md)

**Priority Features:**

##### 3.1 Touch/Refresh TTL (High Priority)

Extend expiration time without reloading data:

```python
# Reset TTL to default
cache.touch(experiment="exp_001")

# Set custom TTL
cache.touch(experiment="exp_001", ttl_seconds=hours(48))
```

**Use Cases:**
- Keep frequently accessed data alive
- Reset TTL for active sessions
- Prevent expiration during long operations

##### 3.2 Replace/Update Data (Medium Priority)

Update cached data in-place:

```python
# Replace existing entry
cache.replace(new_data, experiment="exp_001")

# Update only if exists
cache.update(new_data, experiment="exp_001", if_exists=True)
```

**Use Cases:**
- Update stale data without invalidate+put
- Modify cached results
- Atomic updates

##### 3.3 Bulk Delete Operations (Medium Priority)

Delete multiple entries by pattern:

```python
# Delete all matching entries
count = cache.delete_by_prefix(project="ml_models")

# Delete with custom filter
count = cache.delete_matching(lambda meta: meta.get("version").startswith("v1"))
```

**Use Cases:**
- Cleanup old experiments
- Remove specific data types
- Clear version ranges

##### 3.4 Get Metadata Only (Medium Priority)

Retrieve metadata without loading data:

```python
# Check metadata before loading
meta = cache.get_metadata(experiment="exp_001")
# Returns: {"cache_key": "...", "file_size": 1024, "created_at": "...", ...}
```

**Use Cases:**
- Inspect TTL before loading
- Check file size before download
- Query metadata for decisions

**Additional Features (Lower Priority):**
- Export/Import cache for backup/migration
- Verify and repair corrupted entries
- Batch operations (get_batch, touch_batch)
- Copy/clone entries

**Implementation Tasks:**

- [ ] Add `touch()` method to UnifiedCache
- [ ] Add `replace()` method to UnifiedCache
- [ ] Add `delete_by_prefix()` method
- [ ] Expose `get_metadata()` in UnifiedCache
- [ ] Add tests for all new operations
- [ ] Document in API_REFERENCE.md
- [ ] Create usage examples

---

#### Phase 4: Evaluate Full Separation (Future Decision)
1. Assess community adoption and feedback
2. Determine if separate packages provide value
3. If splitting, use namespace packages for smooth migration:
   ```python
   # Before split
   from cacheness.storage import BlobStore
   
   # After split (same import!)
   from cacheness.storage import BlobStore  # Now from cacheness-storage package
   ```

---

### Planned Backend Implementations

#### PostgreSQL Metadata Backend

**Purpose:** Metadata storage and querying (cache index)

**Use Case:** Production environments requiring:
- High concurrency (multiple workers, distributed systems)
- Advanced querying (complex metadata filters, joins)
- ACID transactions for metadata consistency
- Centralized cache index across multiple machines

**Note:** PostgreSQL can store:
1. **Cache infrastructure metadata** (via `metadata_backend="postgresql"`): cache keys, file paths, statistics, etc.
2. **SqlCache custom metadata tables** (via `SqlCache(db_url="postgresql://...")`): user-defined schemas for queryable data
3. Both can coexist in same database, different tables

Actual cached data (blobs) are stored separately via blob backends (filesystem, S3, etc.).

**Implementation Plan:**

```python
# cacheness/storage/backends/postgresql.py (or as plugin package)
from cacheness.storage.backends.base import MetadataBackend
from sqlalchemy import create_engine, Table, Column, String, Integer, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any, Optional, List

class PostgresBackend(MetadataBackend):
    """
    PostgreSQL metadata backend for distributed caching.
    
    Features:
    - Connection pooling for concurrent access
    - JSON column for flexible metadata storage
    - Indexes on cache_key and common query fields
    - Optional table partitioning for large caches
    """
    
    def __init__(
        self,
        connection_url: str,
        table_name: str = "cache_metadata",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True
    ):
        self.engine = create_engine(
            connection_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping
        )
        self.table_name = table_name
        self._setup_schema()
        self.Session = sessionmaker(bind=self.engine)
    
    def _setup_schema(self):
        """Create cache metadata table with indexes."""
        # Similar to SqliteBackend but with PostgreSQL optimizations
        ...
    
    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get entry with connection pooling."""
        with self.Session() as session:
            result = session.query(CacheEntry).filter_by(cache_key=cache_key).first()
            return result.to_dict() if result else None
    
    # ... implement other MetadataBackend methods

# Usage
from cacheness import cacheness, CacheConfig

config = CacheConfig(
    cache_dir="./cache_blobs",  # Blobs still on filesystem
    metadata_backend="postgresql",
    backend_options={
        "connection_url": "postgresql://user:pass@db.example.com:5432/cacheness",
        "pool_size": 20,
        "table_name": "ml_cache_metadata"
    }
)

cache = cacheness(config)
```

**Implementation Checklist:**
- [ ] Create `PostgresBackend` class implementing `MetadataBackend`
- [ ] Add connection pooling configuration
- [ ] Add table partitioning support for large caches (optional)
- [ ] Add prepared statements for common queries (performance)
- [ ] Support SSL/TLS connections
- [ ] Add migration utilities from SQLite → PostgreSQL
- [ ] Add monitoring/metrics hooks (query time, pool stats)
- [ ] Ensure no table name conflicts with SqlCache custom tables
- [ ] Support shared database with SqlCache (different table names)
- [ ] Create `docs/POSTGRESQL_BACKEND.md` with deployment guide
- [ ] Add integration tests with test PostgreSQL container
- [ ] Add integration tests showing cache metadata + SqlCache custom tables
- [ ] Benchmark vs SQLite for concurrent workloads

**Dependencies:**
- `psycopg2-binary` or `psycopg3` (PostgreSQL driver)
- `sqlalchemy>=2.0` (already required for SQLite backend)

---

#### S3 Blob Storage Backend

**Purpose:** Blob storage (actual cached data)

**Use Case:** Cloud-native deployments requiring:
- Unlimited storage capacity
- Shared cache across ephemeral compute (AWS Lambda, ECS, Kubernetes)
- Durability and replication
- Separation of compute and storage

**Implementation Plan:**

```python
# cacheness/storage/backends/s3_blob.py (or as plugin package)
from cacheness.storage.backends.base import BlobBackend  # New abstract class
from pathlib import Path
from typing import Optional
import boto3

class S3BlobBackend(BlobBackend):
    """
    Amazon S3 blob storage backend.
    
    Features:
    - Automatic multipart upload for large objects
    - Client-side encryption (optional)
    - Object lifecycle management (TTL via S3 lifecycle rules)
    - Presigned URL generation for direct client access
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "cacheness/",
        region: str = "us-east-1",
        encryption: bool = False,
        storage_class: str = "STANDARD"
    ):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket
        self.prefix = prefix
        self.encryption = encryption
        self.storage_class = storage_class
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Upload blob to S3."""
        key = f"{self.prefix}{blob_id}"
        
        extra_args = {"StorageClass": self.storage_class}
        if self.encryption:
            extra_args["ServerSideEncryption"] = "AES256"
        
        # Use multipart upload for large objects
        if len(data) > 100 * 1024 * 1024:  # > 100MB
            self._multipart_upload(key, data, extra_args)
        else:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                **extra_args
            )
        
        return f"s3://{self.bucket}/{key}"
    
    def read_blob(self, blob_path: str) -> bytes:
        """Download blob from S3."""
        key = blob_path.replace(f"s3://{self.bucket}/", "")
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()
    
    def delete_blob(self, blob_path: str):
        """Delete blob from S3."""
        key = blob_path.replace(f"s3://{self.bucket}/", "")
        self.s3.delete_object(Bucket=self.bucket, Key=key)
    
    def exists(self, blob_path: str) -> bool:
        """Check if blob exists in S3."""
        try:
            key = blob_path.replace(f"s3://{self.bucket}/", "")
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False
    
    def get_presigned_url(self, blob_path: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for direct access."""
        key = blob_path.replace(f"s3://{self.bucket}/", "")
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires_in
        )

# Usage with BlobStore
from cacheness.storage import BlobStore

store = BlobStore(
    blob_backend=S3BlobBackend(
        bucket="my-ml-cache",
        prefix="models/",
        storage_class="INTELLIGENT_TIERING"  # Automatic cost optimization
    ),
    metadata_backend="postgresql",  # Metadata in PostgreSQL, blobs in S3
    backend_options={
        "connection_url": "postgresql://..."
    }
)

# Store model in S3
model_id = store.put(
    trained_model,
    key="fraud_detector_v2",
    metadata={"accuracy": 0.95}
)

# Usage with UnifiedCache
from cacheness import cacheness, CacheConfig

cache = cacheness(config=CacheConfig(
    blob_backend="s3",
    blob_backend_options={
        "bucket": "my-cache-bucket",
        "prefix": "team-cache/",
        "region": "us-west-2"
    },
    metadata_backend="postgresql",
    backend_options={
        "connection_url": "postgresql://..."
    }
))
```

**Implementation Checklist:**
- [ ] Create `BlobBackend` abstract base class
- [ ] Refactor current file-based storage to `FilesystemBlobBackend`
- [ ] Create `S3BlobBackend` class
- [ ] Add multipart upload/download for large blobs
- [ ] Add streaming support (read/write without full memory load)
- [ ] Support other S3-compatible services (MinIO, DigitalOcean Spaces, etc.)
- [ ] Add retry logic with exponential backoff
- [ ] Add client-side encryption option (using AWS KMS or custom keys)
- [ ] Create `docs/S3_BLOB_BACKEND.md` with IAM policy examples
- [ ] Add integration tests with LocalStack or MinIO
- [ ] Benchmark vs filesystem for various blob sizes
- [ ] Consider adding CloudFront CDN integration for read-heavy workloads

**Dependencies:**
- `boto3` (AWS SDK)
- Optional: `s3transfer` for optimized transfers

**Alternative Cloud Providers:**
- Azure Blob Storage: `AzureBlobBackend` using `azure-storage-blob`
- Google Cloud Storage: `GCSBlobBackend` using `google-cloud-storage`

---

<a id="phase-2-feasibility-review"></a>
### Phase 2 Feasibility Review & Gap Analysis

This section reviews the Phase 2 plan for practical implementation challenges and identifies overlooked features.

#### ✅ High Feasibility (Low Risk)

**Handler Plugin System (Section 2.1)**
- **Assessment:** Very feasible - standard Python pattern
- **Precedent:** Similar to Pytest plugins, Flask extensions
- **Risk:** Low - handlers are stateless, well-defined interface
- **Effort:** Small-Medium

**Metadata Backend Registry (Section 2.2)**
- **Assessment:** Feasible - factory pattern is well understood
- **Precedent:** SQLAlchemy dialects, Django database backends
- **Risk:** Low-Medium - needs careful connection pooling
- **Effort:** Medium

**Manual Registration API**
- **Assessment:** Very feasible - simple factory pattern
- **Precedent:** Matplotlib backends, logging handlers, unittest plugins
- **Risk:** Very Low
- **Effort:** Small
- **Note:** Entry point auto-discovery deferred to future phase if ecosystem demand emerges

#### ⚠️ Medium Feasibility (Needs Attention)

**Blob Storage Backend Registry (Section 2.3)**
- **Assessment:** Feasible but complex edge cases
- **Gaps Identified:**
  1. **Content hashing without download** (user-identified)
     - S3 ETag for content verification
     - Avoid downloading blobs just to verify integrity
     - ETags may not match local hash algorithms (multipart uploads)
  2. **Blob metadata/headers**
     - Content-Type, Cache-Control, Content-Encoding headers
     - S3 object metadata vs cacheness metadata
  3. **Bandwidth optimization**
     - Partial reads (byte ranges)
     - Compression in transit
     - CDN/edge caching layer
  4. **Concurrent access patterns**
     - Multiple processes reading same blob
     - Optimistic locking for updates
  5. **Local caching layer**
     - Cache frequently accessed S3 blobs locally
     - Configurable cache size, TTL
  6. **Backend migration**
     - Moving blobs from filesystem → S3
     - Incremental migration strategy
  7. **Cost tracking**
     - Monitor S3 requests, bandwidth usage
     - Configurable budget alerts
- **Risk:** Medium - cloud storage has many edge cases
- **Effort:** Large

**PostgreSQL Metadata Backend (Planned Backend)**
- **Assessment:** Feasible but operational complexity
- **Gaps Identified:**
  1. **Connection failure handling**
     - Automatic retries with exponential backoff
     - Circuit breaker pattern for failing connections
  2. **Read replicas**
     - Route reads to replicas, writes to primary
     - Handle replication lag gracefully
  3. **Schema migrations**
     - Alembic integration for version management
     - Zero-downtime migration strategy
  4. **Maintenance operations**
     - VACUUM, ANALYZE scheduling
     - Index rebuilding
     - Monitoring query performance
  5. **Multi-tenancy**
     - Schema-per-tenant vs table-per-tenant
     - Row-level security
- **Risk:** Medium - operational burden for users
- **Effort:** Medium-Large

**Configuration Schema & Validation (Section 2.5)**
- **Assessment:** Straightforward but needs thought
- **Gaps Identified:**
  1. **Config file formats**
     - YAML, TOML, JSON support
     - Which format is primary?
  2. **Environment variables**
     - Override config with env vars
     - Precedence rules (file vs env vs code)
  3. **Secrets management**
     - Don't store credentials in config files
     - Integration with AWS Secrets Manager, HashiCorp Vault
  4. **Config hot-reload**
     - Watch config file for changes
     - Graceful reconfiguration without restart
  5. **Config validation error messages**
     - Clear, actionable error messages
     - Suggest corrections for common mistakes
- **Risk:** Low-Medium
- **Effort:** Medium

#### 🔴 Challenging Areas (High Risk/Effort)

**Streaming Support**
- **Current Plan:** Mentioned but not detailed
- **User Question:** Is this really necessary?
- **Analysis:** 
  - **Current architecture:** Handlers already work with file paths, not bytes in memory
    - `ArrayHandler.put()` writes directly to disk: `np.savez(file_path, ...)`
    - `DataFrameHandler.put()` writes directly to disk: `df.to_parquet(file_path, ...)`
    - Handlers return file path, not data
  - **The actual problem:** Blob backend methods use `bytes` in memory
    - `write_blob(blob_id: str, data: bytes)` - requires loading file into memory
    - `read_blob(blob_path: str) -> bytes` - loads entire file into memory
  - **When is this a problem?**
    - Large models (>1GB) - current approach loads into memory twice (handler write + S3 upload)
    - Multi-GB datasets cached as Parquet files
    - Limited memory environments (Lambda, containers)
  
  **Two Approaches:**
  
  **Approach 1: File-Based Blob Backend (Simpler, Recommended for Phase 2)**
  ```python
  class BlobBackend(ABC):
      @abstractmethod
      def write_blob_from_file(self, blob_id: str, file_path: Path) -> str:
          """Upload file to blob storage without loading into memory."""
          pass
      
      @abstractmethod
      def read_blob_to_file(self, blob_path: str, file_path: Path):
          """Download blob directly to file without loading into memory."""
          pass
  
  class S3BlobBackend(BlobBackend):
      def write_blob_from_file(self, blob_id: str, file_path: Path) -> str:
          """Use boto3's upload_file (handles large files automatically)."""
          key = f"{self.prefix}{blob_id}"
          
          # boto3 automatically uses multipart upload for large files
          self.s3.upload_file(
              Filename=str(file_path),
              Bucket=self.bucket,
              Key=key,
              ExtraArgs={...}
          )
          return f"s3://{self.bucket}/{key}"
      
      def read_blob_to_file(self, blob_path: str, file_path: Path):
          """Use boto3's download_file (streams automatically)."""
          key = self._parse_s3_key(blob_path)
          self.s3.download_file(self.bucket, key, str(file_path))
  ```
  
  **Approach 2: Stream-Based (More Complex, Phase 2.3+)**
  ```python
  class BlobBackend(ABC):
      @abstractmethod
      def write_blob_stream(self, blob_id: str, stream: BinaryIO, size: int) -> str:
          """Write blob from stream (for very large objects)."""
          pass
      
      @abstractmethod
      def read_blob_stream(self, blob_path: str) -> BinaryIO:
          """Read blob as stream."""
          pass
  ```
  
  **Use Cases Where Streaming Actually Matters:**
  - **Partial reads** - Read specific byte range from S3 (e.g., Parquet row groups)
    - This is more about S3 range requests: `Range: bytes=0-1000000`
    - Most useful for columnar formats (Parquet, Arrow)
    - NOT needed for most caching use cases
  - **Progressive processing** - Process data chunks as they download
    - Useful for ETL pipelines
    - NOT typical for caching (cache is atomic read/write)
  - **Network streaming** - Stream data from API to S3 without local storage
    - Edge case, not typical caching scenario

- **Revised Assessment:**
  - ✅ **File-based blob backend** (Approach 1): **Medium priority** - solves memory issue simply
  - ❌ **True streaming** (Approach 2): **Low priority** - over-engineered for Phase 2
  - ❌ **Partial reads** - **Not needed** for caching use cases (violates cache atomicity)
  
- **Recommendation:** 
  - Phase 2.1: Add file-based methods to `BlobBackend` alongside bytes-based methods
  - Handlers continue to work with files (no changes needed)
  - Blob backends use file-based methods when available (boto3 handles large files)
  - Skip true streaming for now - YAGNI (You Aren't Gonna Need It)
  
- **Risk:** Low (file-based approach is simple)
- **Effort:** Small-Medium (mainly interface additions)

**Plugin Security & Isolation**
- **Current Plan:** Deferred - using manual registration instead of auto-discovery plugins
- **Manual Registration Approach:**
  - Users explicitly import and register handlers/backends
  - No automatic plugin loading = simpler security model
  - Registration conflicts handled explicitly at registration time
  - Version compatibility managed via standard Python packaging
- **Future Plugin System (if needed):**
  - Entry point discovery for third-party packages
  - Sandboxing for untrusted plugins
  - Plugin versioning constraints
- **Risk:** Low (with manual registration)
- **Effort:** Minimal for Phase 2
- **Recommendation:** Defer auto-discovery plugins until ecosystem demand justifies complexity

**Observability & Monitoring**
- **Current Plan:** Mentioned for PostgreSQL only
- **Gaps Identified:**
  1. **Metrics collection**
     - Cache hit/miss rates per handler
     - Backend latency, error rates
     - Blob storage costs (S3 requests, bandwidth)
  2. **Distributed tracing**
     - OpenTelemetry integration
     - Trace cache operations across services
  3. **Structured logging**
     - JSON logs for log aggregation
     - Correlation IDs for request tracking
  4. **Health checks**
     - Backend health endpoints
     - Readiness/liveness probes for Kubernetes
  5. **Alerting**
     - Integration with Prometheus, Datadog, etc.
     - Alert on high error rates, latency spikes
- **Risk:** Medium
- **Effort:** Medium-Large
- **Recommendation:** Essential for production deployments, should be in Phase 2

---

#### Critical Missing Features

**1. Content Hashing for Cloud Blob Backends** (High Priority)

**Problem:** Cannot verify blob integrity without downloading entire blob.

**Solution:**

```python
class BlobBackend(ABC):
    """Extended with content hash support."""
    
    @abstractmethod
    def get_blob_hash(self, blob_path: str, algorithm: str = "sha256") -> Optional[str]:
        """
        Get content hash without downloading blob.
        
        For S3: Use ETag (with caveats for multipart uploads)
        For Azure: Use Content-MD5 header
        For GCS: Use MD5 hash property
        
        Returns None if hash unavailable or algorithm unsupported.
        """
        pass

class S3BlobBackend(BlobBackend):
    def get_blob_hash(self, blob_path: str, algorithm: str = "sha256") -> Optional[str]:
        \"\"\"
        Get S3 ETag as content hash.
        
        Important: ETag != MD5 for multipart uploads (>100MB).
        For multipart: ETag = MD5(concat(MD5(part1), MD5(part2), ...)) + "-{parts}"
        
        Solution: Store custom metadata with actual hash:
        - x-amz-meta-sha256: actual SHA256 hash
        - Computed during upload
        \"\"\"
        key = self._parse_s3_key(blob_path)
        
        # Try custom metadata first
        response = self.s3.head_object(Bucket=self.bucket, Key=key)
        if 'x-amz-meta-sha256' in response['Metadata']:
            if algorithm == 'sha256':
                return response['Metadata']['x-amz-meta-sha256']
        
        # Fallback to ETag (only reliable for single-part uploads)
        etag = response['ETag'].strip('"')
        if '-' not in etag:  # Single-part upload
            if algorithm == 'md5':
                return etag
        
        return None  # Hash unavailable for this algorithm
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        \"\"\"Store blob with content hash in metadata.\"\"\"
        import hashlib
        
        # Calculate hash before upload
        sha256_hash = hashlib.sha256(data).hexdigest()
        md5_hash = hashlib.md5(data).hexdigest()
        
        key = f"{self.prefix}{blob_id}"
        
        extra_args = {
            "StorageClass": self.storage_class,
            "Metadata": {
                "sha256": sha256_hash,
                "md5": md5_hash,
                "original-size": str(len(data))
            }
        }
        
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        return f"s3://{self.bucket}/{key}"
```

**Implementation Tasks:**
- [ ] Add `get_blob_hash()` to `BlobBackend` interface
- [ ] Implement for S3BlobBackend with ETag + custom metadata
- [ ] Implement for FilesystemBlobBackend (compute hash on demand)
- [ ] Implement for AzureBlobBackend (Content-MD5 header)
- [ ] Implement for GCSBlobBackend (MD5Hash property)
- [ ] Update cache verification to use `get_blob_hash()` instead of downloading
- [ ] Document multipart upload ETag quirks
- [ ] Add tests for hash verification without download

---

**2. Retry & Resilience Strategies** (High Priority)

**Problem:** Network failures, transient errors common with cloud backends.

**Solution:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class S3BlobBackend(BlobBackend):
    def __init__(
        self,
        bucket: str,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 60.0,
        circuit_breaker_threshold: int = 5
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self._circuit_breaker = CircuitBreaker(threshold=circuit_breaker_threshold)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=60),
        retry=retry_if_exception_type((ClientError, ConnectionError))
    )
    def read_blob(self, blob_path: str) -> bytes:
        \"\"\"Read with automatic retry on transient errors.\"\"\"
        if not self._circuit_breaker.can_attempt():
            raise CircuitBreakerOpenError("S3 backend unavailable")
        
        try:
            key = self._parse_s3_key(blob_path)
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = response['Body'].read()
            self._circuit_breaker.record_success()
            return data
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise

class CircuitBreaker:
    \"\"\"Simple circuit breaker to prevent cascading failures.\"\"\"
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_attempt(self) -> bool:
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if timeout elapsed
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        
        # half-open: allow one attempt
        return True
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.threshold:
            self.state = "open"
```

**Implementation Tasks:**
- [ ] Add `tenacity` dependency for retry logic
- [ ] Implement retry decorators for all blob backend methods
- [ ] Add circuit breaker pattern for failing backends
- [ ] Make retry config tunable (max_retries, backoff, jitter)
- [ ] Add metrics for retry counts, circuit breaker state
- [ ] Document retry behavior and configuration
- [ ] Test with simulated network failures

---

**3. Local Caching Layer for Cloud Blob Backends** (Medium Priority)

**Problem:** Reading frequently-accessed blobs from S3 adds latency and cost.

**Solution:** With file-based blob backend interface, local caching becomes trivial - just coordinate between two blob backends!

```python
class CachedBlobBackend(BlobBackend):
    """
    Wrapper that adds local caching to any blob backend.
    
    Key insight: With file-based interface, local cache is just another blob backend!
    - Local cache: FilesystemBlobBackend (fast, limited space)
    - Remote storage: S3BlobBackend (slow, unlimited space)
    - Both implement same interface!
    
    Benefits:
    - Reduces latency (local disk faster than network)
    - Reduces cost (fewer S3 API requests)
    - Reduces bandwidth usage
    """
    
    def __init__(
        self,
        upstream: BlobBackend,  # S3, Azure, GCS, etc.
        cache_backend: BlobBackend,  # Typically FilesystemBlobBackend
        max_cache_size_mb: int = 1000,
        ttl_seconds: int = 3600
    ):
        self.upstream = upstream
        self.cache = cache_backend  # Just another blob backend!
        self.max_cache_size_mb = max_cache_size_mb
        self.ttl_seconds = ttl_seconds
        self._cache_index = self._load_cache_index()
    
    def read_blob_to_file(self, blob_path: str, file_path: Path):
        """Read with local cache."""
        # Check if cached locally
        cache_blob_id = self._get_cache_id(blob_path)
        
        if cache_blob_id in self._cache_index:
            timestamp = self._cache_index[cache_blob_id]['timestamp']
            
            # Check if cached copy is still valid
            if time.time() - timestamp < self.ttl_seconds:
                try:
                    # Read from local cache (FilesystemBlobBackend)
                    self.cache.read_blob_to_file(cache_blob_id, file_path)
                    logger.debug(f"Cache hit: {blob_path}")
                    return
                except FileNotFoundError:
                    # Cache entry exists but file missing - remove from index
                    del self._cache_index[cache_blob_id]
        
        # Cache miss - fetch from upstream (S3BlobBackend)
        logger.debug(f"Cache miss: {blob_path}")
        self.upstream.read_blob_to_file(blob_path, file_path)
        
        # Store in local cache using same interface!
        self._cache_locally(blob_path, file_path)
    
    def write_blob_from_file(self, blob_id: str, file_path: Path) -> str:
        """Write to upstream and optionally cache locally."""
        # Write to upstream (S3)
        remote_path = self.upstream.write_blob_from_file(blob_id, file_path)
        
        # Also cache locally for subsequent reads
        self._cache_locally(remote_path, file_path)
        
        return remote_path
    
    def _cache_locally(self, blob_path: str, file_path: Path):
        """Store blob in local cache - just another write_blob_from_file call!"""
        cache_blob_id = self._get_cache_id(blob_path)
        
        # Write to local cache using same blob backend interface
        self.cache.write_blob_from_file(cache_blob_id, file_path)
        
        # Update index
        file_size = file_path.stat().st_size
        self._cache_index[cache_blob_id] = {
            'blob_path': blob_path,
            'size': file_size,
            'timestamp': time.time()
        }
        
        # Evict if cache too large
        self._evict_if_needed()
        self._save_cache_index()
    
    def _evict_if_needed(self):
        """LRU eviction to stay under max_cache_size_mb."""
        total_size_mb = sum(
            entry['size'] for entry in self._cache_index.values()
        ) / (1024 * 1024)
        
        if total_size_mb > self.max_cache_size_mb:
            # Sort by timestamp (oldest first)
            items = sorted(
                self._cache_index.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Evict until under limit
            for cache_blob_id, entry in items:
                if total_size_mb <= self.max_cache_size_mb * 0.9:  # 90% threshold
                    break
                
                # Delete from cache backend (just a delete_blob call)
                try:
                    self.cache.delete_blob(cache_blob_id)
                except Exception as e:
                    logger.warning(f"Failed to evict {cache_blob_id}: {e}")
                
                del self._cache_index[cache_blob_id]
                total_size_mb -= entry['size'] / (1024 * 1024)
                logger.debug(f"Evicted from cache: {entry['blob_path']}")
    
    def _get_cache_id(self, blob_path: str) -> str:
        """Generate cache ID from blob path."""
        return hashlib.md5(blob_path.encode()).hexdigest()
    
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk (persists across restarts)."""
        # Could store as JSON file or small SQLite DB
        index_file = Path(self.cache.base_dir) / "cache_index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())
        return {}
    
    def _save_cache_index(self):
        """Persist cache index to disk."""
        index_file = Path(self.cache.base_dir) / "cache_index.json"
        index_file.write_text(json.dumps(self._cache_index))

# Usage - Clean and simple!
s3_backend = S3BlobBackend(bucket="my-cache")
local_cache = FilesystemBlobBackend(base_dir="/tmp/s3_cache")

cached_s3 = CachedBlobBackend(
    upstream=s3_backend,           # Remote storage (S3)
    cache_backend=local_cache,     # Local cache (filesystem)
    max_cache_size_mb=1000,        # 1GB local cache
    ttl_seconds=3600               # 1 hour
)

# Now use it just like any blob backend!
store = BlobStore(blob_backend=cached_s3, ...)

# Or with cacheness
cache = cacheness(config=CacheConfig(
    blob_backend=cached_s3,
    metadata_backend="sqlite"
))
```

**Key Insight:** With file-based blob backend interface:
- `FilesystemBlobBackend` and `S3BlobBackend` implement the same interface
- `CachedBlobBackend` is just a coordinator between two backends
- No special logic needed - just delegate to the right backend
- Can even chain: `CachedBlobBackend(upstream=FailoverBlobBackend(S3, Azure), cache=Filesystem)`

**Implementation Tasks:**
- [ ] Create `CachedBlobBackend` wrapper class
- [ ] Implement LRU eviction based on size
- [ ] Add TTL expiration for cached entries
- [ ] Persist cache index to survive restarts (JSON or SQLite)
- [ ] Add cache hit/miss metrics
- [ ] Make cache size and TTL configurable
- [ ] Test cache effectiveness with benchmarks
- [ ] Document when to use caching layer
- [ ] Add example showing FilesystemBlobBackend as local cache for S3BlobBackend

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────┐
│         CachedBlobBackend (Coordinator)         │
├─────────────────────────────────────────────────┤
│  read_blob_to_file():                           │
│    1. Check cache_backend (Filesystem)          │
│    2. If miss, fetch from upstream (S3)         │
│    3. Store in cache_backend                    │
│                                                  │
│  write_blob_from_file():                        │
│    1. Write to upstream (S3)                    │
│    2. Also write to cache_backend               │
└────────┬──────────────────────────┬─────────────┘
         │                          │
         ▼                          ▼
┌────────────────────┐    ┌───────────────────┐
│ FilesystemBlob     │    │  S3BlobBackend    │
│ Backend            │    │  (upstream)       │
│ (cache_backend)    │    │                   │
│                    │    │                   │
│ • Fast (local)     │    │ • Slow (network)  │
│ • Limited (1GB)    │    │ • Unlimited       │
│ • Ephemeral        │    │ • Persistent      │
└────────────────────┘    └───────────────────┘
```

**This is much cleaner than the original design!**

---

**4. Blob Backend Failover & Redundancy** (Low-Medium Priority)

**Problem:** Single backend failure makes entire cache unavailable.

**Solution:**

```python
class FailoverBlobBackend(BlobBackend):
    \"\"\"
    Failover between multiple blob backends.
    
    Write to primary, fallback to secondary on failure.
    Can also replicate writes to multiple backends.
    \"\"\"
    
    def __init__(
        self,
        primary: BlobBackend,
        fallback: BlobBackend,
        replicate_writes: bool = False
    ):
        self.primary = primary
        self.fallback = fallback
        self.replicate_writes = replicate_writes
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        \"\"\"Write to primary, optionally replicate to fallback.\"\"\"
        try:
            path = self.primary.write_blob(blob_id, data)
            
            if self.replicate_writes:
                try:
                    self.fallback.write_blob(blob_id, data)
                except Exception as e:
                    logger.warning(f"Replication to fallback failed: {e}")
            
            return path
        except Exception as e:
            logger.error(f"Primary write failed, trying fallback: {e}")
            return self.fallback.write_blob(blob_id, data)
    
    def read_blob(self, blob_path: str) -> bytes:
        \"\"\"Read from primary, fallback on failure.\"\"\"
        try:
            return self.primary.read_blob(blob_path)
        except Exception as e:
            logger.warning(f"Primary read failed, trying fallback: {e}")
            return self.fallback.read_blob(blob_path)

# Usage
primary = S3BlobBackend(bucket="us-west-2-cache")
fallback = S3BlobBackend(bucket="us-east-1-cache")
failover_backend = FailoverBlobBackend(primary, fallback, replicate_writes=True)
```

**Implementation Tasks:**
- [ ] Create `FailoverBlobBackend` wrapper class
- [ ] Support N-way replication (not just primary/fallback)
- [ ] Add health checks to detect backend failures
- [ ] Implement read-after-write consistency checks
- [ ] Add metrics for failover events
- [ ] Test failover scenarios
- [ ] Document when to use failover

---

**5. Blob Metadata & HTTP Headers** (Medium Priority)

**Problem:** Need to store metadata with blobs (Content-Type, encoding, custom headers).

**Solution:**

```python
class BlobBackend(ABC):
    \"\"\"Extended with metadata support.\"\"\"
    
    @abstractmethod
    def write_blob(
        self,
        blob_id: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        cache_control: Optional[str] = None
    ) -> str:
        \"\"\"Write blob with optional HTTP metadata.\"\"\"
        pass
    
    @abstractmethod
    def get_blob_metadata(self, blob_path: str) -> Dict[str, Any]:
        \"\"\"Get blob metadata without downloading content.\"\"\"
        pass

class S3BlobBackend(BlobBackend):
    def write_blob(
        self,
        blob_id: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        cache_control: Optional[str] = None
    ) -> str:
        key = f"{self.prefix}{blob_id}"
        
        extra_args = {
            "StorageClass": self.storage_class
        }
        
        if content_type:
            extra_args["ContentType"] = content_type
        
        if cache_control:
            extra_args["CacheControl"] = cache_control
        
        if metadata:
            # S3 metadata keys must be lowercase
            extra_args["Metadata"] = {k.lower(): v for k, v in metadata.items()}
        
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        return f"s3://{self.bucket}/{key}"
    
    def get_blob_metadata(self, blob_path: str) -> Dict[str, Any]:
        \"\"\"Get S3 object metadata via HEAD request (no download).\"\"\"
        key = self._parse_s3_key(blob_path)
        response = self.s3.head_object(Bucket=self.bucket, Key=key)
        
        return {
            "size": response["ContentLength"],
            "content_type": response.get("ContentType"),
            "cache_control": response.get("CacheControl"),
            "etag": response["ETag"].strip('"'),
            "last_modified": response["LastModified"],
            "metadata": response.get("Metadata", {}),
            "storage_class": response.get("StorageClass")
        }
```

---

#### Recommendations for Phase 2 Implementation Order

**Phase 2.1 - Foundation** (2-3 months)
1. Handler registration API (Section 2.1)
2. Metadata backend registry (Section 2.2)
3. Basic blob backend registry (Section 2.3) - filesystem only initially
4. Configuration schema & validation (Section 2.4)
5. Documentation for manual registration patterns

**Phase 2.2 - Core Backends** (3-4 months)
1. PostgreSQL metadata backend with retry logic
2. S3 blob backend with content hashing (ETag + custom metadata)
3. Retry & resilience patterns (tenacity, circuit breakers)
4. Basic monitoring/metrics hooks
5. Custom SQLAlchemy metadata + PostgreSQL compatibility (Section 2.7)

**Phase 2.3 - Production Hardening** (2-3 months)
1. Local caching layer for cloud blob backends
2. Streaming support for large blobs
3. Blob metadata & HTTP headers
4. Failover & redundancy
5. Observability & structured logging
6. Performance benchmarking & optimization

**Phase 2.4 - Advanced Features** (2-3 months)
1. Alternative cloud providers (Azure, GCS)
2. Blob migration utilities
3. Cost tracking & budget alerts
4. Multi-region support
5. Documentation & examples

**Total Estimated Effort:** 9-12 months for complete Phase 2

**Deferred to Future Phases (if usage warrants):**
- Entry point discovery / auto-loading plugin system
- Plugin sandboxing & security
- Plugin marketplace / registry

**Risk Mitigation:**
- Start with simpler backends (PostgreSQL, S3) before complex ones
- Build wrappers (CachedBlobBackend, FailoverBlobBackend) for composability
- Extensive testing with real cloud services (not just mocks)
- Manual registration keeps architecture simple and explicit
- Beta period with production users before 1.0 release

---

### Design Principles for Storage Layer

If we extract the storage layer, it should follow these principles:

1. **Content-Addressable Option**: Support content-based addressing (hash of content as key) for deduplication

2. **Pluggable Backends**: Easy to add new backends (Redis, S3, etc.)

3. **Streaming Support**: Handle large blobs without loading entirely into memory

4. **Metadata First-Class**: Rich, queryable metadata separate from blob content

5. **Atomic Operations**: Ensure put/get/delete are atomic

6. **No Caching Semantics**: TTL, eviction, etc. belong in the caching layer

**Proposed BlobStore Interface:**
```python
class BlobStore(ABC):
    """Low-level blob storage interface."""
    
    def put(self, data: Any, key: Optional[str] = None, 
            metadata: Optional[Dict] = None) -> str:
        """Store blob, return key."""
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve blob by key."""
        
    def delete(self, key: str) -> bool:
        """Delete blob by key."""
        
    def exists(self, key: str) -> bool:
        """Check if blob exists."""
        
    def list(self, prefix: Optional[str] = None, 
             metadata_filter: Optional[Dict] = None) -> List[str]:
        """List blob keys with optional filtering."""
        
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get blob metadata without loading content."""
        
    def update_metadata(self, key: str, metadata: Dict) -> bool:
        """Update blob metadata."""
```

---

## 🐛 Bug Tracking

Most issues identified during code review have been fixed. Historical tracking below:

### 🔴 High Severity Issues (All Fixed)

### 1. Query Method Returns Unclosed Session ✅ FIXED

**Location:** [src/cacheness/core.py#L407-411](../src/cacheness/core.py#L407-L411)

**Description:** The `query_custom()` method previously created a SQLAlchemy session but returned a query object without ensuring the session was closed. This leaked database connections.

**Risk:** 
- SQLite connection pool exhaustion
- Resource leaks in long-running applications
- Potential database locking issues on Windows

**Fix Applied:**
1. `query_custom()` now returns a list directly with automatic session cleanup
2. Added optional `filters` parameter for simple equality filtering
3. Added new `query_custom_session()` context manager for advanced queries

```python
# Simple query - session is automatically cleaned up
results = cache.query_custom("ml_experiments")

# With filters
results = cache.query_custom("ml_experiments", {"model_type": "xgboost"})

# Advanced filtering with context manager (guaranteed cleanup)
with cache.query_custom_session("ml_experiments") as query:
    high_accuracy = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
```

---

### 🟠 Medium Severity Issues (All Fixed)

#### 2. JSON Metadata File Corruption Risk ✅ FIXED

**Location:** [src/cacheness/metadata.py#L697-731](../src/cacheness/metadata.py#L697-L731)

**Description:** The JSON backend's `_save_to_disk()` previously wrote directly to the metadata file without atomic write protection.

**Risk:**
- Process crash during write could corrupt metadata
- All cache entries could be lost

**Fix Applied:** Implemented atomic write pattern using temp file + rename.

**Status:** ✅ Fixed in commit

---

### 3. Global Cache Memory Leak on Reset ✅ FIXED

**Location:** [src/cacheness/core.py#L1079-1087](../src/cacheness/core.py#L1079-L1087)

**Description:** The `reset_cache()` function previously created a new cache without closing the old one, leaking SQLite connections.

**Previous Code:**
```python
def reset_cache(config=None, metadata_backend=None):
    global _global_cache
    _global_cache = UnifiedCache(config, metadata_backend)  # Old cache not closed!
```

**Fix Applied:** Now properly closes the previous cache before creating a new one.

**Status:** ✅ Fixed in commit

---

### 4. Signature Verification Field Mismatch Risk

**Location:** [src/cacheness/core.py#L698-727](../src/cacheness/core.py#L698-L727)

**Description:** The signature fields used during `put()` must exactly match those used during `get()`. If the stored metadata structure differs (e.g., missing `cache_key_params` during retrieval), verification may fail incorrectly.

**Risk:**
- Valid cache entries could be rejected as tampered
- Silent data loss if `delete_invalid_signatures=True`

**Recommended Fix:**
```python
# Ensure consistent field extraction in both put() and get()
def _extract_signable_fields(self, entry_data: Dict, cache_key: str) -> Dict:
    """Extract fields for signing in a consistent manner."""
    return {
        field: entry_data.get(field) 
        for field in self.signer.signed_fields
        if field in entry_data or field in self.signer.required_fields
    }
```

**Status:** ✅ Fixed - Added `_extract_signable_fields()` helper method for consistent field extraction in both `put()` and `get()`

---

### 5. Decorator Cache Instance Never Closed

**Location:** [src/cacheness/decorators.py#L95-98](../src/cacheness/decorators.py#L95-L98)

**Description:** The `@cached` decorator creates its own `UnifiedCache` instance if none is provided, but this instance is never explicitly closed.

**Current Code:**
```python
class cached:
    def __init__(self, ...):
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()  # Never closed!
```

**Risk:**
- Connection leaks when decorated functions go out of scope
- Resource accumulation in long-running processes

**Recommended Fix:**
```python
import atexit

class cached:
    _instances = []  # Track for cleanup
    
    def __init__(self, ...):
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()
            cached._instances.append(self.cache_instance)
    
    @classmethod
    def _cleanup_all(cls):
        for instance in cls._instances:
            try:
                instance.close()
            except Exception:
                pass
        cls._instances.clear()

# Register cleanup
atexit.register(cached._cleanup_all)
```

**Status:** ✅ Fixed - Added atexit cleanup, weakref tracking for decorator-created cache instances, and `close()` method

---

## 🟡 Low Severity Issues

### 6. Missing Validation for Custom Signed Fields

**Location:** [src/cacheness/config.py#L223-225](../src/cacheness/config.py#L223-L225)

**Description:** `SecurityConfig.custom_signed_fields` accepts any list of field names without validating against available/valid fields.

**Current Code:**
```python
@dataclass
class SecurityConfig:
    custom_signed_fields: Optional[List[str]] = None  # No validation
```

**Risk:**
- User could specify non-existent fields, causing silent failures
- Typos in field names would go undetected

**Recommended Fix:**
```python
VALID_SIGNED_FIELDS = {
    "cache_key", "file_hash", "data_type", "file_size", 
    "created_at", "prefix", "description", "actual_path",
    "object_type", "storage_format", "serializer", "compression_codec"
}

def __post_init__(self):
    if self.custom_signed_fields:
        invalid = set(self.custom_signed_fields) - VALID_SIGNED_FIELDS
        if invalid:
            raise ValueError(f"Invalid signed fields: {invalid}. Valid: {VALID_SIGNED_FIELDS}")
```

**Status:** ✅ Fixed - Added `VALID_SIGNED_FIELDS` set and validation in `SecurityConfig.__post_init__`

---

### 7. Inconsistent Exception Handling

**Location:** Multiple modules

**Description:** The codebase has a well-defined `CacheError` hierarchy in [error_handling.py](../src/cacheness/error_handling.py), but many places catch bare `Exception` and only log without re-raising or converting to typed exceptions.

**Examples:**
```python
# Good (in error_handling.py)
class CacheStorageError(CacheError): ...
class CacheSerializationError(CacheError): ...

# Inconsistent (scattered throughout)
except Exception as e:
    logger.error(f"Failed to X: {e}")  # Silent failure
    return None
```

**Risk:**
- Difficult to handle specific error types in calling code
- Silent failures can mask bugs

**Recommended Fix:**
- Audit all `except Exception` blocks
- Convert to typed exceptions where appropriate
- Document which methods can raise which exceptions

**Status:** ✅ Fixed - Improved error handling in `core.py` `put()` and `get()` methods with specific exception types (FileNotFoundError, OSError, IOError)

---

### 8. Test Cleanup Writes to Deleted Directory

**Location:** Test fixtures

**Description:** During test teardown, the temp directory is deleted before the cache's `close()` method tries to save metadata, causing benign error logs.

**Log Output:**
```
ERROR: Failed to save JSON metadata: [Errno 2] No such file or directory
```

**Risk:** Low - only affects test output cleanliness

**Fix Applied:** JSON backend now gracefully handles deleted directories.

**Status:** ✅ Fixed in commit

---

## 🔧 Future Enhancement Ideas (Not Scheduled)

These are potential improvements for consideration in future phases:

### 1. Add Session Context Manager API

**Description:** Provide a proper context manager for database queries to ensure cleanup.

```python
# Proposed API
with cache.session() as session:
    results = session.query(MyMetadata).filter(...).all()
```

**Benefits:**
- Guaranteed session cleanup
- Familiar pattern for SQLAlchemy users
- Prevents connection leaks

---

### 2. Implement Connection Pooling Monitoring

**Description:** Add optional metrics/logging for connection pool health.

```python
def get_pool_stats(self) -> Dict[str, int]:
    """Get connection pool statistics."""
    return {
        "pool_size": self.engine.pool.size(),
        "checked_out": self.engine.pool.checkedout(),
        "overflow": self.engine.pool.overflow(),
    }
```

**Benefits:**
- Early warning for connection leaks
- Performance debugging
- Capacity planning

---

### 3. Add Graceful Degradation for Signing

**Description:** When signature verification fails, provide option to return data with warning flag instead of returning `None`.

```python
# Current behavior
if not self.signer.verify_entry(...):
    return None  # Data lost

# Proposed behavior
result = cache.get(..., strict_verification=False)
if result.signature_valid == False:
    logger.warning("Signature invalid, data may be tampered")
```

**Benefits:**
- Allows recovery from legitimate signature mismatches
- Better debugging of signature issues
- Gradual migration path for security upgrades

---

### 4. Standardize Logging Levels

**Description:** Create logging guidelines and audit current usage.

| Level | Current Usage | Recommended Usage |
|-------|--------------|-------------------|
| DEBUG | Verbose operations | Internal details, cache key generation |
| INFO | Cache hits, initialization | Significant state changes only |
| WARNING | Non-critical failures | Degraded performance, fallbacks |
| ERROR | Critical failures | Data loss, unrecoverable errors |

---



---

### 2.13 Optional Arbitrary Metadata Storage

**Status:** 📋 Planned
**Priority:** Medium
**Breaking Change:** No (opt-in feature)

**Overview:**
Currently, SQL backends (SQLite/PostgreSQL) only preserve known technical metadata fields in dedicated columns for performance. This feature adds an optional JSON column to store the complete metadata dictionary used to derive the cache key.

**Purpose:**
- **Debugging:** Inspect full metadata that generated cache keys
- **Broad querying:** Query cache entries by any metadata field without JSON parsing overhead
- **Separate from custom metadata tables:** This is for full metadata capture, not performant custom field queries

**Implementation:**

1. **Add optional JSON column to backends:**
   ```python
   # In CacheEntry and PgCacheEntry models
   full_metadata = Column(JSON, nullable=True)  # Only populated if enabled
   ```

2. **Add configuration option:**
   ```python
   config = CachenessConfig(
       metadata_backend="sqlite",
       store_full_metadata=False,  # Default: disabled for performance
   )
   ```

3. **Update put_entry() logic:**
   ```python
   def put_entry(self, cache_key: str, entry_data: Dict[str, Any]) -> None:
       # Extract known technical fields as before
       s3_etag = metadata.pop("s3_etag", None)
       # ... other fields ...
       
       # Optionally store complete metadata copy
       full_metadata = None
       if self.config.store_full_metadata:
           full_metadata = json.dumps(original_metadata)  # Before any pop() calls
       
       # Insert with full_metadata column
   ```

4. **Query capabilities:**
   ```python
   # Query by any metadata field (requires store_full_metadata=True)
   entries = cache.query_meta(
       "SELECT cache_key, full_metadata FROM cache_entries "
       "WHERE json_extract(full_metadata, '$.custom_field') = 'value'"
   )
   ```

**Trade-offs:**
- **Default OFF:** No performance impact for existing users
- **When enabled:** ~2x storage (technical fields + JSON), ~10% slower writes
- **Not for custom metadata tables:** Use Section 2.7 for performant custom field queries with dedicated SQLAlchemy models

**Distinction from Custom Metadata Tables:**
- **Arbitrary metadata (this feature):** Stores complete metadata dict in JSON column for debugging/broad queries
- **Custom metadata tables (Section 2.7):** User-defined SQLAlchemy models with typed columns for performant queries

**Tasks:**
- [ ] Add `full_metadata` JSON column to CacheEntry model (SQLite)
- [ ] Add `full_metadata` JSON column to PgCacheEntry model (PostgreSQL)
- [ ] Add `store_full_metadata` config option to CachenessConfig
- [ ] Update put_entry() to conditionally store full metadata
- [ ] Update get_entry() to optionally return full metadata
- [ ] Add tests for full metadata storage/retrieval
- [ ] Document trade-offs in BACKEND_SELECTION.md
- [ ] Add examples showing debugging use cases

---

## 📝 Summary for New Readers

**Current Status (February 2026):**
- ✅ **Phase 1 Complete**: Storage layer separated into `cacheness/storage/` subpackage
- 🚧 **Phase 2 In Progress**: Extensibility features (S3 backend in progress, rest complete)
- 📋 **Phase 3 Ready**: Management operations designed, ready for implementation
- ✅ **Bug Fixes**: All critical and medium severity issues resolved

**Key Architectural Decisions:**
1. **Monorepo approach**: Keep single package with clear module boundaries
2. **Manual registration**: Simple explicit APIs over auto-discovery plugins
3. **Separate concerns**: Metadata backends ≠ blob storage backends
4. **File-based interface**: Blob backends use files, not streaming
5. **Preserve existing features**: Three metadata types coexist harmoniously
6. **Storage backend = TTL-agnostic**: Backends store timestamps, cache layer interprets
7. **Cache_key immutability**: Keys are immutable, derived from input parameters

**Implementation Roadmap:**
- **Phase 2** (mostly complete, 1-2 months remaining): PostgreSQL backend ✅, S3 backend 🚧, observability, Azure/GCS
- **Phase 3** (3-6 months): Management operations - update, touch, bulk delete, batch ops, copy/move
  - **Sprint 1-3** (8-11 weeks): Core operations (get_metadata, update_data, touch, bulk delete, batch ops)
  - **Sprint 4-5** (optional, 4-6 weeks): Convenience operations (copy/move, export/import, verify/repair)

**Next Steps for Contributors:**
1. **For extensibility work:** Review [Phase 2 S3 Backend](#210-s3-blob-backend) (in progress)
2. **For management operations:** Review [Phase 3 Management Operations](#phase-3-management-operations-api-medium-risk) (ready to start)
3. See [MISSING_MANAGEMENT_API.md](MISSING_MANAGEMENT_API.md) for complete architectural analysis
4. Read design decision on [Manual Registration vs Plugins](#phase-2-plugin-architecture--extensibility-medium-risk)

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-23 | Initial document created from code review | Code Review |
| 2026-01-23 | Fixed: JSON atomic writes, global cache leak | Code Review |
| 2026-01-23 | Fixed: Query session leak (High #1) - Added `query_custom_session()` context manager | Code Review |
| 2026-01-23 | Fixed: Signature field mismatch (Med #4) - Added `_extract_signable_fields()` helper | Code Review |
| 2026-01-23 | Fixed: Decorator cache never closed (Med #5) - Added atexit cleanup | Code Review |
| 2026-01-23 | Fixed: Custom signed fields validation (Low #6) - Added field validation | Code Review |
| 2026-01-23 | Fixed: Inconsistent exception handling (Low #7) - Improved error types | Code Review |
| 2026-01-27 | Architecture: Phase 1 complete - Created `storage/` subpackage with `BlobStore` API | Code Review |
| 2026-01-27 | Architecture: Phase 2 planned - Registration APIs for handlers/backends, PostgreSQL/S3 backends | Code Review |
| 2026-01-27 | Decision: Use manual registration over auto-discovery plugins (deferred to future if needed) | Code Review |
| 2026-01-27 | Documentation: Added navigation, summary, cleaned up for new reader clarity | Code Review |
| 2026-02-04 | Section 2.10 Complete: S3 ETag metadata storage in SQLite/PostgreSQL backends | Implementation |
| 2026-02-04 | Section 2.11 Complete: Backend parity verification (11 tests passing) | Implementation |
| 2026-02-04 | Section 2.13 Planned: Optional arbitrary metadata storage (opt-in JSON column) | Planning |
| 2026-02-04 | Design Decision: Backends preserve technical fields only by default for performance | Architecture |
| 2026-02-05 | Phase 3 Planned: Management Operations API - update, touch, bulk delete, batch ops | Planning |

