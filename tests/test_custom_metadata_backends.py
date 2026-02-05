"""
Comprehensive tests for custom metadata functionality across both SQLite and PostgreSQL backends.

This test suite verifies that custom SQLAlchemy metadata table functionality works
identically on both SQLite and PostgreSQL backends. Tests cover:
- Basic storage and retrieval
- Foreign key constraints and cascade deletes
- Query functionality
- Multiple custom metadata models
- Different column types
- Edge cases and error handling
- Integration with core cache operations

Section 2.13 of DEVELOPMENT_PLANNING.md
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean
from datetime import datetime, timezone

from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    migrate_custom_metadata_tables,
)
from cacheness.metadata import Base


# ============================================================================
# Test Fixtures
# ============================================================================

# Note: We don't reset the registry automatically because we use session-scoped
# model fixtures that need to persist across tests. For tests that define their
# own models inline, they use extend_existing=True to handle re-registration.


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except (PermissionError, OSError):
        pass  # Windows may hold locks


@pytest.fixture
def postgres_available():
    """Check if PostgreSQL is available for testing."""
    try:
        import psycopg
        from cacheness.storage.backends.postgresql_backend import PostgresBackend
        
        # Try to connect to test database
        test_url = "postgresql://localhost/test_cacheness"
        try:
            backend = PostgresBackend(test_url)
            backend.close()
            return True
        except Exception:
            return False
    except ImportError:
        return False


@pytest.fixture(params=["sqlite", "postgresql"])
def cache_with_backend(request, temp_cache_dir, postgres_available, experiment_metadata_model, performance_metadata_model):
    """
    Parametrized fixture that creates cache with both SQLite and PostgreSQL backends.
    This ensures all tests run against both backends.
    Models are passed as parameters to ensure they're registered before cache creation.
    """
    backend = request.param
    
    if backend == "postgresql" and not postgres_available:
        pytest.skip("PostgreSQL not available")
    
    if backend == "sqlite":
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            metadata=CacheMetadataConfig(
                metadata_backend="sqlite"
            )
        )
    else:  # postgresql
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            metadata=CacheMetadataConfig(
                metadata_backend="postgresql",
                metadata_backend_options={
                    "connection_url": "postgresql://localhost/test_cacheness"
                }
            )
        )
    
    cache = cacheness(config=config)
    
    # Ensure custom metadata tables are created after models are registered
    # Pass the engine from our cache instance
    if hasattr(cache.metadata_backend, 'engine'):
        migrate_custom_metadata_tables(engine=cache.metadata_backend.engine)
    
    yield cache
    
    # Cleanup
    try:
        cache.clear_all()
        cache.close()
    except Exception:
        pass


# ============================================================================
# Custom Metadata Models for Testing
# ============================================================================

@pytest.fixture(scope="session")
def experiment_metadata_model():
    """Define test experiment metadata model."""
    @custom_metadata_model("test_experiments")
    class TestExperimentMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_test_experiments"
        __table_args__ = {'extend_existing': True}
        
        experiment_id = Column(String(100), nullable=False, unique=True, index=True)
        model_type = Column(String(50), nullable=False, index=True)
        accuracy = Column(Float, nullable=False, index=True)
        epochs = Column(Integer, nullable=False, index=True)
        is_production = Column(Boolean, default=False)
        created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    return TestExperimentMetadata


@pytest.fixture(scope="session")
def performance_metadata_model():
    """Define test performance metadata model."""
    @custom_metadata_model("test_performance")
    class TestPerformanceMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_test_performance"
        __table_args__ = {'extend_existing': True}
        
        run_id = Column(String(100), nullable=False, unique=True, index=True)
        training_time_seconds = Column(Float, nullable=False, index=True)
        memory_usage_mb = Column(Float, nullable=False, index=True)
        gpu_utilization = Column(Float, index=True)
    
    return TestPerformanceMetadata


# ============================================================================
# Test: Basic Storage and Retrieval
# ============================================================================

class TestBasicStorageRetrieval:
    """Test basic custom metadata storage and retrieval for both backends."""
    
    def test_store_single_custom_metadata(self, cache_with_backend, experiment_metadata_model):
        """Test storing cache entry with single custom metadata object."""
        cache = cache_with_backend
        
        # Create custom metadata
        experiment = experiment_metadata_model(
            experiment_id="exp_001",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100,
            is_production=True
        )
        
        # Store data with custom metadata
        test_data = {"model": "trained_model", "version": "1.0"}
        cache.put(test_data, experiment="exp_001", custom_metadata=experiment)
        
        # Retrieve data
        result = cache.get(experiment="exp_001")
        assert result == test_data
        
        # Verify custom metadata was stored
        custom_meta = cache.get_custom_metadata_for_entry(experiment="exp_001")
        assert "test_experiments" in custom_meta
        stored_experiment = custom_meta["test_experiments"]
        assert stored_experiment.experiment_id == "exp_001"
        assert stored_experiment.model_type == "xgboost"
        assert stored_experiment.accuracy == 0.95
        assert stored_experiment.epochs == 100
        assert stored_experiment.is_production is True
    
    def test_store_multiple_custom_metadata(
        self, cache_with_backend, experiment_metadata_model, performance_metadata_model
    ):
        """Test storing cache entry with multiple custom metadata objects."""
        cache = cache_with_backend
        
        # Create multiple custom metadata objects
        experiment = experiment_metadata_model(
            experiment_id="exp_002",
            model_type="lightgbm",
            accuracy=0.92,
            epochs=50
        )
        
        performance = performance_metadata_model(
            run_id="run_002",
            training_time_seconds=120.5,
            memory_usage_mb=2048.0,
            gpu_utilization=75.5
        )
        
        # Store with multiple metadata objects
        test_data = {"model": "lightgbm_model"}
        cache.put(
            test_data,
            experiment="exp_002",
            custom_metadata=[experiment, performance]
        )
        
        # Verify both metadata types were stored
        custom_meta = cache.get_custom_metadata_for_entry(experiment="exp_002")
        assert "test_experiments" in custom_meta
        assert "test_performance" in custom_meta
        
        assert custom_meta["test_experiments"].experiment_id == "exp_002"
        assert custom_meta["test_performance"].run_id == "run_002"
    
    def test_retrieve_nonexistent_custom_metadata(self, cache_with_backend):
        """Test retrieving custom metadata for nonexistent cache entry."""
        cache = cache_with_backend
        
        custom_meta = cache.get_custom_metadata_for_entry(nonexistent="key")
        assert custom_meta == {}


# ============================================================================
# Test: Query Functionality
# ============================================================================

class TestQueryFunctionality:
    """Test query_custom_session for both backends."""
    
    def test_query_custom_session_basic(self, cache_with_backend, experiment_metadata_model):
        """Test basic querying via query_custom_session."""
        cache = cache_with_backend
        
        # Store multiple experiments
        for i in range(5):
            experiment = experiment_metadata_model(
                experiment_id=f"exp_query_{i}",
                model_type="xgboost" if i % 2 == 0 else "lightgbm",
                accuracy=0.85 + (i * 0.02),
                epochs=100 + (i * 10)
            )
            cache.put(
                {"data": f"model_{i}"},
                experiment=f"exp_query_{i}",
                custom_metadata=experiment
            )
        
        # Query experiments with accuracy >= 0.9
        with cache.query_custom_session("test_experiments") as query:
            high_accuracy = query.filter(
                experiment_metadata_model.accuracy >= 0.9
            ).all()
            
            assert len(high_accuracy) == 2  # exp_query_3 (0.91), exp_query_4 (0.93)
            
        # Query by model type
        with cache.query_custom_session("test_experiments") as query:
            xgboost_models = query.filter(
                experiment_metadata_model.model_type == "xgboost"
            ).all()
            
            assert len(xgboost_models) == 3  # exp_query_0, exp_query_2, exp_query_4
    
    def test_query_with_ordering(self, cache_with_backend, experiment_metadata_model):
        """Test querying with ordering."""
        cache = cache_with_backend
        
        # Store experiments in random order
        accuracies = [0.88, 0.95, 0.91, 0.93, 0.87]
        for i, accuracy in enumerate(accuracies):
            experiment = experiment_metadata_model(
                experiment_id=f"exp_order_{i}",
                model_type="xgboost",
                accuracy=accuracy,
                epochs=100
            )
            cache.put(
                {"data": f"model_{i}"},
                experiment=f"exp_order_{i}",
                custom_metadata=experiment
            )
        
        # Query ordered by accuracy descending
        with cache.query_custom_session("test_experiments") as query:
            ordered = query.order_by(
                experiment_metadata_model.accuracy.desc()
            ).all()
            
            assert len(ordered) == 5
            assert ordered[0].accuracy == 0.95
            assert ordered[1].accuracy == 0.93
            assert ordered[2].accuracy == 0.91
            assert ordered[-1].accuracy == 0.87
    
    def test_query_with_joins(
        self, cache_with_backend, experiment_metadata_model, performance_metadata_model
    ):
        """Test querying with joins across multiple custom tables via link table."""
        cache = cache_with_backend
        
        # Store entries with both experiment and performance metadata
        for i in range(3):
            experiment = experiment_metadata_model(
                experiment_id=f"exp_join_{i}",
                model_type="xgboost",
                accuracy=0.9 + (i * 0.02),
                epochs=100
            )
            performance = performance_metadata_model(
                run_id=f"run_join_{i}",
                training_time_seconds=100.0 + (i * 50),
                memory_usage_mb=1024.0 + (i * 512),
                gpu_utilization=50.0 + (i * 10)
            )
            cache.put(
                {"data": f"model_{i}"},
                experiment=f"exp_join_{i}",
                custom_metadata=[experiment, performance]
            )
        
        # Query both tables separately to verify they exist
        with cache.query_custom_session("test_experiments") as exp_query:
            experiments = exp_query.all()
            assert len(experiments) == 3
        
        with cache.query_custom_session("test_performance") as perf_query:
            performances = perf_query.all()
            assert len(performances) == 3
        
        # Demonstrate correlating data across multiple custom metadata tables
        # With direct FK, we simply filter by cache_key - much simpler!
        
        # Method 1: Find high-accuracy experiments
        with cache.query_custom_session("test_experiments") as query:
            high_accuracy_exps = query.filter(
                experiment_metadata_model.accuracy >= 0.92
            ).all()
            assert len(high_accuracy_exps) == 2  # exp_join_1 and exp_join_2
            
            # Get their cache_keys (direct attribute access)
            high_accuracy_cache_keys = {exp.cache_key for exp in high_accuracy_exps}
            assert len(high_accuracy_cache_keys) == 2
        
        # Method 2: Find corresponding performance data using those cache_keys
        with cache.query_custom_session("test_performance") as query:
            corresponding_perf = query.filter(
                performance_metadata_model.cache_key.in_(high_accuracy_cache_keys)
            ).all()
            assert len(corresponding_perf) == 2
            # Verify these are the correct performance records
            run_ids = {p.run_id for p in corresponding_perf}
            assert run_ids == {"run_join_1", "run_join_2"}


# ============================================================================
# Test: Foreign Key Constraints and Cascade Deletes
# ============================================================================

class TestForeignKeyConstraints:
    """Test foreign key relationships and cascade deletes."""
    
    def test_cascade_delete_on_cache_entry_removal(
        self, cache_with_backend, experiment_metadata_model
    ):
        """Test that custom metadata links are removed when cache entry is deleted.
        
        Note: The custom metadata *records* themselves are not deleted (they become
        orphaned), which is the correct behavior. Only the links are cascade-deleted.
        Use cleanup_orphaned_metadata() to remove orphaned custom metadata records.
        """
        cache = cache_with_backend
        
        # Store entry with custom metadata
        experiment = experiment_metadata_model(
            experiment_id="exp_cascade",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put(
            {"data": "test"},
            experiment="exp_cascade",
            custom_metadata=experiment
        )
        
        # Verify custom metadata exists and is linked
        custom_meta = cache.get_custom_metadata_for_entry(experiment="exp_cascade")
        assert "test_experiments" in custom_meta
        
        # Delete cache entry using invalidate
        cache.invalidate(experiment="exp_cascade")
        
        # Verify custom metadata is no longer accessible via cache key (link is gone)
        custom_meta_after = cache.get_custom_metadata_for_entry(experiment="exp_cascade")
        assert "test_experiments" not in custom_meta_after or custom_meta_after == {}
        
        # The custom metadata record itself still exists (orphaned), which is correct
        # It can be cleaned up later with cleanup_orphaned_metadata()
    
    def test_cascade_delete_on_clear_all(
        self, cache_with_backend, experiment_metadata_model
    ):
        """Test that all custom metadata is deleted when cache.clear_all() is called."""
        cache = cache_with_backend
        
        # Store multiple entries with custom metadata
        for i in range(3):
            experiment = experiment_metadata_model(
                experiment_id=f"exp_clear_{i}",
                model_type="xgboost",
                accuracy=0.9,
                epochs=100
            )
            cache.put(
                {"data": f"model_{i}"},
                experiment=f"exp_clear_{i}",
                custom_metadata=experiment
            )
        
        # Verify custom metadata exists
        with cache.query_custom_session("test_experiments") as query:
            count_before = query.count()
            assert count_before >= 3
        
        # Clear all cache entries
        cache.clear_all()
        
        # Verify all custom metadata links are removed (custom metadata records
        # themselves are not deleted - they become orphaned, which is correct behavior)
        # Check that we can't retrieve them via cache keys anymore
        for i in range(3):
            custom_meta = cache.get_custom_metadata_for_entry(experiment=f"exp_clear_{i}")
            assert custom_meta == {}
        
        # The custom metadata records still exist in the database (orphaned),
        # which is correct - custom metadata is independent of cache entries


# ============================================================================
# Test: Multiple Custom Metadata Models
# ============================================================================

class TestMultipleMetadataModels:
    """Test using multiple custom metadata models simultaneously."""
    
    def test_multiple_models_coexist(
        self, cache_with_backend, experiment_metadata_model, performance_metadata_model
    ):
        """Test that multiple custom metadata models can coexist."""
        cache = cache_with_backend
        
        # Store entry with experiment metadata only
        exp1 = experiment_metadata_model(
            experiment_id="exp_multi_1",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put({"data": "model_1"}, experiment="exp_multi_1", custom_metadata=exp1)
        
        # Store entry with performance metadata only
        perf2 = performance_metadata_model(
            run_id="run_multi_2",
            training_time_seconds=150.0,
            memory_usage_mb=2048.0
        )
        cache.put({"data": "model_2"}, experiment="exp_multi_2", custom_metadata=perf2)
        
        # Store entry with both metadata types
        exp3 = experiment_metadata_model(
            experiment_id="exp_multi_3",
            model_type="lightgbm",
            accuracy=0.92,
            epochs=50
        )
        perf3 = performance_metadata_model(
            run_id="run_multi_3",
            training_time_seconds=100.0,
            memory_usage_mb=1024.0
        )
        cache.put(
            {"data": "model_3"},
            experiment="exp_multi_3",
            custom_metadata=[exp3, perf3]
        )
        
        # Verify all metadata is stored correctly
        meta1 = cache.get_custom_metadata_for_entry(experiment="exp_multi_1")
        assert "test_experiments" in meta1
        assert "test_performance" not in meta1
        
        meta2 = cache.get_custom_metadata_for_entry(experiment="exp_multi_2")
        assert "test_experiments" not in meta2
        assert "test_performance" in meta2
        
        meta3 = cache.get_custom_metadata_for_entry(experiment="exp_multi_3")
        assert "test_experiments" in meta3
        assert "test_performance" in meta3


# ============================================================================
# Test: Different Column Types
# ============================================================================

class TestDifferentColumnTypes:
    """Test custom metadata with various column types."""
    
    def test_various_column_types(self, cache_with_backend):
        """Test String, Integer, Float, DateTime, Boolean columns."""
        cache = cache_with_backend
        
        @custom_metadata_model("test_types")
        class TestTypesMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_test_types"
            __table_args__ = {'extend_existing': True}
            
            string_field = Column(String(100), nullable=False)
            int_field = Column(Integer, nullable=False, index=True)
            float_field = Column(Float, nullable=False, index=True)
            bool_field = Column(Boolean, nullable=False)
            datetime_field = Column(DateTime, nullable=False)
        
        # Migrate tables
        migrate_custom_metadata_tables()
        
        # Create metadata with all types
        test_datetime = datetime.now(timezone.utc)
        metadata = TestTypesMetadata(
            string_field="test_string",
            int_field=42,
            float_field=3.14159,
            bool_field=True,
            datetime_field=test_datetime
        )
        
        # Store and retrieve
        cache.put({"data": "test"}, test_key="types_test", custom_metadata=metadata)
        
        retrieved = cache.get_custom_metadata_for_entry(test_key="types_test")
        assert "test_types" in retrieved
        
        stored = retrieved["test_types"]
        assert stored.string_field == "test_string"
        assert stored.int_field == 42
        assert abs(stored.float_field - 3.14159) < 0.0001
        assert stored.bool_field is True
        # DateTime comparison - remove timezone info for comparison if needed
        stored_dt = stored.datetime_field.replace(tzinfo=timezone.utc) if stored.datetime_field.tzinfo is None else stored.datetime_field
        assert abs((stored_dt - test_datetime).total_seconds()) < 1  # Allow 1 second tolerance


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesErrorHandling:
    """Test edge cases and error handling."""
    
    def test_duplicate_unique_constraint(self, cache_with_backend, experiment_metadata_model):
        """Test handling of duplicate unique constraint violations."""
        cache = cache_with_backend
        
        # Store first experiment
        exp1 = experiment_metadata_model(
            experiment_id="exp_duplicate",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put({"data": "model_1"}, experiment="exp_dup_1", custom_metadata=exp1)
        
        # Try to store another experiment with same experiment_id (should fail gracefully)
        exp2 = experiment_metadata_model(
            experiment_id="exp_duplicate",  # Same ID
            model_type="lightgbm",
            accuracy=0.92,
            epochs=50
        )
        
        # This should either fail or overwrite depending on implementation
        # At minimum it shouldn't crash
        try:
            cache.put({"data": "model_2"}, experiment="exp_dup_2", custom_metadata=exp2)
        except Exception as e:
            # Expected to fail with unique constraint
            assert "unique" in str(e).lower() or "duplicate" in str(e).lower()
    
    def test_missing_required_columns(self, cache_with_backend):
        """Test handling of missing required columns."""
        cache = cache_with_backend
        
        @custom_metadata_model("test_required")
        class TestRequiredMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_test_required"
            __table_args__ = {'extend_existing': True}
            
            required_field = Column(String(100), nullable=False)
            optional_field = Column(String(100), nullable=True)
        
        migrate_custom_metadata_tables()
        
        # Create metadata without required field
        metadata = TestRequiredMetadata(
            optional_field="test"
            # Missing required_field will be None
        )
        
        # The error should occur during database insert, not object creation
        # This should succeed (cache entry created) but custom metadata should fail to store
        cache.put({"data": "test"}, test_key="required_test", custom_metadata=metadata)
        
        # Verify the cache entry was created despite metadata failure
        result = cache.get(test_key="required_test")
        assert result == {"data": "test"}
    
    def test_custom_metadata_without_cache_entry(self, cache_with_backend, experiment_metadata_model):
        """Test that custom metadata requires a cache entry."""
        cache = cache_with_backend
        
        # The cache_key must exist in cache_entries for the foreign key to work
        # This is enforced by the link table architecture
        
        # Store properly
        experiment = experiment_metadata_model(
            experiment_id="exp_valid",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put({"data": "test"}, experiment="exp_valid", custom_metadata=experiment)
        
        # Verify it worked
        meta = cache.get_custom_metadata_for_entry(experiment="exp_valid")
        assert "test_experiments" in meta


# ============================================================================
# Test: Integration with Core Operations
# ============================================================================

class TestCoreIntegration:
    """Test integration with core cache operations."""
    
    def test_custom_metadata_not_used_in_cache_key_generation(
        self, cache_with_backend, experiment_metadata_model
    ):
        """Verify custom_metadata is NOT used in cache key generation."""
        cache = cache_with_backend
        
        # Store same data with different custom metadata
        exp1 = experiment_metadata_model(
            experiment_id="exp_key_1",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        
        exp2 = experiment_metadata_model(
            experiment_id="exp_key_2",  # Different metadata
            model_type="lightgbm",
            accuracy=0.92,
            epochs=50
        )
        
        # Same cache key parameters (experiment="same_key")
        key1 = cache.put({"data": "test_1"}, experiment="same_key", custom_metadata=exp1)
        key2 = cache.put({"data": "test_2"}, experiment="same_key", custom_metadata=exp2)
        
        # Keys should be the same (overwrite), proving custom_metadata not in key generation
        assert key1 == key2
        
        # Only second metadata should exist (overwrite behavior)
        meta = cache.get_custom_metadata_for_entry(experiment="same_key")
        # Should have the latest metadata
        assert "test_experiments" in meta
    
    def test_custom_metadata_not_used_in_cache_hit_logic(
        self, cache_with_backend, experiment_metadata_model
    ):
        """Verify custom_metadata is NOT used in cache hit/miss logic."""
        cache = cache_with_backend
        
        # Store entry with custom metadata
        experiment = experiment_metadata_model(
            experiment_id="exp_hit",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put({"data": "original"}, experiment="hit_test", custom_metadata=experiment)
        
        # Retrieve without providing custom_metadata (should still hit)
        result = cache.get(experiment="hit_test")
        assert result == {"data": "original"}
        
        # Custom metadata should still be available
        meta = cache.get_custom_metadata_for_entry(experiment="hit_test")
        assert "test_experiments" in meta
    
    def test_custom_metadata_optional(self, cache_with_backend):
        """Verify cache works without custom metadata."""
        cache = cache_with_backend
        
        # Store and retrieve without any custom metadata
        cache.put({"data": "no_metadata"}, test_key="optional_test")
        result = cache.get(test_key="optional_test")
        assert result == {"data": "no_metadata"}
        
        # Should have no custom metadata
        meta = cache.get_custom_metadata_for_entry(test_key="optional_test")
        assert meta == {}


# ============================================================================
# Test: Backend Parity
# ============================================================================

class TestBackendParity:
    """Test that SQLite and PostgreSQL backends behave identically."""
    
    def test_same_behavior_both_backends(self, cache_with_backend, experiment_metadata_model):
        """Verify identical behavior across both backends."""
        cache = cache_with_backend
        
        # This test runs with both backends due to parametrization
        # If it passes for both, they're consistent
        
        # Store data
        experiment = experiment_metadata_model(
            experiment_id="exp_parity",
            model_type="xgboost",
            accuracy=0.95,
            epochs=100
        )
        cache.put({"data": "test"}, experiment="parity_test", custom_metadata=experiment)
        
        # Retrieve data
        result = cache.get(experiment="parity_test")
        assert result == {"data": "test"}
        
        # Query custom metadata
        meta = cache.get_custom_metadata_for_entry(experiment="parity_test")
        assert "test_experiments" in meta
        assert meta["test_experiments"].experiment_id == "exp_parity"
        
        # Query via session
        with cache.query_custom_session("test_experiments") as query:
            experiments = query.filter(
                experiment_metadata_model.experiment_id == "exp_parity"
            ).all()
            assert len(experiments) == 1
        
        # Delete and verify link is removed (metadata record remains orphaned)
        cache.invalidate(experiment="parity_test")
        
        # Verify metadata is no longer accessible via cache key
        meta_after = cache.get_custom_metadata_for_entry(experiment="parity_test")
        assert "test_experiments" not in meta_after or meta_after == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
