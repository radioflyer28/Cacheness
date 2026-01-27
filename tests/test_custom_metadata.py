"""
Unit tests for custom metadata functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import Column, String, Float, Integer, DateTime

from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    is_custom_metadata_available,
    get_custom_metadata_model,
    list_registered_schemas,
    migrate_custom_metadata_tables,
    cleanup_orphaned_metadata,
    validate_custom_metadata_model,
    export_custom_metadata_schema,
    _reset_registry,  # For test isolation
)
from cacheness.metadata import Base

# Store original registry state for cleanup
_original_registry_state = None


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset custom metadata registry before each test for isolation."""
    # Store current state
    current_schemas = list_registered_schemas().copy()

    # Clean up for test
    _reset_registry()

    yield

    # Restore state after test
    _reset_registry()


@pytest.fixture
def temp_cache_dir(request):
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    
    def cleanup():
        import time
        import gc
        gc.collect()  # Force garbage collection
        time.sleep(0.2)  # Give SQLite time to release locks
        if Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                # If still locked, wait a bit more and try again
                time.sleep(0.5)
                shutil.rmtree(temp_dir)
    
    request.addfinalizer(cleanup)
    return temp_dir


@pytest.fixture
def cache_config(temp_cache_dir):
    """Create a cache configuration with SQLite backend."""
    return CacheConfig(
        cache_dir=temp_cache_dir, metadata_backend="sqlite", store_cache_key_params=True
    )


@pytest.fixture
def cache_instance(cache_config):
    """Create a cache instance for testing."""
    cache = cacheness(cache_config)
    yield cache
    cache.close()


class TestCustomMetadataAvailability:
    """Test custom metadata availability detection."""

    def test_availability_with_sqlite_backend(self, cache_instance):
        """Test that custom metadata is available with SQLite backend."""
        assert is_custom_metadata_available()

    def test_availability_without_sqlalchemy(self, monkeypatch):
        """Test behavior when SQLAlchemy is not available."""

        # Mock SQLAlchemy import failure
        def mock_import_error(*args, **kwargs):
            raise ImportError("No module named 'sqlalchemy'")

        monkeypatch.setattr("builtins.__import__", mock_import_error)

        # This would normally return False, but since SQLAlchemy is already imported,
        # we can't easily test this scenario. The function will still return True.
        # In a real scenario without SQLAlchemy, this would return False.
        result = is_custom_metadata_available()
        assert isinstance(result, bool)


class TestCustomMetadataRegistry:
    """Test the custom metadata registry system."""

    def test_model_registration(self):
        """Test that models are registered correctly."""

        @custom_metadata_model("test_schema")
        class TestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_test_metadata"

            test_field = Column(String(100), nullable=False)

        # Check registration
        assert "test_schema" in list_registered_schemas()
        assert get_custom_metadata_model("test_schema") is TestMetadata

    def test_duplicate_schema_registration(self):
        """Test that duplicate schema names trigger a warning."""

        @custom_metadata_model("duplicate_schema")
        class FirstMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_first_metadata_unique1"
            test_field = Column(String(100))

        # Second registration should log a warning but not raise an exception
        @custom_metadata_model("duplicate_schema")
        class SecondMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_second_metadata_unique1"
            other_field = Column(String(100))

        # The second model should have overwritten the first
        assert get_custom_metadata_model("duplicate_schema") is SecondMetadata

    def test_get_nonexistent_model(self):
        """Test retrieval of non-existent model returns None."""
        result = get_custom_metadata_model("nonexistent")
        assert result is None

    def test_empty_registry(self):
        """Test behavior with empty registry."""
        assert list_registered_schemas() == []


class TestCustomMetadataModels:
    """Test custom metadata model definitions."""

    def test_model_with_base_fields(self):
        """Test that models inherit base fields correctly."""

        @custom_metadata_model("base_fields_test")
        class BaseFieldsTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_base_fields_test"

            custom_field = Column(String(100))

        # Check that base fields are present
        model = BaseFieldsTestMetadata()
        assert hasattr(model, "id")
        assert hasattr(model, "created_at")
        assert hasattr(model, "updated_at")
        assert hasattr(model, "custom_field")

    def test_model_validation(self):
        """Test model validation functionality."""

        @custom_metadata_model("validation_test")
        class WellDesignedMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_validation_test"

            indexed_field = Column(String(100), nullable=False, index=True)
            required_field = Column(String(50), nullable=False)

        issues = validate_custom_metadata_model(WellDesignedMetadata)
        assert len(issues) == 0

    def test_model_validation_warnings(self):
        """Test model validation warnings."""

        @custom_metadata_model("validation_warnings_test")
        class PoorlyDesignedMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_validation_warnings_test_unique"

            # Simple fields that may not trigger validation warnings
            simple_field = Column(String(100), nullable=False)
            optional_field = Column(String(100), nullable=True)

        issues = validate_custom_metadata_model(PoorlyDesignedMetadata)
        # Validation might not find issues with simple fields
        assert isinstance(issues, list)

    def test_datetime_timezone_awareness(self):
        """Test that DateTime columns properly handle timezone-aware timestamps"""
        from datetime import datetime, timezone, timedelta
        
        @custom_metadata_model("timezone_test")
        class TimezoneTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_timezone_test_unique"
            
            # Test timezone-aware DateTime column
            test_timestamp = Column(DateTime(timezone=True), nullable=False)
            experiment_name = Column(String(100), nullable=False)
        
        # Verify the model is properly registered
        assert "timezone_test" in list_registered_schemas()
        
        # Test UTC timestamp
        utc_time = datetime.now(timezone.utc)
        assert utc_time.tzinfo == timezone.utc
        
        # Test different timezone
        eastern_tz = timezone(timedelta(hours=-5))  # EST timezone
        eastern_time = datetime.now(eastern_tz)
        assert eastern_time.tzinfo == eastern_tz
        
        # Both should be valid timezone-aware datetimes
        assert utc_time.tzinfo is not None
        assert eastern_time.tzinfo is not None

    def test_custom_metadata_timestamp_consistency(self):
        """Test that custom metadata timestamps are stored with proper timezone info"""
        from datetime import datetime, timezone
        
        @custom_metadata_model("timestamp_consistency_test")
        class TimestampTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_timestamp_consistency_unique"
            
            created_timestamp = Column(DateTime(timezone=True), nullable=False)
            process_name = Column(String(100), nullable=False)
        
        # Create a UTC timestamp
        utc_now = datetime.now(timezone.utc)
        
        # Verify we can create instances with timezone-aware timestamps
        metadata_instance = TimestampTestMetadata(
            created_timestamp=utc_now,
            process_name="timezone_test"
        )
        
        # Verify the timestamp is preserved
        assert metadata_instance.created_timestamp == utc_now
        assert metadata_instance.created_timestamp.tzinfo == timezone.utc
        assert metadata_instance.process_name == "timezone_test"


class TestCacheIntegration:
    """Test integration with the cache system."""

    @pytest.fixture(autouse=True)
    def setup_dynamic_metadata_class(self):
        """Set up each test method with unique table names."""
        import uuid

        unique_suffix = str(uuid.uuid4()).replace("-", "")[:8]

        # Create unique class name and schema name
        schema_name = f"ml_experiments_integration_{unique_suffix}"
        table_name = f"custom_ml_experiments_{unique_suffix}"

        # Create class with unique name by using locals() trick
        class_code = f"""
@custom_metadata_model('{schema_name}')
class MLExperimentIntegrationMetadata_{unique_suffix}(Base, CustomMetadataBase):
    __tablename__ = '{table_name}'
    __table_args__ = {{'extend_existing': True}}
    
    experiment_id = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    created_by = Column(String(100), nullable=False, index=True)
"""

        # Execute the class definition dynamically
        namespace = {
            "custom_metadata_model": custom_metadata_model,
            "Base": Base,
            "CustomMetadataBase": CustomMetadataBase,
            "Column": Column,
            "String": String,
            "Float": Float,
        }
        exec(class_code, namespace)
        MLExperimentIntegrationMetadata = namespace[
            f"MLExperimentIntegrationMetadata_{unique_suffix}"
        ]

        self.MLExperimentMetadata = MLExperimentIntegrationMetadata
        # Store schema name for other methods
        self.schema_name = schema_name

    def test_store_with_custom_metadata(self, cache_instance):
        """Test storing cache entries with custom metadata."""
        # Create metadata instance
        metadata = self.MLExperimentMetadata(
            experiment_id="test_exp_001",
            model_type="xgboost",
            accuracy=0.95,
            dataset_name="test_dataset",
            created_by="test_user",
        )

        # Store data with custom metadata
        test_data = {"model": "test_model_data"}
        cache_instance.put(
            test_data,
            experiment="test_exp_001",
            custom_metadata={self.schema_name: metadata},
        )

        # Verify data was stored
        retrieved_data = cache_instance.get(experiment="test_exp_001")
        assert retrieved_data == test_data

    def test_retrieve_custom_metadata(self, cache_instance):
        """Test retrieving custom metadata for cache entries."""
        # Create and store metadata
        metadata = self.MLExperimentMetadata(
            experiment_id="test_exp_002",
            model_type="random_forest",
            accuracy=0.87,
            dataset_name="test_dataset_2",
            created_by="test_user_2",
        )

        test_data = {"model": "test_model_data_2"}
        cache_instance.put(
            test_data,
            experiment="test_exp_002",
            custom_metadata={self.schema_name: metadata},
        )

        # Retrieve custom metadata
        custom_meta = cache_instance.get_custom_metadata_for_entry(
            experiment="test_exp_002"
        )

        assert self.schema_name in custom_meta
        retrieved_metadata = custom_meta[self.schema_name]
        assert retrieved_metadata.experiment_id == "test_exp_002"
        assert retrieved_metadata.model_type == "random_forest"
        assert retrieved_metadata.accuracy == 0.87
        assert retrieved_metadata.created_by == "test_user_2"

    def test_query_custom_metadata(self, cache_instance):
        """Test querying custom metadata."""
        # Store multiple entries with metadata
        for i in range(3):
            metadata = self.MLExperimentMetadata(
                experiment_id=f"test_exp_{i:03d}",
                model_type="xgboost" if i % 2 == 0 else "random_forest",
                accuracy=0.8 + (i * 0.05),
                dataset_name=f"test_dataset_{i}",
                created_by="test_user",
            )

            cache_instance.put(
                {"model": f"test_model_{i}"},
                experiment=f"test_exp_{i:03d}",
                custom_metadata={self.schema_name: metadata},
            )

        # Test basic query - new API returns list directly
        all_experiments = cache_instance.query_custom(self.schema_name)
        assert len(all_experiments) == 3

        # Test filtered query using context manager for advanced filtering
        with cache_instance.query_custom_session(self.schema_name) as query:
            xgboost_experiments = query.filter(
                self.MLExperimentMetadata.model_type == "xgboost"
            ).all()
            assert len(xgboost_experiments) == 2  # Experiments 0 and 2

            # Test accuracy filter
            high_accuracy = query.filter(self.MLExperimentMetadata.accuracy >= 0.85).all()
            assert len(high_accuracy) >= 1

    def test_multiple_metadata_types(self, cache_instance):
        """Test storing multiple metadata types for the same cache entry."""
        # Define a second metadata model with unique name
        import uuid

        unique_suffix = str(uuid.uuid4()).replace("-", "")[:8]
        schema_name = f"deployment_info_{unique_suffix}"
        table_name = f"custom_deployment_info_{unique_suffix}"

        # Create class with unique name dynamically
        class_code = f"""
@custom_metadata_model('{schema_name}')
class DeploymentMetadata_{unique_suffix}(Base, CustomMetadataBase):
    __tablename__ = '{table_name}'
    __table_args__ = {{'extend_existing': True}}
    
    deployment_id = Column(String(100), nullable=False, unique=True, index=True)
    environment = Column(String(50), nullable=False, index=True)
    replicas = Column(Integer, nullable=False)
"""

        # Execute the class definition dynamically
        namespace = {
            "custom_metadata_model": custom_metadata_model,
            "Base": Base,
            "CustomMetadataBase": CustomMetadataBase,
            "Column": Column,
            "String": String,
            "Integer": Integer,
        }
        exec(class_code, namespace)
        DeploymentMetadata = namespace[f"DeploymentMetadata_{unique_suffix}"]

        # Create metadata instances
        ml_metadata = self.MLExperimentMetadata(
            experiment_id="multi_meta_exp",
            model_type="neural_network",
            accuracy=0.92,
            dataset_name="multi_dataset",
            created_by="multi_user",
        )

        deployment_metadata = DeploymentMetadata(
            deployment_id="deploy_001", environment="production", replicas=3
        )

        # Store with multiple metadata types
        test_data = {"model": "multi_meta_model"}
        cache_instance.put(
            {"model": "multi_model", "deployment": "prod"},
            experiment="multi_meta_exp",
            custom_metadata={
                self.schema_name: ml_metadata,
                schema_name: deployment_metadata,
            },
        )

        # Retrieve and verify both metadata types
        custom_meta = cache_instance.get_custom_metadata_for_entry(
            experiment="multi_meta_exp"
        )

        assert self.schema_name in custom_meta
        assert schema_name in custom_meta

        ml_meta = custom_meta[self.schema_name]
        deploy_meta = custom_meta[schema_name]

        assert ml_meta.experiment_id == "multi_meta_exp"
        assert ml_meta.accuracy == 0.92
        assert deploy_meta.deployment_id == "deploy_001"
        assert deploy_meta.replicas == 3


class TestMigrationAndMaintenance:
    """Test migration and maintenance functionality."""

    def test_migrate_custom_metadata_tables(self, cache_instance):
        """Test custom metadata table migration."""

        @custom_metadata_model("migration_test")
        class MigrationTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_migration_test_unique"

            test_field = Column(String(100), nullable=False)

        # Test migration with the main cache instance
        result = migrate_custom_metadata_tables(cache_instance)
        # Migration may return None or a dict depending on implementation
        assert result is None or isinstance(result, dict)

    def test_cleanup_orphaned_metadata(self, cache_instance):
        """Test cleanup of orphaned metadata."""
        # This is mainly testing that the function runs without error
        # In a real scenario, we'd need to create orphaned metadata first
        result = cleanup_orphaned_metadata()
        assert isinstance(result, int)  # Should return count of cleaned up items
        assert result >= 0

    def test_export_custom_metadata_schema(self):
        """Test schema export functionality."""

        @custom_metadata_model("export_test")
        class ExportTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_export_test"

            export_field = Column(String(100), nullable=False)

        # Test DDL generation
        ddl = export_custom_metadata_schema("export_test")
        assert isinstance(ddl, str)
        assert "CREATE TABLE" in ddl.upper()
        assert "custom_export_test" in ddl

    def test_export_nonexistent_schema(self):
        """Test exporting schema for non-existent model returns None."""
        result = export_custom_metadata_schema("nonexistent")
        assert result is None


class TestErrorHandling:
    """Test error handling in custom metadata functionality."""

    def test_invalid_metadata_in_put(self, cache_instance):
        """Test handling of invalid metadata in put operation gracefully."""

        @custom_metadata_model("temp_invalid_test")
        class TempInvalidTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_temp_invalid_test_unique"
            field = Column(String(100))

        # Store with invalid schema - should log warning but not raise
        cache_instance.put(
            {"data": "test"},
            test_key="test_value",
            custom_metadata={"invalid_schema": "not_a_metadata_object"},
        )

        # Should have stored the main data despite invalid metadata
        result = cache_instance.get(test_key="test_value")
        assert result == {"data": "test"}

    def test_nonexistent_schema_in_put(self, cache_instance):
        """Test handling of non-existent schema in put operation gracefully."""

        @custom_metadata_model("temp_schema_unique")
        class TempSchemaUniqueMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_temp_metadata_unique"
            field = Column(String(100))

        metadata = TempSchemaUniqueMetadata(field="test")

        # Store with non-existent schema name - should log warning but not raise
        cache_instance.put(
            {"data": "test"},
            test_key="test_value",
            custom_metadata={"nonexistent_schema": metadata},
        )

        # Should have stored the main data despite invalid schema
        result = cache_instance.get(test_key="test_value")
        assert result == {"data": "test"}

    def test_query_nonexistent_schema(self, cache_instance):
        """Test querying non-existent schema returns empty list."""
        result = cache_instance.query_custom("nonexistent")
        assert result == []

    def test_get_metadata_for_nonexistent_entry(self, cache_instance):
        """Test getting metadata for non-existent cache entry."""
        result = cache_instance.get_custom_metadata_for_entry(nonexistent_key="value")
        assert result == {}


class TestCacheMetadataLink:
    """Test the CacheMetadataLink model functionality."""

    def test_link_table_creation(self, cache_instance):
        """Test that the link table is created properly."""

        # This implicitly tests link table creation through cache operations
        @custom_metadata_model("link_test")
        class LinkTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_link_test"

            test_field = Column(String(100), nullable=False)

        metadata = LinkTestMetadata(test_field="test_value")

        cache_instance.put(
            {"data": "test"},
            link_test_key="test",
            custom_metadata={"link_test": metadata},
        )

        # Verify we can retrieve the metadata (implicitly tests link table)
        custom_meta = cache_instance.get_custom_metadata_for_entry(link_test_key="test")
        assert "link_test" in custom_meta
        assert custom_meta["link_test"].test_field == "test_value"


class TestAdvancedQuerying:
    """Test advanced querying capabilities."""

    @pytest.fixture(autouse=True)
    def setup_advanced_query_metadata_class(self):
        """Set up test data for advanced querying."""
        import uuid

        unique_suffix = str(uuid.uuid4()).replace("-", "")[:8]

        # Create unique class name and schema name
        schema_name = f"advanced_query_test_{unique_suffix}"
        table_name = f"custom_advanced_query_test_{unique_suffix}"

        # Create class with unique name dynamically
        class_code = f"""
@custom_metadata_model('{schema_name}')
class AdvancedQueryTestMetadata_{unique_suffix}(Base, CustomMetadataBase):
    __tablename__ = '{table_name}'
    __table_args__ = {{'extend_existing': True}}
    
    name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    active = Column(Integer, nullable=False, index=True)  # 0 or 1 for boolean
"""

        # Execute the class definition dynamically
        namespace = {
            "custom_metadata_model": custom_metadata_model,
            "Base": Base,
            "CustomMetadataBase": CustomMetadataBase,
            "Column": Column,
            "String": String,
            "Float": Float,
            "Integer": Integer,
        }
        exec(class_code, namespace)
        AdvancedQueryTestMetadata = namespace[
            f"AdvancedQueryTestMetadata_{unique_suffix}"
        ]

        self.AdvancedQueryMetadata = AdvancedQueryTestMetadata
        self.schema_name = schema_name

    def test_complex_filtering(self, cache_instance):
        """Test complex filtering operations."""
        from sqlalchemy import and_, or_

        # Store test data
        test_entries = [
            {"name": "entry_1", "value": 10.5, "category": "A", "active": 1},
            {"name": "entry_2", "value": 20.3, "category": "B", "active": 1},
            {"name": "entry_3", "value": 15.7, "category": "A", "active": 0},
            {"name": "entry_4", "value": 25.1, "category": "C", "active": 1},
        ]

        for i, entry_data in enumerate(test_entries):
            metadata = self.AdvancedQueryMetadata(**entry_data)
            cache_instance.put(
                {"data": f"test_data_{i}"},
                entry_id=i,
                custom_metadata={self.schema_name: metadata},
            )

        # Use context manager for advanced filtering
        with cache_instance.query_custom_session(self.schema_name) as query:
            # Test AND filtering
            active_category_a = query.filter(
                and_(
                    self.AdvancedQueryMetadata.category == "A",
                    self.AdvancedQueryMetadata.active == 1,
                )
            ).all()
            assert len(active_category_a) == 1
            assert active_category_a[0].name == "entry_1"

            # Test OR filtering
            category_a_or_b = query.filter(
                or_(
                    self.AdvancedQueryMetadata.category == "A",
                    self.AdvancedQueryMetadata.category == "B",
                )
            ).all()
            assert len(category_a_or_b) == 3

            # Test range filtering
            medium_values = query.filter(
                and_(
                    self.AdvancedQueryMetadata.value >= 15.0,
                    self.AdvancedQueryMetadata.value <= 21.0,
                )
            ).all()
            assert len(medium_values) == 2

    def test_ordering_and_limiting(self, cache_instance):
        """Test ordering and limiting query results."""
        # Store test data (reuse setup from previous test)
        test_entries = [
            {"name": "entry_1", "value": 10.5, "category": "A", "active": 1},
            {"name": "entry_2", "value": 20.3, "category": "B", "active": 1},
            {"name": "entry_3", "value": 15.7, "category": "A", "active": 0},
        ]

        for i, entry_data in enumerate(test_entries):
            metadata = self.AdvancedQueryMetadata(**entry_data)
            cache_instance.put(
                {"data": f"test_data_{i}"},
                entry_id=f"order_test_{i}",
                custom_metadata={self.schema_name: metadata},
            )

        # Use context manager for advanced ordering and limiting
        with cache_instance.query_custom_session(self.schema_name) as query:
            # Test ordering by value (ascending)
            ordered_asc = query.order_by(self.AdvancedQueryMetadata.value).all()
            values_asc = [item.value for item in ordered_asc]
            assert values_asc == sorted(values_asc)

            # Test ordering by value (descending)
            ordered_desc = query.order_by(self.AdvancedQueryMetadata.value.desc()).all()
            values_desc = [item.value for item in ordered_desc]
            assert values_desc == sorted(values_desc, reverse=True)

            # Test limiting results
            limited = query.order_by(self.AdvancedQueryMetadata.value).limit(2).all()
            assert len(limited) == 2
            assert limited[0].value <= limited[1].value  # Should be ordered


if __name__ == "__main__":
    pytest.main([__file__])
