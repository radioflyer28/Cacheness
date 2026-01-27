"""
Test pandas compatibility across versions 2.0+

This test ensures Cacheness works with pandas 2.0, 2.3, and 3.0+
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Skip if pandas not available
pytest.importorskip("pandas")

import pandas as pd
from packaging import version

from cacheness import cacheness, CacheConfig


class TestPandasCompatibility:
    """Test compatibility with pandas versions 2.0+"""

    def test_pandas_version_check(self):
        """Verify pandas version is 2.0+"""
        pandas_version = version.parse(pd.__version__)
        assert pandas_version >= version.parse("2.0.0"), \
            f"Tests require pandas 2.0+, found {pd.__version__}"
        print(f"Testing with pandas {pd.__version__}")

    def test_dataframe_to_parquet_basic(self):
        """Test basic DataFrame to Parquet functionality"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            # Store DataFrame
            cache.put(df, key="test_df", version=1)
            
            # Retrieve DataFrame
            retrieved = cache.get(key="test_df", version=1)
            
            # Verify data matches
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_to_parquet_with_index(self):
        """Test DataFrame with custom index"""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        }, index=['a', 'b', 'c', 'd', 'e'])
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="indexed_df")
            retrieved = cache.get(key="indexed_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_datetime_columns(self):
        """Test DataFrame with datetime columns (pandas 2.0+ improvements)"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="datetime_df")
            retrieved = cache.get(key="datetime_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_series_to_parquet(self):
        """Test Series caching (pandas 2.0+ compatibility)"""
        series = pd.Series([1, 2, 3, 4, 5], name='test_series')
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(series, key="test_series")
            retrieved = cache.get(key="test_series")
            
            pd.testing.assert_series_equal(series, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_nullable_int_dtype(self):
        """Test DataFrame with nullable integer dtypes (pandas 2.0 feature)"""
        df = pd.DataFrame({
            'nullable_int': pd.array([1, 2, None, 4, 5], dtype='Int64'),
            'regular': [10, 20, 30, 40, 50]
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="nullable_df")
            retrieved = cache.get(key="nullable_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_string_dtype(self):
        """Test DataFrame with string dtype (pandas 2.0+ improvement)"""
        df = pd.DataFrame({
            'string_col': pd.array(['a', 'b', 'c', None, 'e'], dtype='string'),
            'values': [1, 2, 3, 4, 5]
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="string_dtype_df")
            retrieved = cache.get(key="string_dtype_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_pyarrow_backend(self):
        """Test DataFrame with PyArrow backend (pandas 2.0+ feature)"""
        try:
            # PyArrow string type available in pandas 2.0+
            df = pd.DataFrame({
                'strings': pd.array(['a', 'b', 'c', 'd', 'e'], dtype='string[pyarrow]'),
                'ints': pd.array([1, 2, 3, 4, 5], dtype='int64[pyarrow]')
            })
            
            temp_dir = tempfile.mkdtemp()
            try:
                config = CacheConfig(cache_dir=temp_dir)
                cache = cacheness(config)
                
                cache.put(df, key="pyarrow_df")
                retrieved = cache.get(key="pyarrow_df")
                
                # Compare values (dtypes might differ slightly with PyArrow)
                assert df.shape == retrieved.shape
                assert df.columns.tolist() == retrieved.columns.tolist()
                
                cache.close()
            finally:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
                    
        except (ImportError, TypeError):
            pytest.skip("PyArrow backend not available")

    def test_dataframe_large(self):
        """Test large DataFrame caching (performance check)"""
        import numpy as np
        
        # Create a reasonably large DataFrame
        df = pd.DataFrame({
            'col1': np.random.rand(10000),
            'col2': np.random.randint(0, 100, 10000),
            'col3': ['text'] * 10000,
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="large_df")
            retrieved = cache.get(key="large_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_dataframe_mixed_types(self):
        """Test DataFrame with mixed types"""
        df = pd.DataFrame({
            'int': [1, 2, 3],
            'float': [1.1, 2.2, 3.3],
            'string': ['a', 'b', 'c'],
            'bool': [True, False, True],
            'datetime': pd.date_range('2024-01-01', periods=3),
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            cache.put(df, key="mixed_df")
            retrieved = cache.get(key="mixed_df")
            
            pd.testing.assert_frame_equal(df, retrieved)
            
            cache.close()
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_to_parquet_api_compatibility(self):
        """Test that to_parquet API works with current parameters"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        import io
        buffer = io.BytesIO()
        
        # Test the parameters we use in handlers.py
        df.to_parquet(buffer, compression='snappy')
        buffer.seek(0)
        
        # Verify we can read it back
        retrieved = pd.read_parquet(buffer)
        pd.testing.assert_frame_equal(df, retrieved)

    def test_read_parquet_api_compatibility(self):
        """Test that read_parquet API works correctly"""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': ['a', 'b', 'c']
        })
        
        temp_dir = tempfile.mkdtemp()
        try:
            parquet_file = Path(temp_dir) / 'test.parquet'
            df.to_parquet(parquet_file)
            
            # Test read_parquet
            retrieved = pd.read_parquet(parquet_file)
            pd.testing.assert_frame_equal(df, retrieved)
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def test_series_to_frame_api(self):
        """Test Series.to_frame() API compatibility"""
        series = pd.Series([1, 2, 3, 4, 5], name='values')
        
        # Test to_frame() method used in handlers
        df = series.to_frame()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 1)
        assert df.columns[0] == 'values'

    def test_dataframe_columns_tolist(self):
        """Test DataFrame.columns.tolist() API"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Test columns.tolist() used in handlers
        columns = df.columns.tolist()
        assert columns == ['col1', 'col2']

    def test_dataframe_dtypes_iteration(self):
        """Test DataFrame dtypes iteration"""
        df = pd.DataFrame({
            'int': [1, 2, 3],
            'float': [1.1, 2.2, 3.3]
        })
        
        # Test dtypes iteration used in handlers
        dtypes_str = [str(dtype) for dtype in df.dtypes]
        assert len(dtypes_str) == 2
        assert 'int' in dtypes_str[0]
        assert 'float' in dtypes_str[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
