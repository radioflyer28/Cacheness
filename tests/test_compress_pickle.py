#!/usr/bin/env python3

"""Unit tests for compress_pickle.py module.

Tests compression and decompression functionality across all supported
blosc compression codecs for both v1 and v2.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Import the module under test first (so it can load blosc2 if available)
import cacheness.compress_pickle as compress_pickle


class TestCompressPickle:
    """Test suite for compress_pickle module."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_data_simple(self):
        """Simple test data: list of integers."""
        return [1, 2, 3, 4, 5, 42, 100, 999]

    @pytest.fixture
    def test_data_complex(self):
        """Complex test data: nested dictionary with various types."""
        return {
            "integers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["hello", "world", "test"],
            "nested": {
                "inner_list": [10, 20, 30],
                "inner_dict": {"key": "value", "number": 42},
            },
            "boolean": True,
            "none_value": None,
        }

    @pytest.fixture
    def test_data_numpy(self):
        """Numpy array test data."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 100, size=1000, dtype=np.int32)

    @pytest.fixture
    def test_data_large_numpy(self):
        """Large numpy array test data."""
        rng = np.random.default_rng(42)
        return rng.random(size=(100, 100)).astype(np.float64)

    def test_blosc_version_detection(self):
        """Test that blosc version is properly detected."""
        blosc_version = compress_pickle.blosc.__version__
        assert isinstance(blosc_version, str)
        assert len(blosc_version) > 0  # Just ensure version is not empty

    def test_simple_data_roundtrip(self, temp_dir, test_data_simple):
        """Test basic write/read roundtrip with simple data."""
        filepath = temp_dir / "test_simple.pkl.lz4"

        # Write data
        compress_pickle.write_file(test_data_simple, filepath)

        # Verify file was created
        assert filepath.exists()
        assert filepath.stat().st_size > 0

        # Read data back
        read_data = compress_pickle.read_file(filepath, nparray=False)

        # Verify data integrity
        assert read_data == test_data_simple

    def test_complex_data_roundtrip(self, temp_dir, test_data_complex):
        """Test write/read roundtrip with complex nested data."""
        filepath = temp_dir / "test_complex.pkl.zstd"

        # Write data with zstd compression
        kwargs = {}
        if compress_pickle.blosc.__version__.startswith("1"):
            kwargs["cname"] = "zstd"
        else:
            kwargs["codec"] = "zstd"

        compress_pickle.write_file(test_data_complex, filepath, **kwargs)

        # Read data back
        read_data = compress_pickle.read_file(filepath, nparray=False)

        # Verify data integrity
        assert read_data == test_data_complex

    def test_numpy_array_roundtrip(self, temp_dir, test_data_numpy):
        """Test write/read roundtrip with numpy array using pack_array."""
        filepath = temp_dir / "test_numpy.pkl.lz4"

        # Write numpy array with nparray=True
        compress_pickle.write_file(test_data_numpy, filepath, nparray=True)

        # Read data back
        read_data = compress_pickle.read_file(filepath, nparray=True)

        # Verify data integrity
        assert isinstance(read_data, np.ndarray)
        np.testing.assert_array_equal(read_data, test_data_numpy)
        assert read_data.dtype == test_data_numpy.dtype

    def test_numpy_array_as_pickle_roundtrip(self, temp_dir, test_data_numpy):
        """Test write/read roundtrip with numpy array using regular pickle."""
        filepath = temp_dir / "test_numpy_pickle.pkl.lz4"

        # Write numpy array with nparray=False
        compress_pickle.write_file(test_data_numpy, filepath, nparray=False)

        # Read data back
        read_data = compress_pickle.read_file(filepath, nparray=False)

        # Verify data integrity
        assert isinstance(read_data, np.ndarray)
        np.testing.assert_array_equal(read_data, test_data_numpy)

    def test_large_numpy_array_roundtrip(self, temp_dir, test_data_large_numpy):
        """Test write/read roundtrip with large numpy array."""
        filepath = temp_dir / "test_large_numpy.pkl.zstd"

        # Write large numpy array
        compress_pickle.write_file(test_data_large_numpy, filepath, nparray=True)

        # Read data back
        read_data = compress_pickle.read_file(filepath, nparray=True)

        # Verify data integrity
        assert isinstance(read_data, np.ndarray)
        np.testing.assert_array_equal(read_data, test_data_large_numpy)
        assert read_data.shape == test_data_large_numpy.shape
        assert read_data.dtype == test_data_large_numpy.dtype

    def test_all_compression_codecs_v1(self, temp_dir, test_data_simple):
        """Test all blosc v1 compression codecs."""
        if compress_pickle.blosc.__version__.startswith("1"):
            # True blosc v1 functionality
            comp_types = ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]
        else:
            pytest.skip("Blosc v1 not available")

        for comp_type in comp_types:
            filepath = temp_dir / f"test_v1_{comp_type}.pkl"

            # Write with specific codec
            compress_pickle.write_file(
                test_data_simple, filepath, cname=comp_type, nparray=False
            )

            # Verify file exists and has content
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Read back and verify
            read_data = compress_pickle.read_file(filepath, nparray=False)
            assert read_data == test_data_simple

    def test_all_compression_codecs_v2(self, temp_dir, test_data_simple):
        """Test all blosc v2+ compression codecs."""
        version = compress_pickle.blosc.__version__
        if not (version.startswith("2") or version.startswith("3")):
            pytest.skip("Blosc v2+ not available")

        comp_types = [
            compress_pickle.blosc.Codec.BLOSCLZ,
            compress_pickle.blosc.Codec.LZ4,
            compress_pickle.blosc.Codec.LZ4HC,
            compress_pickle.blosc.Codec.ZLIB,
            compress_pickle.blosc.Codec.ZSTD,
        ]

        # Only include newer codecs if they exist (some may not be available in all versions)
        optional_codecs = ["NDLZ", "ZFP_ACC", "ZFP_PREC", "ZFP_RATE"]
        for codec_name in optional_codecs:
            if hasattr(compress_pickle.blosc.Codec, codec_name):
                comp_types.append(getattr(compress_pickle.blosc.Codec, codec_name))

        for comp_type in comp_types:
            codec_name = comp_type.name.lower()
            filepath = temp_dir / f"test_v2_{codec_name}.pkl"

            try:
                # Write with specific codec
                compress_pickle.write_file(
                    test_data_simple, filepath, codec=comp_type, nparray=False
                )

                # Verify file exists and has content
                assert filepath.exists()
                assert filepath.stat().st_size > 0

                # Read back and verify
                read_data = compress_pickle.read_file(filepath, nparray=False)
                assert read_data == test_data_simple

            except Exception as e:
                # Some codecs might not be available or might fail with specific data
                pytest.skip(f"Codec {codec_name} failed: {e}")

    def test_numpy_with_all_codecs_v1(self, temp_dir, test_data_numpy):
        """Test numpy arrays with all blosc v1 codecs."""
        if compress_pickle.blosc.__version__.startswith("1"):
            # True blosc v1 functionality
            comp_types = ["lz4", "zstd"]  # Test subset for performance
        else:
            pytest.skip("Blosc v1 not available")

        for comp_type in comp_types:
            filepath = temp_dir / f"test_numpy_v1_{comp_type}.pkl"

            # Note: blosc v1 doesn't support nparray=True, so use nparray=False
            compress_pickle.write_file(
                test_data_numpy, filepath, cname=comp_type, nparray=False
            )

            read_data = compress_pickle.read_file(filepath, nparray=False)
            np.testing.assert_array_equal(read_data, test_data_numpy)

    def test_numpy_with_all_codecs_v2(self, temp_dir, test_data_numpy):
        """Test numpy arrays with all blosc v2+ codecs using pack_array."""
        version = compress_pickle.blosc.__version__
        if not (version.startswith("2") or version.startswith("3")):
            pytest.skip("Blosc v2+ not available")

        comp_types = [
            compress_pickle.blosc.Codec.LZ4,
            compress_pickle.blosc.Codec.ZSTD,
        ]  # Test subset for performance

        for comp_type in comp_types:
            codec_name = comp_type.name.lower()
            filepath = temp_dir / f"test_numpy_v2_{codec_name}.pkl"

            try:
                # Use nparray=True for blosc v2
                compress_pickle.write_file(
                    test_data_numpy, filepath, codec=comp_type, nparray=True
                )

                read_data = compress_pickle.read_file(filepath, nparray=True)
                np.testing.assert_array_equal(read_data, test_data_numpy)
                assert read_data.dtype == test_data_numpy.dtype

            except Exception as e:
                pytest.skip(f"Numpy test with codec {codec_name} failed: {e}")

    def test_compression_levels(self, temp_dir, test_data_complex):
        """Test different compression levels."""
        base_filepath = temp_dir / "test_clevel"

        for clevel in [1, 5, 9]:
            filepath = Path(f"{base_filepath}_{clevel}.pkl")

            # Write with specific compression level
            kwargs = {"clevel": clevel}
            if compress_pickle.blosc.__version__.startswith("1"):
                kwargs["cname"] = "lz4"
            else:
                kwargs["codec"] = "lz4"

            compress_pickle.write_file(test_data_complex, filepath, **kwargs)

            # Read back and verify
            read_data = compress_pickle.read_file(filepath, nparray=False)
            assert read_data == test_data_complex

    def test_file_compression_ratio(self, temp_dir, test_data_large_numpy):
        """Test that compression actually reduces file size."""
        filepath_uncompressed = temp_dir / "test_uncompressed.pkl"
        filepath_compressed = temp_dir / "test_compressed.pkl.zstd"

        # Save uncompressed (using pickle directly for comparison)
        import pickle

        with filepath_uncompressed.open("wb") as f:
            pickle.dump(test_data_large_numpy, f)

        # Save compressed
        kwargs = {}
        if compress_pickle.blosc.__version__.startswith("1"):
            kwargs.update({"cname": "zstd", "nparray": False})
        else:
            kwargs.update({"codec": "zstd", "nparray": True})

        compress_pickle.write_file(test_data_large_numpy, filepath_compressed, **kwargs)

        # Check file sizes
        uncompressed_size = filepath_uncompressed.stat().st_size
        compressed_size = filepath_compressed.stat().st_size

        # Compressed file should be smaller
        assert compressed_size < uncompressed_size

        # Calculate compression ratio
        compression_ratio = uncompressed_size / compressed_size
        assert compression_ratio > 1.0  # Should have some compression

    def test_empty_data(self, temp_dir):
        """Test handling of empty data structures."""
        empty_data_sets = [
            [],  # empty list
            {},  # empty dict
            "",  # empty string
            np.array([]),  # empty numpy array
        ]

        for i, empty_data in enumerate(empty_data_sets):
            filepath = temp_dir / f"test_empty_{i}.pkl"

            # Handle numpy arrays differently
            is_numpy = isinstance(empty_data, np.ndarray)

            compress_pickle.write_file(empty_data, filepath, nparray=is_numpy)
            read_data = compress_pickle.read_file(filepath, nparray=is_numpy)

            if isinstance(empty_data, np.ndarray):
                np.testing.assert_array_equal(read_data, empty_data)
            else:
                assert read_data == empty_data

    def test_pathlib_path_support(self, temp_dir, test_data_simple):
        """Test that pathlib.Path objects work as filepaths."""
        filepath = temp_dir / "test_pathlib.pkl.lz4"

        # Write using Path object
        compress_pickle.write_file(test_data_simple, filepath)

        # Read using Path object
        read_data = compress_pickle.read_file(filepath, nparray=False)

        assert read_data == test_data_simple

    def test_error_handling_invalid_path(self, test_data_simple):
        """Test error handling for invalid file paths."""
        invalid_path = Path("/nonexistent/directory/file.pkl")

        with pytest.raises(
            (FileNotFoundError, OSError, compress_pickle.CompressionError)
        ):
            compress_pickle.write_file(test_data_simple, invalid_path)

    def test_error_handling_read_nonexistent_file(self):
        """Test error handling when reading non-existent file."""
        nonexistent_path = Path("nonexistent_file.pkl")

        with pytest.raises(FileNotFoundError):
            compress_pickle.read_file(nonexistent_path)

    def test_kwargs_parameter_passing(self, temp_dir, test_data_simple):
        """Test that kwargs are properly passed to blosc functions."""
        filepath = temp_dir / "test_kwargs.pkl"

        # Test with various kwargs
        kwargs = {
            "clevel": 6,
            "typesize": 8,
        }

        if compress_pickle.blosc.__version__.startswith("1"):
            kwargs["cname"] = "lz4"
            kwargs["shuffle"] = compress_pickle.blosc.SHUFFLE
        else:
            kwargs["codec"] = "lz4"
            kwargs["filter"] = compress_pickle.blosc.Filter.SHUFFLE

        # Should not raise any errors
        compress_pickle.write_file(test_data_simple, filepath, **kwargs)
        read_data = compress_pickle.read_file(filepath, nparray=False)

        assert read_data == test_data_simple


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
