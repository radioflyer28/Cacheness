#!/usr/bin/env python3

"""Functions to compress and decompress Python pickle files.
Uses blosc library for compression.
Benchmarks and tests commented out at bottom of code.

Benchmark result on FalconPC with NVME drive:
    Pickling 10,000,000 integers in list (about 20MB)

    19,554KB
    Un-compressed write time: 0.34
    Un-compressed read time: 0.52

    8,520KB
    LZMA (.xz) write time: 14.89
    LZMA (.xz) read time: 1.36

    11,867KB
    blosclz write time: 0.32 sec
    blosclz read time: 1.51 sec

    9,879KB
    lz4 write time: 0.25 sec
    lz4 read time: 1.44 sec

    9,865KB
    lz4hc write time: 0.36 sec
    lz4hc read time: 1.51 sec

    10,271KB
    snappy write time: 0.31 sec
    snappy read time: 1.54 sec

    8,308KB
    zlib write time: 0.38 sec
    zlib read time: 1.57 sec

    8,260KB
    zstd write time: 0.70 sec
    zstd read time: 1.54 sec
"""

# %%
__author__ = "Andrew Kriz"
__copyright__ = ""
__credits__ = [
    "https://github.com/limix/pickle-blosc/blob/master/pickle_blosc/_core.py"
]
__license__ = ""
__version__ = "2024.11.27"
__maintainer__ = "Andrew Kriz"
__email__ = "akriz@vt.edu"
__status__ = "production"

import numpy as np
import pickle
from pathlib import Path


class CompressionError(Exception):
    """Raised when compression fails."""
    pass


class DecompressionError(Exception):
    """Raised when decompression fails."""
    pass


try:
    import blosc2 as blosc
    shuffle_obj = blosc.Filter.SHUFFLE
    BLOSC_AVAILABLE = True
    blosc_version = blosc.__version__
except ImportError:
    try:
        import blosc
        shuffle_obj = blosc.SHUFFLE
        BLOSC_AVAILABLE = True
        blosc_version = blosc.__version__
    except ImportError:
        # Create mock blosc module to prevent import errors
        class MockBlosc:
            __version__ = "not available"
            SHUFFLE = 1
            MAX_BUFFERSIZE = 2**31 - 1
            
            class Filter:
                SHUFFLE = 1
            
            class Codec:
                LZ4 = "lz4"
                LZ4HC = "lz4hc"
                ZSTD = "zstd"
                ZLIB = "zlib"
                BLOSCLZ = "blosclz"
                NDLZ = "ndlz"
                ZFP_ACC = "zfp_acc"
                ZFP_PREC = "zfp_prec"
                ZFP_RATE = "zfp_rate"
                OPENHTJ2K = "openhtj2k"
                GROK = "grok"
            
            def pack_array(self, *args, **kwargs):
                raise ImportError("blosc2/blosc not available")
            
            def unpack_array(self, *args, **kwargs):
                raise ImportError("blosc2/blosc not available")
                
            def compress(self, *args, **kwargs):
                raise ImportError("blosc2/blosc not available")
                
            def decompress(self, *args, **kwargs):
                raise ImportError("blosc2/blosc not available")
        
        blosc = MockBlosc()
        shuffle_obj = blosc.Filter.SHUFFLE
        BLOSC_AVAILABLE = False
        blosc_version = "not available"

# Make blosc available through the module for tests
__all__ = ["blosc", "blosc_version", "get_compression_info", "list_available_codecs", "read_file", "write_file", 
           "get_recommended_settings", "benchmark_codecs", "CompressionError", "DecompressionError",
           "write_file_with_metadata", "read_file_with_metadata", "is_pickleable", "verify_pickleable"]
# print(f'blosc version: {blosc_version}')


def is_pickleable(obj) -> bool:
    """Check if an object can be pickled without actually pickling it.
    
    This function attempts to pickle the object and returns True if successful,
    False if it fails. It doesn't save the pickled data.
    
    Parameters
    ----------
    obj : object
        Any Python object to test for picklability
        
    Returns
    -------
    bool
        True if the object can be pickled, False otherwise
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def verify_pickleable(obj, raise_on_error: bool = True):
    """Verify that an object can be pickled and optionally raise detailed error.
    
    This function provides more detailed error information when an object
    cannot be pickled, which is useful for debugging.
    
    Parameters
    ----------
    obj : object
        Any Python object to verify for picklability
    raise_on_error : bool, default True
        If True, raises the original pickle exception on failure
        If False, returns a tuple (success: bool, error_message: str)
        
    Returns
    -------
    bool or tuple
        If raise_on_error=True: Returns True if successful, raises exception if not
        If raise_on_error=False: Returns (success: bool, error_message: str)
        
    Raises
    ------
    Exception
        The original pickle exception if raise_on_error=True and pickling fails
    """
    try:
        # Test pickling
        pickled_data = pickle.dumps(obj)
        
        # Test unpickling to ensure round-trip works
        pickle.loads(pickled_data)
        
        if raise_on_error:
            return True
        else:
            return True, ""
            
    except Exception as e:
        error_msg = f"Object cannot be pickled: {type(e).__name__}: {str(e)}"
        
        if raise_on_error:
            raise CompressionError(error_msg) from e
        else:
            return False, error_msg


def get_recommended_settings(data_type="general", priority="balanced"):
    """Get recommended compression settings based on data type and priority.
    
    Parameters
    ----------
    data_type : str
        Type of data: "general", "numpy", "text", "binary"
    priority : str  
        Priority: "speed", "compression", "balanced"
        
    Returns
    -------
    dict
        Recommended compression parameters
    """
    settings = {}
    
    if priority == "speed":
        settings = {"codec": "lz4", "clevel": 1}
    elif priority == "compression":
        settings = {"codec": "zstd", "clevel": 9}
    else:  # balanced
        settings = {"codec": "lz4", "clevel": 5}
    
    # Adjust for data type
    if data_type == "numpy":
        settings["nparray"] = True
    elif data_type == "text":
        settings["codec"] = "zstd"  # Better for text
    elif data_type == "binary":
        settings["codec"] = "lz4hc"  # Good for binary data
        
    return settings


def get_compression_info(filepath):
    """Get compression information about a file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the compressed file
        
    Returns
    -------
    dict
        Dictionary with file size, compression ratio estimate, etc.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size = filepath.stat().st_size
    return {
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "blosc_version": blosc_version,
    }


def list_available_codecs():
    """List all available compression codecs for the current blosc version.
    
    Returns
    -------
    list
        List of available codec names
    """
    if blosc_version.startswith("2"):
        return [codec.name.lower() for codec in blosc.Codec]
    return ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]


def benchmark_codecs(data, codecs=None, temp_dir=None):
    """Benchmark different codecs on sample data.
    
    Parameters
    ----------
    data : object
        Data to benchmark compression on
    codecs : list, optional
        List of codec names to test. If None, tests all available codecs.
    temp_dir : str or Path, optional
        Directory for temporary files. If None, uses system temp.
        
    Returns
    -------
    dict
        Benchmark results with compression ratios, speeds, etc.
    """
    import time
    import tempfile
    
    if codecs is None:
        codecs = list_available_codecs()[:5]  # Limit to main codecs for speed
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(exist_ok=True)
    
    # Determine if numpy array optimization should be used
    is_numpy = isinstance(data, np.ndarray)
    
    results = {}
    
    # Get uncompressed size for comparison
    uncompressed_data = pickle.dumps(data, -1)
    uncompressed_size = len(uncompressed_data)
    
    for codec in codecs:
        try:
            filepath = Path(temp_dir) / f"benchmark_{codec}.pkl"
            
            # Time compression
            start_time = time.perf_counter()
            write_file(data, filepath, codec=codec, nparray=is_numpy)
            write_time = time.perf_counter() - start_time
            
            # Get compressed file size
            compressed_size = filepath.stat().st_size
            compression_ratio = uncompressed_size / compressed_size
            
            # Time decompression
            start_time = time.perf_counter()
            read_data = read_file(filepath, nparray=is_numpy)
            read_time = time.perf_counter() - start_time
            
            # Verify data integrity
            if isinstance(data, np.ndarray):
                data_match = np.array_equal(data, read_data)
            else:
                data_match = data == read_data
            
            results[codec] = {
                "write_time": write_time,
                "read_time": read_time,
                "compression_ratio": compression_ratio,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "data_match": data_match,
            }
            
            # Clean up temp file
            filepath.unlink(missing_ok=True)
            
        except Exception as e:
            results[codec] = {"error": str(e)}
    
    return results


def write_file_with_metadata(obj, filepath, metadata=None, **kwargs):
    """Write file with optional metadata.
    
    Parameters
    ----------
    obj : object
        Data to compress and save
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Optional metadata to include
    **kwargs
        Additional arguments passed to write_file
        
    Returns
    -------
    None
    """
    import time
    
    data_to_write = {
        "data": obj,
        "metadata": metadata or {},
        "compress_pickle_version": __version__,
        "timestamp": time.time(),
        "blosc_version": blosc_version,
    }
    return write_file(data_to_write, filepath, **kwargs)


def read_file_with_metadata(filepath, **kwargs):
    """Read file with metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
    **kwargs
        Additional arguments passed to read_file
        
    Returns
    -------
    tuple
        (data, metadata_dict)
    """
    container = read_file(filepath, **kwargs)
    if isinstance(container, dict) and "data" in container:
        return container["data"], {
            "metadata": container.get("metadata", {}),
            "compress_pickle_version": container.get("compress_pickle_version"),
            "timestamp": container.get("timestamp"),
            "blosc_version": container.get("blosc_version"),
        }
    else:
        # File doesn't have metadata structure, return as-is
        return container, {}


# %%
def write_file(obj, filepath, *, nparray=True, **kwargs):
    """Pickle, compress, and save it to a file.

    Parameters
    ----------
    obj : object
        Any python object to be pickled.
    filepath : str
        File path destination (suggested to add .lz4 to end to indicate compression type)
    nparray : bool
        True, uses special blosc pack function for numpy arrays. It makes decompression of the resulting file much faster.
    kwargs : dict
        Blosc library options, such as cname, clevel, shuffle, etc.

    kwargs for Blosc v1:
        typesize : int
            Size of the data type in bytes.
        clevel : int
            Compression level.
        shuffle : int
            Shuffle filter.
        cname : str
            Compression method.

    kwargs for Blosc v2:
        typesize : int
            Size of the data type in bytes.
        clevel : int
            Compression level.
        shuffle : blosc2.Filter(Enum)
            Shuffle filter.
        codec : blosc2.Codec(Enum)
            Compression method.

    https://github.com/limix/pickle-blosc/blob/master/pickle_blosc/_core.py

    Raises
    ------
    CompressionError
        If compression fails
    ValueError
        If invalid parameters are provided
    """
    # Input validation
    if not filepath:
        raise ValueError("filepath cannot be empty")
    
    # Handle codec parameter for both Blosc v1 and v2/v3 compatibility
    if "codec" in kwargs and blosc_version.startswith("1"):
        # Convert codec to cname for Blosc v1 compatibility
        kwargs["cname"] = kwargs.pop("codec")
    elif "cname" in kwargs and (blosc_version.startswith("2") or blosc_version.startswith("3")):
        # Convert cname to codec for Blosc v2/v3 compatibility  
        kwargs["codec"] = kwargs.pop("cname")
    
    # Validate codec early for better error messages
    if (blosc_version.startswith("2") or blosc_version.startswith("3")) and "codec" in kwargs:
        codec = kwargs["codec"]
        if isinstance(codec, str):
            valid_codecs = ["lz4", "lz4hc", "zstd", "zlib", "blosclz", "ndlz", "zfp_acc", "zfp_prec", "zfp_rate", "openhtj2k", "grok"]
            if codec.lower() not in valid_codecs:
                raise ValueError(f"Unsupported codec: {codec}. Supported: {valid_codecs}")

    # typesize = kwargs.get('typesize', 8)
    # clevel = kwargs.get('clevel', 9)
    # shuffle = kwargs.get('shuffle', shuffle_obj)
    # cname = kwargs.get('cname', 'lz4')

    try:
        # check version of blosc library - blosc2 uses version 2.x or 3.x
        if blosc_version.startswith("2") or blosc_version.startswith("3"):
            itemsize = getattr(obj, "itemsize", 8)
            kwargs["typesize"] = kwargs.get("typesize", itemsize)
            kwargs["clevel"] = kwargs.get("clevel", 9)
            kwargs["filter"] = kwargs.get("filter", shuffle_obj)
            # Convert string codec to enum if needed
            codec = kwargs.get("codec", blosc.Codec.LZ4)
            if isinstance(codec, str):
                codec_map = {
                    "lz4": blosc.Codec.LZ4,
                    "lz4hc": blosc.Codec.LZ4HC,
                    "zstd": blosc.Codec.ZSTD,
                    "zlib": blosc.Codec.ZLIB,
                    "blosclz": blosc.Codec.BLOSCLZ,
                    "ndlz": blosc.Codec.NDLZ,
                    "zfp_acc": blosc.Codec.ZFP_ACC,
                    "zfp_prec": blosc.Codec.ZFP_PREC,
                    "zfp_rate": blosc.Codec.ZFP_RATE,
                    "openhtj2k": blosc.Codec.OPENHTJ2K,
                    "grok": blosc.Codec.GROK,
                }
                codec = codec_map.get(codec.lower(), blosc.Codec.LZ4)
            kwargs["codec"] = codec
            kwargs["_ignore_multiple_size"] = kwargs.get("_ignore_multiple_size", True)
        else:
            nparray = False
            itemsize = getattr(obj, "itemsize", 8)
            kwargs["typesize"] = kwargs.get("typesize", itemsize)
            kwargs["clevel"] = kwargs.get("clevel", 9)
            kwargs["shuffle"] = kwargs.get("shuffle", shuffle_obj)
            kwargs["cname"] = kwargs.get("cname", "lz4")

        if isinstance(obj, np.ndarray) and nparray:
            for key in ("typesize", "_ignore_multiple_size"):
                del kwargs[key]  # remove un-supported keys for pack_array
            with Path(filepath).open("wb") as f:
                f.write(blosc.pack_array(obj, **kwargs))
        else:
            arr = pickle.dumps(obj, -1)
            with Path(filepath).open("wb") as f:
                # Handle different parameter names for blosc.compress()
                compress_kwargs = {}
                if blosc_version.startswith("2") or blosc_version.startswith("3"):
                    # blosc2.compress() parameters
                    compress_kwargs["codec"] = kwargs.get("codec", blosc.Codec.LZ4)
                    compress_kwargs["clevel"] = kwargs.get("clevel", 9)
                    compress_kwargs["filter"] = kwargs.get("filter", shuffle_obj)
                    # Adjust typesize to ensure compatibility with data length
                    default_typesize = kwargs.get("typesize", 8)
                    # Find the largest typesize that divides evenly into the data length
                    data_len = len(arr)
                    if data_len > 0:
                        for ts in [default_typesize, 4, 2, 1]:
                            if data_len % ts == 0:
                                compress_kwargs["typesize"] = ts
                                break
                        else:
                            compress_kwargs["typesize"] = 1
                    else:
                        compress_kwargs["typesize"] = 1
                else:
                    # blosc v1 compress() parameters
                    compress_kwargs["cname"] = kwargs.get("cname", "lz4")
                    compress_kwargs["clevel"] = kwargs.get("clevel", 9)
                    compress_kwargs["shuffle"] = kwargs.get("shuffle", shuffle_obj)
                    # Adjust typesize to ensure compatibility with data length
                    default_typesize = kwargs.get("typesize", 8)
                    # Find the largest typesize that divides evenly into the data length
                    data_len = len(arr)
                    if data_len > 0:
                        for ts in [default_typesize, 4, 2, 1]:
                            if data_len % ts == 0:
                                compress_kwargs["typesize"] = ts
                                break
                        else:
                            compress_kwargs["typesize"] = 1
                    else:
                        compress_kwargs["typesize"] = 1
                
                start = 0
                while start < len(arr):
                    end = min(start + blosc.MAX_BUFFERSIZE, len(arr))
                    chunk = arr[start:end]
                    # Ensure typesize is compatible with chunk length
                    chunk_len = len(chunk)
                    if chunk_len > 0:
                        for ts in [compress_kwargs["typesize"], 4, 2, 1]:
                            if chunk_len % ts == 0:
                                compress_kwargs["typesize"] = ts
                                break
                        else:
                            compress_kwargs["typesize"] = 1
                    else:
                        compress_kwargs["typesize"] = 1
                    carr = blosc.compress(chunk, **compress_kwargs)
                    f.write(carr)
                    start = end
    except Exception as e:
        raise CompressionError(f"Failed to compress data: {e}") from e


def read_file(filepath, *, nparray=True):
    """Read, decompress, and unpickle a python object.

    Parameters
    ----------
    filepath : str
        File path to a pickled file.
    nparray : bool
        True, uses special blosc unpack function for numpy arrays.
    kwargs : dict

    Returns
    -------
    obj : object
        Unpickled object.

    https://github.com/limix/pickle-blosc/blob/master/pickle_blosc/_core.py

    """

    def decomp_byte_arr(f):
        buffsize = blosc.MAX_BUFFERSIZE
        while buffsize > 0:
            try:
                carr = f.read(buffsize)
            except (OverflowError, MemoryError):
                buffsize = buffsize // 2
                continue

            if len(carr) == 0:
                break

            yield blosc.decompress(carr)

        if buffsize == 0:
            raise RuntimeError("Could not determine a buffer size.")

    with Path(filepath).open("rb") as f:
        if nparray:
            return blosc.unpack_array(f.read())
        return pickle.loads(b"".join(decomp_byte_arr(f)))


# %% TEST AND BENCHMARK
# if __name__ == "__main__":
# import time
# import random
# import pickle
# import lzma
# import numpy as np

# #saves and reads from this file
# #adds extra extension for various compression types, example: test_pickle.pickle.xz
# filepath = 'blosc_benchmark.pickle'

# # %%
# #generate list of 10 million random numbers, about 10MB un-compressed
# # results = random.choices(range(0,101), k=int(1e7))
# # blosc_pack_numpy = True

# #generate two lists of a million random numbers
# list_len = 1e7
# rand_list = random.choices(range(0,101), k=int(round(list_len/2)))
# results = rand_list + rand_list
# blosc_pack_numpy = True

# #generate numpy array of 10 million random numbers, about 80MB un-compressed
# # results = np.random.randint(0, 100, int(1e7))
# # blosc_pack_numpy = True

# print(f'{len(results)} integers in list')

# bench_results = {}

# #test no compression
# start_write = time.perf_counter()
# with open(filepath, 'wb') as outfile:
#     pickle.dump(results, outfile)
# end_write = time.perf_counter()
# write_time = end_write - start_write
# print(f'Un-compressed write time: {write_time:.4f}')

# #get size of un-compressed file in MiB
# uncomp_file_size = Path(filepath).stat().st_size / 1024**2
# print(f'Un-compressed file size: {uncomp_file_size}MiB')

# start_read = time.perf_counter()
# with open(filepath, 'rb') as infile:
#     read_results = pickle.load(infile)
# end_read = time.perf_counter()
# read_time = end_read - start_read
# print(f'Un-compressed read time: {read_time:.4f}')

# bench_results['uncompressed'] = {'write_time': write_time, 'read_time': read_time, 'compression_ratio': 1}

# #test LZMA compression (smallest size)
# # start_write = time.perf_counter()
# # with lzma.open(filepath+'.xz', 'wb') as pkl_file:
# #     pickle.dump(results, pkl_file)
# # end_write = time.perf_counter()
# # print(f'LZMA (.xz) write time: {(end_write-start_write):.4f}')

# # start_read = time.perf_counter()
# # with lzma.open(filepath+'.xz', 'rb') as pkl_file:
# #     read_results = pickle.load(pkl_file)
# # end_read = time.perf_counter()
# # print(f'LZMA (.xz) read time: {(end_read-start_read):.4f}')

# #test all blosc library compression types
# if blosc_version.startswith('1'):
#     comp_types = ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
# if blosc_version.startswith('2'):
#     comp_types = [
#         blosc.Codec.BLOSCLZ,
#         blosc.Codec.LZ4,
#         blosc.Codec.LZ4HC,
#         blosc.Codec.ZLIB,
#         blosc.Codec.ZSTD,
#         blosc.Codec.NDLZ,
#         blosc.Codec.ZFP_ACC,
#         blosc.Codec.ZFP_PREC,
#         blosc.Codec.ZFP_RATE,
#     ]
# for comp_type in comp_types:
#     try:
#         if blosc_version.startswith('1'):
#             blosc_opts = dict(cname=comp_type)
#             comp_str = comp_type
#         if blosc_version.startswith('2'):
#             blosc_opts = dict(codec=comp_type)
#             comp_str = comp_type.name.lower()

#         start_write = time.perf_counter()
#         write_file(results, filepath+'.'+comp_str, nparray=blosc_pack_numpy, **blosc_opts)
#         end_write = time.perf_counter()
#         write_time = end_write - start_write
#         print(f'{comp_str} write time: {write_time:.4f} sec')
#     except Exception as e:
#         print(f'Write of {comp_str} failed: {e}')
#         continue

#     try:
#         #compute compression ratio compared to un-compressed file
#         comp_file_size = Path(filepath+'.'+comp_str).stat().st_size / 1024**2
#         compression_ratio = 1 / (comp_file_size / uncomp_file_size)
#         print(f'compression ratio: {compression_ratio:.2f}')
#     except Exception as e:
#         print(f'Error computing compression ratio: {e}')
#         continue

#     try:
#         start_read = time.perf_counter()
#         read_results = read_file(filepath+'.'+comp_str)
#         end_read = time.perf_counter()
#         read_time = end_read - start_read
#         print(f'{comp_str} read time: {read_time:.4f} sec')
#     except Exception as e:
#         print(f'Read of {comp_str} failed: {e}')
#         continue

#     #check if the read results match the original results
#     try:
#         if isinstance(results, np.ndarray):
#             if not np.array_equal(results, read_results):
#                 print(f'{comp_str} results do not match original results')
#                 continue
#         elif results != read_results:
#             print(f'{comp_str} results do not match original results')
#             continue
#     except Exception as e:
#         print(f'Error comparing {comp_str} results: {e}')
#         continue

#     bench_results[comp_str] = {'write_time': write_time, 'read_time': read_time, 'compression_ratio': compression_ratio}

# import json
# print(json.dumps(bench_results, indent=4))  #pretty print results

# %%
