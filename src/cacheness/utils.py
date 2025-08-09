"""
Utility functions for cacheness
===============================

This module provides utility functions for optimized operations like
parallel file hashing for large directories.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

import xxhash

logger = logging.getLogger(__name__)


def _hash_single_file(file_info: Tuple[Path, Path]) -> Tuple[str, str]:
    """
    Hash a single file for multiprocessing.
    
    Args:
        file_info: Tuple of (file_path, base_directory_path)
        
    Returns:
        Tuple of (relative_path_string, content_hash)
    """
    file_path, base_path = file_info
    try:
        # Get relative path
        rel_path = str(file_path.relative_to(base_path))
        
        # Hash the file content
        hasher = xxhash.xxh3_64()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return rel_path, hasher.hexdigest()
    except OSError:
        # If we can't read the file, use its path and size as fallback
        try:
            size = file_path.stat().st_size
            return rel_path, f"unreadable:{size}"
        except OSError:
            return rel_path, "unreadable:unknown"


def hash_directory_parallel(directory_path: Path, max_workers: Optional[int] = None) -> str:
    """
    Hash a directory's contents using parallel processing for better performance.
    
    Args:
        directory_path: Path to the directory to hash
        max_workers: Maximum number of worker processes (defaults to CPU count)
        
    Returns:
        Hex string hash of the directory contents
    """
    if not directory_path.exists() or not directory_path.is_dir():
        return f"missing_directory:{str(directory_path)}"
    
    # Collect all files in the directory
    file_paths = []
    total_size = 0
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_paths.append((file_path, directory_path))
            try:
                total_size += file_path.stat().st_size
            except OSError:
                pass  # Skip files we can't stat
    
    if not file_paths:
        # Empty directory
        return xxhash.xxh3_64(f"empty_directory:{str(directory_path)}".encode()).hexdigest()
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(file_paths), 8)  # Cap at 8 processes
    
    # Smart threshold based on comprehensive benchmark results:
    # - Parallel processing is beneficial starting around 5GB total data OR 100+ files
    # - Conservative thresholds (4GB/80 files) to avoid the ~0.9s overhead when not beneficial
    # - Benchmark showed 1.24x average speedup when conditions are met
    total_size_mb = total_size / (1024 * 1024)
    use_parallel = (
        max_workers > 1 and
        (total_size_mb > 4000 or len(file_paths) > 80)
    )
    
    if not use_parallel:
        return _hash_directory_sequential(directory_path, file_paths)
    
    try:
        # Use ProcessPoolExecutor for better resource management
        file_hashes = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file hashing tasks
            future_to_file = {
                executor.submit(_hash_single_file, file_info): file_info
                for file_info in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    rel_path, content_hash = future.result()
                    file_hashes.append((rel_path, content_hash))
                except Exception as e:
                    file_info = future_to_file[future]
                    logger.warning(f"Failed to hash file {file_info[0]}: {e}")
                    # Add a fallback hash for failed files
                    rel_path = str(file_info[0].relative_to(file_info[1]))
                    file_hashes.append((rel_path, f"error:{str(e)[:50]}"))
        
        # Sort by relative path for consistent hashing
        file_hashes.sort(key=lambda x: x[0])
        
        # Create final hash from all file hashes
        final_hasher = xxhash.xxh3_64()
        for rel_path, content_hash in file_hashes:
            final_hasher.update(rel_path.encode())
            final_hasher.update(content_hash.encode())
        
        logger.debug(f"Hashed directory {directory_path} with {len(file_hashes)} files using {max_workers} workers")
        return final_hasher.hexdigest()
        
    except Exception as e:
        logger.warning(f"Parallel directory hashing failed for {directory_path}: {e}, falling back to sequential")
        return _hash_directory_sequential(directory_path, file_paths)


def _hash_directory_sequential(directory_path: Path, file_paths: List[Tuple[Path, Path]]) -> str:
    """
    Hash a directory sequentially (fallback method).
    
    Uses the same approach as parallel method: hash each file individually,
    then combine those hashes for consistency.
    
    Args:
        directory_path: Path to the directory
        file_paths: List of (file_path, base_path) tuples
        
    Returns:
        Hex string hash of the directory contents
    """
    # Hash each file individually, just like the parallel method
    file_hashes = []
    
    # Sort files by relative path for consistent ordering (same as parallel method)
    # First convert to (rel_path, file_path, base_path) tuples for sorting
    file_info_with_rel_paths = []
    for file_path, base_path in file_paths:
        rel_path = str(file_path.relative_to(base_path))
        file_info_with_rel_paths.append((rel_path, file_path, base_path))
    
    # Sort by relative path (same sorting key as parallel method)
    sorted_file_info = sorted(file_info_with_rel_paths, key=lambda x: x[0])
    
    for rel_path, file_path, _ in sorted_file_info:
        
        # Hash the file content
        try:
            file_hasher = xxhash.xxh3_64()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    file_hasher.update(chunk)
            content_hash = file_hasher.hexdigest()
        except OSError:
            # If we can't read the file, use its path and size as fallback
            try:
                size = file_path.stat().st_size
                content_hash = f"unreadable:{size}"
            except OSError:
                content_hash = "unreadable:unknown"
        
        file_hashes.append((rel_path, content_hash))
    
    # Create final hash from all file hashes (same as parallel method)
    final_hasher = xxhash.xxh3_64()
    for rel_path, content_hash in file_hashes:
        final_hasher.update(rel_path.encode())
        final_hasher.update(content_hash.encode())
    
    return final_hasher.hexdigest()


def hash_file_content(file_path: Path) -> str:
    """
    Hash a single file's content.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Hex string hash of the file content
    """
    if not file_path.exists():
        return f"missing_file:{str(file_path)}"
    
    if not file_path.is_file():
        return f"not_a_file:{str(file_path)}"
    
    try:
        hasher = xxhash.xxh3_64()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):  # Read in 8KB chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as e:
        logger.warning(f"Could not hash file content for {file_path}: {e}")
        return f"error_reading:{str(file_path)}:{str(e)}"
