#!/usr/bin/env python3
"""
File Storage Operations
========================

put_file / get_file on UnifiedCache and BlobStore:

  Copy-in   cache.put_file(path)                — store file, keep original
  Move-in   cache.put_file(path, move=True)      — store file, delete original
  Copy-out  cache.get_file(key, dest=dir)        — write to disk, keep in cache
  Move-out  cache.get_file(key, dest=dir, move)  — write to disk, remove from cache
  Safe      cache.get_file(key, dest, overwrite=False) — raise FileExistsError if dest exists

Usage:
    uv run python examples/file_operations_demo.py
"""

import shutil
import tempfile
from pathlib import Path

from cacheness import cacheness, CacheConfig
from cacheness.storage import BlobStore


def demo_unified_cache(tmp: Path):
    print("── UnifiedCache  put_file / get_file ────────────────────────────")

    cache_dir = tmp / "cache"
    cache = cacheness(CacheConfig(
        cache_dir=str(cache_dir),
        metadata_backend="sqlite",
        store_full_metadata=True,   # enables original_filename in metadata_dict
    ))
    src_dir = tmp / "src"
    src_dir.mkdir()
    out_dir = tmp / "out"
    out_dir.mkdir()

    # --- copy-in ---
    report = src_dir / "report.pdf"
    report.write_bytes(b"%PDF-1.4 fake pdf content")
    key = cache.put_file(report, name="report", version="jan")
    assert report.exists(), "original should still exist after copy-in"
    print(f"  copy-in  key={key[:16]}…  original still exists: {report.exists()}")

    # --- copy-out ---
    dest = cache.get_file(cache_key=key, dest=out_dir)
    assert dest is not None and dest.exists()
    print(f"  copy-out → {dest.name}  still in cache: {cache.exists(cache_key=key)}")

    # --- overwrite=False (dest already exists from copy-out) ---
    try:
        cache.get_file(cache_key=key, dest=dest, overwrite=False)
        print("  ERROR: expected FileExistsError")
    except FileExistsError:
        print(f"  overwrite=False → FileExistsError raised correctly ✓")

    # --- get as raw bytes (no dest) ---
    raw = cache.get_file(cache_key=key)
    assert isinstance(raw, bytes)
    print(f"  get bytes → {len(raw)} bytes")

    # --- move-in ---
    temp_csv = src_dir / "data.csv"
    temp_csv.write_text("a,b\n1,2\n")
    move_key = cache.put_file(temp_csv, name="data", version="extract", move=True)
    assert not temp_csv.exists(), "original should be deleted after move-in"
    print(f"  move-in  key={move_key[:16]}…  original deleted: {not temp_csv.exists()}")

    # --- move-out ---
    move_dest = cache.get_file(cache_key=move_key, dest=out_dir / "data.csv", move=True)
    assert move_dest is not None and move_dest.exists()
    assert not cache.exists(cache_key=move_key), "entry removed after move-out"
    print(f"  move-out → {move_dest.name}  removed from cache: {not cache.exists(cache_key=move_key)}")
    print()


def demo_blob_store(tmp: Path):
    print("── BlobStore  put_file / get_file ───────────────────────────────")

    store_dir = tmp / "store"
    src_dir = tmp / "blobsrc"
    src_dir.mkdir()
    out_dir = tmp / "blobout"
    out_dir.mkdir()

    with BlobStore(cache_dir=str(store_dir), backend="sqlite") as store:
        # copy-in
        model_file = src_dir / "model.onnx"
        model_file.write_bytes(b"ONNXMODELDATA" * 100)
        key = store.put_file(model_file, metadata={"version": "1.0"})
        print(f"  copy-in  key={key[:16]}…  original exists: {model_file.exists()}")

        # move-in
        tmp_weights = src_dir / "weights.bin"
        tmp_weights.write_bytes(b"\x00\xFF" * 512)
        wkey = store.put_file(tmp_weights, move=True)
        print(f"  move-in  original deleted: {not tmp_weights.exists()}")

        # copy-out (explicit dest path — BlobStore does not auto-resolve filenames)
        dest = store.get_file(key, dest=out_dir / "model.onnx")
        print(f"  copy-out → {dest.name}  still in store: {store.exists(key)}")

        # move-out
        wdest = store.get_file(wkey, dest=out_dir / "weights.bin", move=True)
        print(f"  move-out → {wdest.name}  removed: {not store.exists(wkey)}")
    print()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cacheness_file_ops_"))
    try:
        demo_unified_cache(tmp)
        demo_blob_store(tmp)
        print("Done.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
