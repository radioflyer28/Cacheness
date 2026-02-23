#!/usr/bin/env python3
"""
Convenience Metadata Helpers
==============================

put_with_meta / get_with_meta — store kwargs as *both* cache key and
metadata_dict in one call, eliminating the redundant double-pass pattern.

put_with_model / get_with_model — same idea, but stores a SQLAlchemy ORM
instance alongside the data so you can query it later with query_custom().

The ``on=`` discriminator lets you create distinct cache entries that share
identical metadata (e.g. the same experiment config but different epochs).

Usage:
    uv run python examples/convenience_metadata_helpers.py
"""

import shutil
import sys
import tempfile

import numpy as np
from sqlalchemy import Column, Float, Integer, String

from cacheness import cacheness, CacheConfig
from cacheness.custom_metadata import (
    CustomMetadataBase,
    custom_metadata_model,
    is_custom_metadata_available,
    migrate_custom_metadata_tables,
)
from cacheness.metadata import Base


# -- 1. put_with_meta / get_with_meta ----------------------------------------

def demo_put_get_with_meta(cache):
    print("── put_with_meta / get_with_meta ────────────────────────────────")

    data = np.random.rand(100, 10)
    # All kwargs become BOTH the cache key AND metadata_dict entries
    cache.put_with_meta(data, experiment="exp_001", model="xgboost", accuracy=0.94)
    print("  stored: experiment='exp_001', model='xgboost', accuracy=0.94")

    # Retrieve with the same kwargs (they re-derive the same cache key)
    result = cache.get_with_meta(experiment="exp_001", model="xgboost", accuracy=0.94)
    if result:
        loaded, meta = result
        print(f"  loaded shape : {loaded.shape}")
        print(f"  meta         : {meta}")
    else:
        print("  miss (unexpected)")

    # Miss returns None
    assert cache.get_with_meta(experiment="exp_001", model="lightgbm", accuracy=0.94) is None
    print("  non-existent key → None ✓\n")


# -- 2. on= key discriminator ------------------------------------------------

def demo_on_discriminator(cache):
    print("── on= key discriminator ─────────────────────────────────────────")

    shared_meta = dict(experiment="exp_001", model="xgboost")
    for epoch in (5, 10, 20):
        weights = np.random.rand(50)
        cache.put_with_meta(weights, on={"epoch": epoch}, **shared_meta)
        print(f"  stored epoch={epoch}")

    # Each epoch is a distinct cache entry, but metadata is shared
    for epoch in (5, 10, 20):
        result = cache.get_with_meta(on={"epoch": epoch}, **shared_meta)
        if result:
            data, meta = result
            print(f"  epoch={epoch} → shape={data.shape}, meta={meta}")
    print()


# -- 3. query_with_meta -------------------------------------------------------

def demo_query_with_meta(cache):
    print("── query_with_meta ───────────────────────────────────────────────")
    hits = list(cache.query_with_meta(experiment="exp_001", model="xgboost"))
    print(f"  found {len(hits)} entries for experiment='exp_001', model='xgboost'")
    for data, meta in hits:
        print(f"    shape={data.shape}  meta={meta}")
    print()


# -- 4. put_with_model / get_with_model (requires SQLAlchemy) ----------------

def demo_put_get_with_model(cache):
    print("── put_with_model / get_with_model ───────────────────────────────")

    if not is_custom_metadata_available():
        print("  skipped (sqlalchemy not installed)\n")
        return

    @custom_metadata_model("ml_runs")
    class RunMeta(Base, CustomMetadataBase):
        __tablename__ = "custom_ml_runs"
        run_id     = Column(String(50), nullable=False, index=True)
        model_type = Column(String(50), nullable=False, index=True)
        accuracy   = Column(Float,      nullable=False)
        n_samples  = Column(Integer,    nullable=False)

    migrate_custom_metadata_tables(cache)

    data = np.random.rand(200, 20)
    cache.put_with_model(
        data, RunMeta,
        run_id="run_42",
        model_type="random_forest",
        accuracy=0.91,
        n_samples=200,
    )
    print("  stored with ORM model")

    result = cache.get_with_model(
        RunMeta,
        run_id="run_42",
        model_type="random_forest",
        accuracy=0.91,
        n_samples=200,
    )
    if result:
        loaded, orm_instance = result
        print(f"  loaded shape : {loaded.shape}")
        print(f"  orm.accuracy : {orm_instance.accuracy}")
        print(f"  orm.run_id   : {orm_instance.run_id}")
    else:
        print("  miss (unexpected)")
    print()


def main():
    tmp = tempfile.mkdtemp(prefix="cacheness_conv_meta_")
    try:
        cache = cacheness(CacheConfig(
            cache_dir=tmp,
            metadata_backend="sqlite",
            store_full_metadata=True,
        ))

        demo_put_get_with_meta(cache)
        demo_on_discriminator(cache)
        demo_query_with_meta(cache)
        demo_put_get_with_model(cache)

        print("Done.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
