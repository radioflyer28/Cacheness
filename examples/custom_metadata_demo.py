#!/usr/bin/env python3
"""
Custom Metadata
================

Attach typed, queryable metadata to cache entries using SQLAlchemy models.
Requires the ``sqlite`` metadata backend.

Usage:
    uv run python examples/custom_metadata_demo.py
"""

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

if not is_custom_metadata_available():
    print("SQLAlchemy not installed — run: uv add sqlalchemy")
    sys.exit(1)


# -- Define a metadata schema ------------------------------------------------
@custom_metadata_model("experiments")
class ExperimentMeta(Base, CustomMetadataBase):
    __tablename__ = "custom_experiments"

    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False, index=True)
    created_by = Column(String(100), nullable=False, index=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        config = CacheConfig(
            cache_dir=tmp,
            metadata_backend="sqlite",
            store_full_metadata=True,
        )
        cache = cacheness(config)
        migrate_custom_metadata_tables()

        # -- Store entries with custom metadata --------------------------------
        experiments = [
            ("exp_001", "xgboost", 0.95, 100, "alice"),
            ("exp_002", "cnn", 0.88, 50, "bob"),
            ("exp_003", "random_forest", 0.92, 0, "alice"),
        ]

        for eid, mtype, acc, ep, user in experiments:
            meta = ExperimentMeta(
                experiment_id=eid,
                model_type=mtype,
                accuracy=acc,
                epochs=ep,
                created_by=user,
            )
            cache.put(
                np.random.random((10, 5)),
                experiment=eid,
                custom_metadata=meta,
            )
            print(f"Stored {eid}: {mtype} acc={acc}")

        # -- Query: high-accuracy models --------------------------------------
        print("\nHigh-accuracy experiments (>= 0.9):")
        with cache.query_custom_session("experiments") as q:
            for exp in q.filter(ExperimentMeta.accuracy >= 0.9).all():
                print(f"  {exp.experiment_id}: {exp.model_type} acc={exp.accuracy}")

        # -- Query: filter by author ------------------------------------------
        print("\nAlice's experiments:")
        with cache.query_custom_session("experiments") as q:
            for exp in q.filter(ExperimentMeta.created_by == "alice").all():
                print(f"  {exp.experiment_id}: {exp.model_type}")

        cache.close()
        print("\nDone.")
