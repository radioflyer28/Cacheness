#!/usr/bin/env python3
"""
Custom Metadata — Airplane GPS Tracks
=======================================

Cache per-aircraft GPS track DataFrames and attach typed, queryable flight
metadata using a SQLAlchemy model.  Requires the ``sqlite`` metadata backend.

Usage:
    uv run python examples/custom_metadata_demo.py
"""

import sys
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
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


# -- Define a metadata schema for flights ------------------------------------
@custom_metadata_model("flights")
class FlightMeta(Base, CustomMetadataBase):
    __tablename__ = "custom_flights"

    tail_number  = Column(String(20),  nullable=False, unique=True, index=True)
    airline      = Column(String(100), nullable=False, index=True)
    aircraft_type = Column(String(50), nullable=False, index=True)
    origin       = Column(String(10),  nullable=False, index=True)
    destination  = Column(String(10),  nullable=False, index=True)
    altitude_max_ft = Column(Integer,  nullable=False, index=True)
    distance_nm  = Column(Float,       nullable=False, index=True)


def _make_gps_track(origin_lat, origin_lon, n_points=120, rng=None):
    """Generate a synthetic GPS track DataFrame."""
    if rng is None:
        rng = np.random.default_rng()
    t0 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(seconds=30 * i) for i in range(n_points)]
    lat = origin_lat + np.cumsum(rng.normal(0.01, 0.003, n_points))
    lon = origin_lon + np.cumsum(rng.normal(0.015, 0.004, n_points))
    alt = np.clip(
        np.concatenate([
            np.linspace(0, 35000, n_points // 3),
            np.full(n_points // 3, 35000) + rng.normal(0, 100, n_points // 3),
            np.linspace(35000, 0, n_points - 2 * (n_points // 3)),
        ]),
        0, 41000,
    )
    speed = np.clip(alt / 35000 * 480 + rng.normal(0, 10, n_points), 0, 600)
    return pd.DataFrame({
        "timestamp": timestamps,
        "latitude":  lat,
        "longitude": lon,
        "altitude_ft": alt.astype(int),
        "speed_kts": speed.astype(int),
    })


# -- Flight catalogue --------------------------------------------------------
FLIGHTS = [
    dict(
        tail_number="N12345",
        airline="United Airlines",
        aircraft_type="Boeing 737-800",
        origin="ORD",
        destination="LAX",
        altitude_max_ft=37000,
        distance_nm=1745,
        origin_lat=41.978,
        origin_lon=-87.904,
    ),
    dict(
        tail_number="N98765",
        airline="Delta Air Lines",
        aircraft_type="Airbus A321",
        origin="ATL",
        destination="JFK",
        altitude_max_ft=33000,
        distance_nm=762,
        origin_lat=33.640,
        origin_lon=-84.427,
    ),
    dict(
        tail_number="N55501",
        airline="Southwest Airlines",
        aircraft_type="Boeing 737 MAX 8",
        origin="DAL",
        destination="PHX",
        altitude_max_ft=39000,
        distance_nm=868,
        origin_lat=32.847,
        origin_lon=-96.851,
    ),
]

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as tmp:
        config = CacheConfig(
            cache_dir=tmp,
            metadata_backend="sqlite",
        )
        cache = cacheness(config)
        migrate_custom_metadata_tables()

        # -- Store GPS tracks with custom flight metadata ---------------------
        print("Storing GPS tracks:")
        for flight in FLIGHTS:
            track = _make_gps_track(flight["origin_lat"], flight["origin_lon"], rng=rng)
            meta = FlightMeta(
                tail_number=flight["tail_number"],
                airline=flight["airline"],
                aircraft_type=flight["aircraft_type"],
                origin=flight["origin"],
                destination=flight["destination"],
                altitude_max_ft=flight["altitude_max_ft"],
                distance_nm=flight["distance_nm"],
            )
            cache.put(track, tail=flight["tail_number"], custom_metadata=meta)
            print(
                f"  {flight['tail_number']:8s}  {flight['origin']} → {flight['destination']}"
                f"  {len(track)} pts  {flight['aircraft_type']}"
            )

        # -- Query: long-haul flights (> 1000 nm) ----------------------------
        print("\nLong-haul flights (> 1000 nm):")
        with cache.query_custom_session("flights") as q:
            for f in q.filter(FlightMeta.distance_nm > 1000).all():
                track = cache.get(cache_key=f.cache_key)
                print(
                    f"  {f.tail_number}  {f.origin} → {f.destination}"
                    f"  {f.distance_nm:.0f} nm"
                )
                print(track.describe())

        cache.close()
        print("\nDone.")
