#!/usr/bin/env python3
"""
Custom Metadata — Linked Tables
=================================

Demonstrates linking a plain SQLAlchemy lookup table (``Aircraft``) to a
``CustomMetadataBase`` table (``FlightMeta``) via a foreign key.

  Aircraft          — static aircraft facts (manufacturer, model, range, …)
  FlightMeta        — per-flight / per-cache-entry facts (routing, distance, …)
                      FK → Aircraft.tail_number
                      FK → cache_entries.cache_key  (via CustomMetadataBase)

This pattern lets you normalise reference data away from the per-entry
metadata table while still querying both tables in a single session.

Usage:
    uv run python examples/custom_metadata_linked_tables.py
"""

import sys
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from cacheness import CacheConfig, cacheness
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


# ---------------------------------------------------------------------------
# Aircraft lookup table — plain SQLAlchemy model, no CustomMetadataBase.
# Holds static facts that don't change per flight.
# ---------------------------------------------------------------------------
class Aircraft(Base):
    __tablename__ = "aircraft"

    tail_number = Column(String(20), primary_key=True)
    airline = Column(String(100), nullable=False, index=True)
    manufacturer = Column(String(50), nullable=False, index=True)
    model = Column(String(50), nullable=False, index=True)
    engine_count = Column(Integer, nullable=False)
    max_range_nm = Column(Integer, nullable=False)
    year_built = Column(Integer, nullable=False)

    # back-ref: aircraft_obj.flights → list of FlightMeta rows
    flights = relationship("FlightMeta", back_populates="aircraft")

    def __repr__(self):
        return f"<Aircraft({self.tail_number} {self.manufacturer} {self.model})>"


# ---------------------------------------------------------------------------
# FlightMeta — registered with Cacheness, carries the cache_key FK.
# Holds per-flight facts and a FK to Aircraft for the static details.
# ---------------------------------------------------------------------------
@custom_metadata_model("flights")
class FlightMeta(Base, CustomMetadataBase):
    __tablename__ = "custom_flights"

    tail_number = Column(
        String(20),
        ForeignKey("aircraft.tail_number"),
        nullable=False,
        index=True,
    )
    flight_number = Column(String(20), nullable=False, index=True)
    origin = Column(String(10), nullable=False, index=True)
    destination = Column(String(10), nullable=False, index=True)
    distance_nm = Column(Float, nullable=False, index=True)
    altitude_max_ft = Column(Integer, nullable=False, index=True)

    # ORM relationship — free join to Aircraft, no Cacheness involvement
    aircraft = relationship("Aircraft", back_populates="flights")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
AIRCRAFT_DATA = [
    dict(
        tail_number="N12345",
        airline="United Airlines",
        manufacturer="Boeing",
        model="737-800",
        engine_count=2,
        max_range_nm=2935,
        year_built=2008,
    ),
    dict(
        tail_number="N98765",
        airline="Delta Air Lines",
        manufacturer="Airbus",
        model="A321",
        engine_count=2,
        max_range_nm=3200,
        year_built=2015,
    ),
    dict(
        tail_number="N55501",
        airline="Southwest Airlines",
        manufacturer="Boeing",
        model="737 MAX 8",
        engine_count=2,
        max_range_nm=3550,
        year_built=2020,
    ),
]

FLIGHT_DATA = [
    dict(
        tail_number="N12345",
        flight_number="UA 110",
        origin="ORD",
        destination="LAX",
        distance_nm=1745,
        altitude_max_ft=37000,
        origin_lat=41.978,
        origin_lon=-87.904,
    ),
    dict(
        tail_number="N98765",
        flight_number="DL 402",
        origin="ATL",
        destination="JFK",
        distance_nm=762,
        altitude_max_ft=33000,
        origin_lat=33.640,
        origin_lon=-84.427,
    ),
    dict(
        tail_number="N55501",
        flight_number="WN 1823",
        origin="DAL",
        destination="PHX",
        distance_nm=868,
        altitude_max_ft=39000,
        origin_lat=32.847,
        origin_lon=-96.851,
    ),
]


def _make_gps_track(
    origin_lat: float, origin_lon: float, n_points: int = 120, rng=None
) -> pd.DataFrame:
    """Return a synthetic GPS track DataFrame."""
    if rng is None:
        rng = np.random.default_rng()
    t0 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(seconds=30 * i) for i in range(n_points)]
    lat = origin_lat + np.cumsum(rng.normal(0.01, 0.003, n_points))
    lon = origin_lon + np.cumsum(rng.normal(0.015, 0.004, n_points))
    thirds = n_points // 3
    alt = np.clip(
        np.concatenate(
            [
                np.linspace(0, 35000, thirds),
                np.full(thirds, 35000) + rng.normal(0, 100, thirds),
                np.linspace(35000, 0, n_points - 2 * thirds),
            ]
        ),
        0,
        41000,
    )
    speed = np.clip(alt / 35000 * 480 + rng.normal(0, 10, n_points), 0, 600)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "latitude": lat,
            "longitude": lon,
            "altitude_ft": alt.astype(int),
            "speed_kts": speed.astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    from pathlib import Path

    tmp = Path("./custom_meta_linked_cache")
    tmp.mkdir(exist_ok=True)

    try:
        config = CacheConfig(cache_dir=str(tmp), metadata_backend="sqlite")
        cache = cacheness(config)

        engine = cache.metadata_backend.engine

        # Create the plain Aircraft table manually (not in the custom_metadata registry)
        Base.metadata.create_all(engine, tables=[Aircraft.__table__])

        # Create FlightMeta table (registered via @custom_metadata_model)
        migrate_custom_metadata_tables()

        # -- Populate Aircraft lookup table ----------------------------------
        from sqlalchemy.orm import Session

        with Session(engine) as session:
            session.add_all([Aircraft(**a) for a in AIRCRAFT_DATA])
            session.commit()

        # -- Store GPS tracks with linked FlightMeta -------------------------
        print("Storing GPS tracks:")
        for flight in FLIGHT_DATA:
            track = _make_gps_track(flight["origin_lat"], flight["origin_lon"], rng=rng)
            meta = FlightMeta(
                tail_number=flight["tail_number"],
                flight_number=flight["flight_number"],
                origin=flight["origin"],
                destination=flight["destination"],
                distance_nm=flight["distance_nm"],
                altitude_max_ft=flight["altitude_max_ft"],
            )
            cache.put(track, flight=flight["flight_number"], custom_metadata=meta)
            print(
                f"  {flight['flight_number']:8s}  {flight['origin']} → {flight['destination']}"
                f"  {len(track)} pts"
            )

        # -- Query 1: join FlightMeta → Aircraft in one session --------------
        print("\nAll flights with aircraft details (joined):")
        with cache.query_custom_session("flights") as q:
            for f in q.all():
                ac = f.aircraft  # ORM relationship traversal — no extra query
                track = cache.get(cache_key=f.cache_key)
                print(
                    f"  {f.flight_number:8s}  {f.origin} → {f.destination}"
                    f"  {ac.manufacturer} {ac.model} ({ac.year_built})"
                    f"  —  track shape: {track.shape}"
                )

        # -- Query 2: long-haul flights on Boeing aircraft -------------------
        print("\nLong-haul Boeing flights (> 1000 nm):")
        with cache.query_custom_session("flights") as q:
            results = (
                q.join(FlightMeta.aircraft)
                .filter(FlightMeta.distance_nm > 1000)
                .filter(Aircraft.manufacturer == "Boeing")
                .all()
            )
            for f in results:
                print(
                    f"  {f.flight_number}  {f.tail_number}"
                    f"  {f.distance_nm:.0f} nm  {f.aircraft.model}"
                    f"  max range {f.aircraft.max_range_nm:,} nm"
                )

        # -- Query 3: Boeing fleet ordered by year ---------------------------
        print("\nBoeing fleet (newest first):")
        with cache.query_custom_session("flights") as q:
            results = (
                q.join(FlightMeta.aircraft)
                .filter(Aircraft.manufacturer == "Boeing")
                .order_by(Aircraft.year_built.desc())
                .all()
            )
            for f in results:
                print(
                    f"  {f.tail_number}  {f.aircraft.model}"
                    f"  built {f.aircraft.year_built}"
                    f"  range {f.aircraft.max_range_nm:,} nm"
                )

        cache.close()
        print("\nDone.")
    except Exception as ex:
        print(f"Error: {ex}")
    # finally:
    #     import shutil
    #     shutil.rmtree(tmp, ignore_errors=True)
