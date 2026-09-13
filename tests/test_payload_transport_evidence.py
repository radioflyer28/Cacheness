"""Contracts for signed, immutable-payload-bound transport observations."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from cacheness.error_handling import CacheManifestIntegrityError
from cacheness.storage.transport_evidence import (
    PayloadTransportEvidence,
    PayloadTransportObservation,
)


_SIGNING_KEY = b"e" * 32
_IDENTITY = {
    "store_identity": "store-identity",
    "key": "logical-key",
    "generation": "generation-1",
    "locator": "generations/generation-1.native",
    "payload_sha256": "a" * 64,
    "payload_byte_size": 17,
}


def _issue(
    observation: PayloadTransportObservation | None = None,
) -> PayloadTransportEvidence:
    """Issue one independently known-good signed evidence value."""
    return PayloadTransportEvidence.issue(
        observation
        or PayloadTransportObservation(
            e_tag='"opaque-not-a-digest"', byte_size=17, version="version-1"
        ),
        signing_key=_SIGNING_KEY,
        **_IDENTITY,
    )


def test_signed_observation_verifies_only_for_its_exact_immutable_identity() -> None:
    """A remote observation is reusable only with the exact immutable generation."""
    evidence = _issue()

    assert evidence.verify(signing_key=_SIGNING_KEY, **_IDENTITY) == (
        PayloadTransportObservation(
            e_tag='"opaque-not-a-digest"', byte_size=17, version="version-1"
        )
    )

    with pytest.raises(CacheManifestIntegrityError):
        evidence.verify(signing_key=_SIGNING_KEY, **(_IDENTITY | {"generation": "other"}))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda record: record.__setitem__("e_tag", "different-opaque-observation"),
        lambda record: record.__setitem__("payload_byte_size", 18),
        lambda record: record.__setitem__("signature", "0" * 64),
    ),
)
def test_tampered_evidence_fails_closed(
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    """Altered signed content cannot be parsed as evidence for a payload."""
    raw_record = json.loads(_issue().canonical_bytes().decode("utf-8"))
    mutator(raw_record)

    with pytest.raises(CacheManifestIntegrityError):
        PayloadTransportEvidence.from_canonical_bytes(
            json.dumps(raw_record, separators=(",", ":")).encode("utf-8")
        ).verify(signing_key=_SIGNING_KEY, **_IDENTITY)


def test_opaque_etag_is_preserved_without_digest_interpretation() -> None:
    """An ETag remains an opaque observation even when it resembles a digest."""
    opaque = 'W/"0123456789abcdef-2"'
    evidence = _issue(PayloadTransportObservation(e_tag=opaque, byte_size=17, version=None))

    observation = evidence.verify(signing_key=_SIGNING_KEY, **_IDENTITY)

    assert observation.e_tag == opaque
    assert observation.e_tag != _IDENTITY["payload_sha256"]


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":2}',
        b'{"unknown":true}',
        b"{}",
    ),
)
def test_malformed_or_unknown_evidence_is_rejected_before_verification(raw: bytes) -> None:
    """The evidence parser accepts only the exact bounded current schema."""
    with pytest.raises(CacheManifestIntegrityError):
        PayloadTransportEvidence.from_canonical_bytes(raw)


def test_observation_rejects_oversized_text_and_invalid_size() -> None:
    """Untrusted metadata is bounded before it reaches authenticated encoding."""
    with pytest.raises(CacheManifestIntegrityError):
        PayloadTransportObservation(e_tag="x" * 4097, byte_size=17, version=None)

    with pytest.raises(CacheManifestIntegrityError):
        PayloadTransportObservation(e_tag=None, byte_size=-1, version=None)
