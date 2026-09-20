"""Stress obstore create-if-absent and deterministic recovery semantics."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import boto3
from moto.server import ThreadedMotoServer
from obstore.exceptions import AlreadyExistsError, NotFoundError, PreconditionError
from obstore.store import LocalStore, MemoryStore, S3Store


def event(category: str, **details: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "category": category,
        **details,
    }


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_object(store: Any, locator: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in store.get(locator):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def deterministic_locator(payload: bytes) -> str:
    return f"generations/{digest_bytes(payload)}.bin"


def reconcile(store: Any, locator: str, payload: bytes) -> str:
    try:
        observed_digest, observed_size = digest_object(store, locator)
    except (NotFoundError, FileNotFoundError):
        return "ABSENT"
    expected_digest = digest_bytes(payload)
    if observed_size == len(payload) and observed_digest == expected_digest:
        return "COMMITTED"
    return "CONFLICT"


def lost_response_put(store: Any, locator: str, payload: bytes) -> None:
    """Deterministically lose the response only after obstore returns success."""
    store.put(locator, payload, mode="create")
    raise ConnectionError("injected response loss after successful publication")


def collision_case(store_name: str, store: Any) -> tuple[dict[str, Any], list[dict]]:
    locator = "collisions/shared-locator.bin"
    contenders = [f"payload-from-writer-{index}".encode() for index in range(12)]
    barrier = threading.Barrier(len(contenders))
    log: list[dict[str, Any]] = []
    log_lock = threading.Lock()

    def write(index: int) -> tuple[str, int]:
        barrier.wait()
        try:
            store.put(locator, contenders[index], mode="create", use_multipart=False)
            outcome = "created"
        except (AlreadyExistsError, PreconditionError):
            outcome = "collision"
        with log_lock:
            log.append(event("publication", store=store_name, writer=index, outcome=outcome))
        return outcome, index

    with ThreadPoolExecutor(max_workers=len(contenders)) as pool:
        outcomes = list(pool.map(write, range(len(contenders))))
    winners = [index for outcome, index in outcomes if outcome == "created"]
    collisions = [index for outcome, index in outcomes if outcome == "collision"]
    assert len(winners) == 1, (store_name, outcomes)
    assert len(collisions) == len(contenders) - 1
    observed = bytes(store.get(locator).bytes())
    assert observed == contenders[winners[0]]
    return {
        "store": store_name,
        "writers": len(contenders),
        "winner": winners[0],
        "collisions": len(collisions),
        "complete_winner_bytes": True,
    }, log


def response_loss_case(store_name: str, store: Any) -> tuple[dict[str, Any], list[dict]]:
    payload = b"payload whose acknowledgement will be lost"
    locator = deterministic_locator(payload)
    log: list[dict[str, Any]] = []
    try:
        lost_response_put(store, locator, payload)
    except ConnectionError:
        log.append(event("response_lost", store=store_name, locator=locator))
    else:  # pragma: no cover - the injector always raises
        raise AssertionError("response-loss injector did not fire")

    first_recovery = reconcile(store, locator, payload)
    assert first_recovery == "COMMITTED"
    try:
        store.put(locator, payload, mode="create", use_multipart=False)
    except (AlreadyExistsError, PreconditionError):
        retry_outcome = "collision"
    else:  # pragma: no cover - create-if-absent must reject the retry
        retry_outcome = "unexpected overwrite"
    assert retry_outcome == "collision"
    assert reconcile(store, locator, payload) == "COMMITTED"
    assert reconcile(store, locator, b"different bytes") == "CONFLICT"
    absent_payload = b"never published"
    assert reconcile(store, deterministic_locator(absent_payload), absent_payload) == "ABSENT"
    log.append(event("reconciled", store=store_name, outcome=first_recovery))
    return {
        "store": store_name,
        "lost_response": True,
        "recovery": first_recovery,
        "retry": retry_outcome,
        "mismatch": "CONFLICT",
        "missing": "ABSENT",
    }, log


@contextmanager
def mocked_s3() -> Iterator[S3Store]:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    client.create_bucket(Bucket="cacheness-obstore-spike")
    try:
        yield S3Store(
            "cacheness-obstore-spike",
            prefix="managed",
            region="us-east-1",
            endpoint=endpoint,
            access_key_id="test",
            secret_access_key="test",
            virtual_hosted_style_request=False,
            client_options={"allow_http": True},
        )
    finally:
        server.stop()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="obstore-publication-spike-") as root:
        with mocked_s3() as s3:
            stores = (
                ("memory", MemoryStore()),
                ("local", LocalStore(Path(root) / "objects", mkdir=True)),
                ("mocked-s3", s3),
            )
            results: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []
            for name, store in stores:
                collision, collision_events = collision_case(name, store)
                recovery, recovery_events = response_loss_case(name, store)
                results.append({"collision": collision, "response_loss": recovery})
                events.extend(collision_events)
                events.extend(recovery_events)
    print(
        json.dumps(
            {
                "verdict": "VALIDATED",
                "summary": {
                    "stores": 3,
                    "contenders_per_store": 12,
                    "expected_collisions": 33,
                    "events": len(events),
                },
                "results": results,
                "events": events,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
