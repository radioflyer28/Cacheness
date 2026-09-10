"""Measure obstore staging memory and compare direct versus staged S3 publish."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import boto3
import psutil
from moto.server import ThreadedMotoServer
from obstore.store import LocalStore, MemoryStore, S3Store


CHUNK_SIZE = 5 * 1024 * 1024
MAX_CONCURRENCY = 2


def s3_store(endpoint: str, *, copy_mode: bool = False) -> S3Store:
    options: dict[str, Any] = {
        "region": "us-east-1",
        "endpoint": endpoint,
        "access_key_id": "test",
        "secret_access_key": "test",
        "virtual_hosted_style_request": False,
        "client_options": {"allow_http": True},
    }
    if copy_mode:
        options["copy_if_not_exists"] = "multipart"
    return S3Store("cacheness-streaming-spike", prefix="managed", **options)


def run_worker(kind: str, endpoint: str | None, payload_mib: int) -> None:
    payload_size = payload_mib * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="obstore-memory-worker-") as root_text:
        root = Path(root_text)
        stage = root / "handler-stage.bin"
        with stage.open("wb") as output:
            output.truncate(payload_size)

        if kind == "local-create":
            store: Any = LocalStore(root / "objects", mkdir=True)
        elif kind == "memory-create":
            store = MemoryStore()
        elif kind == "s3-create":
            assert endpoint is not None
            store = s3_store(endpoint)
        elif kind == "s3-staged-copy":
            assert endpoint is not None
            store = s3_store(endpoint, copy_mode=True)
        else:  # pragma: no cover
            raise ValueError(kind)

        print("READY", flush=True)
        if sys.stdin.readline().strip() != "GO":
            raise RuntimeError("worker start handshake failed")

        final = f"generations/{kind}-{payload_mib}mib.bin"
        started = time.monotonic()
        if kind == "s3-staged-copy":
            temporary = f"staging/{os.getpid()}.bin"
            store.put(
                temporary,
                stage,
                use_multipart=True,
                chunk_size=CHUNK_SIZE,
                max_concurrency=MAX_CONCURRENCY,
            )
            store.copy(temporary, final, overwrite=False)
            store.delete(temporary)
            publication = "multipart staging + conditional multipart copy"
        else:
            store.put(final, stage, mode="create")
            publication = "direct conditional put"
        upload_seconds = time.monotonic() - started

        response = store.get(final)
        downloaded = 0
        max_download_chunk = 0
        for chunk in response:
            downloaded += len(chunk)
            max_download_chunk = max(max_download_chunk, len(chunk))
        assert downloaded == payload_size
        print(
            json.dumps(
                {
                    "kind": kind,
                    "payload_bytes": payload_size,
                    "publication": publication,
                    "upload_seconds": round(upload_seconds, 4),
                    "downloaded_bytes": downloaded,
                    "max_download_chunk": max_download_chunk,
                }
            ),
            flush=True,
        )


def measure_worker(kind: str, endpoint: str | None, payload_mib: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--payload-mib",
        str(payload_mib),
    ]
    if endpoint is not None:
        command.extend(["--endpoint", endpoint])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    assert process.stderr is not None
    ready = process.stdout.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"worker failed before measurement: {ready} {process.stderr.read()}")
    observed = psutil.Process(process.pid)
    baseline = observed.memory_info().rss
    peak = baseline
    process.stdin.write("GO\n")
    process.stdin.flush()
    while process.poll() is None:
        try:
            peak = max(peak, observed.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.005)
    process.wait()
    remaining_stdout = process.stdout.read().strip().splitlines()
    stderr = process.stderr.read()
    if process.returncode != 0:
        raise RuntimeError(
            f"{kind} worker failed with {process.returncode}: "
            f"stdout={remaining_stdout!r} stderr={stderr}"
        )
    worker_result = json.loads(remaining_stdout[-1])
    worker_result.update(
        {
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "rss_growth_bytes": peak - baseline,
            "rss_growth_payload_ratio": round(
                (peak - baseline) / worker_result["payload_bytes"], 3
            ),
        }
    )
    return worker_result


@contextmanager
def mocked_s3() -> Iterator[str]:
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
    client.create_bucket(Bucket="cacheness-streaming-spike")
    try:
        yield endpoint
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=(
        "local-create", "memory-create", "s3-create", "s3-staged-copy"
    ))
    parser.add_argument("--endpoint")
    parser.add_argument("--payload-mib", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.worker:
        run_worker(arguments.worker, arguments.endpoint, arguments.payload_mib)
        return

    local = measure_worker("local-create", None, 64)
    memory = measure_worker("memory-create", None, 64)
    with mocked_s3() as endpoint:
        direct_small = measure_worker("s3-create", endpoint, 32)
        direct_large = measure_worker("s3-create", endpoint, 128)
        staged_small = measure_worker("s3-staged-copy", endpoint, 32)
        staged_large = measure_worker("s3-staged-copy", endpoint, 128)

    size_delta = direct_large["payload_bytes"] - direct_small["payload_bytes"]
    direct_growth_delta = (
        direct_large["rss_growth_bytes"] - direct_small["rss_growth_bytes"]
    )
    staged_growth_delta = (
        staged_large["rss_growth_bytes"] - staged_small["rss_growth_bytes"]
    )
    direct_is_payload_scaled = direct_growth_delta > size_delta * 0.75
    staged_is_bounded = staged_growth_delta < size_delta * 0.5
    participants = [
        local,
        memory,
        direct_small,
        direct_large,
        staged_small,
        staged_large,
    ]
    result = {
        "verdict": "PARTIAL",
        "participants": participants,
        "findings": {
            "direct_s3_create_materializes_payload_scale_memory": direct_is_payload_scaled,
            "staged_s3_copy_reduces_client_memory": staged_is_bounded,
            "direct_s3_growth_delta_bytes_32_to_128_mib": direct_growth_delta,
            "staged_s3_growth_delta_bytes_32_to_128_mib": staged_growth_delta,
            "memory_store_necessarily_retains_payload_in_process": True,
            "local_and_s3_downloads_are_chunk_iterated": all(
                item["max_download_chunk"] < item["payload_bytes"]
                for item in participants
                if item["kind"] != "memory-create"
            ),
            "memory_download_returns_whole_object": (
                memory["max_download_chunk"] == memory["payload_bytes"]
            ),
            "staged_copy_cost": (
                "additional temporary object, multipart cleanup exposure, and cleanup debt"
            ),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
