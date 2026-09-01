"""Small admission primitives for bounded BlobStore lifecycle transitions."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
import ctypes
from pathlib import Path
from threading import Condition, Lock, get_ident
import time
from typing import BinaryIO, Callable, ClassVar, Iterator, Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobLockReleaseError,
    CacheBlobStoreClosedError,
)

from .path_security import ManagedFileOps, resolve_managed_locator


_INTERPROCESS_LOCK_STRIPES = 64


class _WindowsLockApi(Protocol):
    """Minimal Win32 byte-range lock surface kept injectable for tests."""

    def lock(self, file_descriptor: int, *, exclusive: bool) -> object:
        """Acquire one blocking shared or exclusive whole-file lock."""

    def unlock(self, file_descriptor: int, token: object) -> object:
        """Release the matching whole-file lock."""


class _NativeWindowsLockApi:
    """Use ``LockFileEx`` so Windows retains shared admission semantics."""

    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _MAX_DWORD = 0xFFFFFFFF

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", ctypes.c_uint32),
            ("OffsetHigh", ctypes.c_uint32),
            ("Pointer", ctypes.c_void_p),
        ]

    def __init__(self) -> None:
        try:
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except (AttributeError, ImportError, OSError) as exc:  # pragma: no cover - Windows only.
            raise CacheBlobBackendError(
                "BlobStore lifecycle locking is unavailable on this Windows runtime",
                context={"operation": "lifecycle_lock"},
            ) from exc

        self._get_osfhandle = msvcrt.get_osfhandle
        self._lock_file_ex = kernel32.LockFileEx
        self._lock_file_ex.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._Overlapped),
        )
        self._lock_file_ex.restype = ctypes.c_int
        self._unlock_file_ex = kernel32.UnlockFileEx
        self._unlock_file_ex.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._Overlapped),
        )
        self._unlock_file_ex.restype = ctypes.c_int

    def lock(self, file_descriptor: int, *, exclusive: bool) -> object:
        """Acquire a blocking whole-file lock without downgrading shared callers."""
        flags = self._LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0
        overlapped = self._Overlapped()
        handle = ctypes.c_void_p(self._get_osfhandle(file_descriptor))
        if not self._lock_file_ex(
            handle,
            flags,
            0,
            self._MAX_DWORD,
            self._MAX_DWORD,
            ctypes.byref(overlapped),
        ):
            raise OSError(ctypes.get_last_error(), "LockFileEx failed")
        return overlapped

    def unlock(self, file_descriptor: int, token: object) -> object:
        """Release exactly the byte range acquired by :meth:`lock`."""
        if not isinstance(token, self._Overlapped):
            raise TypeError("Windows lifecycle lock token is invalid")
        handle = ctypes.c_void_p(self._get_osfhandle(file_descriptor))
        if not self._unlock_file_ex(
            handle,
            0,
            self._MAX_DWORD,
            self._MAX_DWORD,
            ctypes.byref(token),
        ):
            raise OSError(ctypes.get_last_error(), "UnlockFileEx failed")
        return None


def _platform_name() -> str:
    """Resolve the runtime lock topology through a narrow test seam."""
    import os

    return os.name


def _windows_lock_api() -> _WindowsLockApi:
    """Construct the production Win32 lock adapter only on Windows."""
    return _NativeWindowsLockApi()


def lock_stripe_index(root: Path, identity: str) -> int:
    """Map one root-scoped transition identity into a fixed lock-file set."""
    import hashlib

    digest = hashlib.sha256(f"{root}\x00{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _INTERPROCESS_LOCK_STRIPES


@contextmanager
def interprocess_open_file_lock(
    handle: BinaryIO,
    *,
    exclusive: bool,
    operation: str,
    close_handle: bool = False,
) -> Iterator[None]:
    """Lock one already-open managed descriptor on every supported OS.

    POSIX uses ``flock`` for shared/exclusive admission. Windows uses Win32
    ``LockFileEx`` over the same durable lock file, preserving shared ordinary
    admission instead of silently turning all normal operations into a global
    exclusive mutex. ``close_handle`` is reserved for short-lived lock
    descriptors; authority owners retain their descriptor for its whole
    lifetime and pass the default.
    """
    unlock: Callable[[], object] | None = None
    try:
        if _platform_name() == "nt":
            api = _windows_lock_api()
            token = api.lock(handle.fileno(), exclusive=exclusive)

            def unlock_windows_lock() -> object:
                return api.unlock(handle.fileno(), token)

            unlock = unlock_windows_lock
        else:
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - exotic non-POSIX runtime.
                raise CacheBlobBackendError(
                    "BlobStore lifecycle locking requires POSIX flock or Win32 LockFileEx",
                    context={"operation": operation},
                ) from exc
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

            def unlock_posix_lock() -> object:
                return fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            unlock = unlock_posix_lock
    except CacheBlobBackendError:
        if close_handle:
            try:
                handle.close()
            except OSError:
                pass
        raise
    except OSError as exc:
        if close_handle:
            try:
                handle.close()
            except OSError:
                pass
        raise CacheBlobBackendError(
            "BlobStore lifecycle lock could not be acquired",
            context={"operation": operation},
        ) from exc

    body_failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_failure = exc
        raise
    finally:
        release_failure: OSError | None = None
        try:
            if unlock is not None:
                unlock()
        except OSError as exc:
            release_failure = exc
        finally:
            try:
                if close_handle:
                    handle.close()
            except OSError as exc:
                if release_failure is None:
                    release_failure = exc
        if release_failure is not None and body_failure is None:
            raise CacheBlobLockReleaseError(
                "BlobStore lifecycle lock could not be released",
                context={"operation": operation},
            ) from release_failure


@contextmanager
def interprocess_file_lock(
    file_ops: ManagedFileOps,
    locator: Path,
    *,
    exclusive: bool,
    operation: str,
) -> Iterator[None]:
    """Create and lock one short-lived contained descriptor.

    Persistent authority owners should create their lock file and retain a
    managed descriptor themselves, then call :func:`interprocess_open_file_lock`.
    The short-lived form is appropriate for striped operation evidence and
    admission transitions.
    """
    try:
        file_ops.create_bytes_durable_exclusive(locator, b"lock\n")
    except FileExistsError:
        pass
    except OSError as exc:
        raise CacheBlobBackendError(
            "BlobStore lifecycle lock could not be acquired",
            context={"operation": operation},
        ) from exc

    try:
        handle = file_ops.open_read(locator)
    except OSError as exc:
        raise CacheBlobBackendError(
            "BlobStore lifecycle lock could not be acquired",
            context={"operation": operation},
        ) from exc
    with interprocess_open_file_lock(
        handle,
        exclusive=exclusive,
        operation=operation,
        close_handle=True,
    ):
        yield


class StoreAdmissionBarrier:
    """Coordinate one finite clear-snapshot admission boundary across processes.

    Ordinary operations share admission and remain concurrent with one another.
    A clear obtains aggregate admission long enough to persist its complete,
    authenticated target inventory, then releases it before reclaiming payloads.
    A root-scoped POSIX advisory lock extends that reader/writer boundary to
    independent ``BlobStore`` processes which share the local metadata store.
    Per-key manifest CAS remains the authority boundary for every individual
    generation transition.
    """

    _instances_guard = Lock()
    _instances: dict[tuple[int, int], "StoreAdmissionBarrier"] = {}
    _before_registry_insert: ClassVar[
        Callable[["StoreAdmissionBarrier"], None] | None
    ] = None

    def __init__(self, root: Path) -> None:
        self._condition = Condition(Lock())
        self._aggregate_active = False
        self._ordinary_active = 0
        # The barrier owns this separate descriptor boundary.  A BlobStore
        # instance can close while another same-root instance remains active;
        # retaining an individual instance's ManagedFileOps would make the
        # shared barrier unusable after that close.
        self._file_ops = ManagedFileOps(root)
        self._root_identity = self._file_ops.root_identity
        self._registry_identity: tuple[int, int] | None = None
        self._leases = 0
        self._lock_locator = resolve_managed_locator(
            self._file_ops.root,
            ".cacheness-lifecycle-admission.lock",
            operation="lifecycle_admission_lock",
            allow_missing_leaf=True,
        )

    @staticmethod
    def _identity_for(root: Path) -> tuple[int, int]:
        """Use filesystem identity so a recreated pathname receives a new barrier."""
        stat = root.stat()
        return stat.st_dev, stat.st_ino

    @classmethod
    def acquire(cls, root: Path) -> "StoreAdmissionBarrier":
        """Lease the root's shared barrier until its owning store closes."""
        candidate = cls(root)
        if cls._before_registry_insert is not None:
            cls._before_registry_insert(candidate)
        identity = candidate._root_identity
        with cls._instances_guard:
            barrier = cls._instances.get(identity)
            if barrier is None:
                candidate._registry_identity = identity
                candidate._leases = 1
                cls._instances[identity] = candidate
                return candidate
            barrier._leases += 1
        candidate._file_ops.close()
        return barrier

    @classmethod
    def for_root(cls, root: Path) -> "StoreAdmissionBarrier":
        """Compatibility constructor for direct callers outside BlobStore ownership.

        Production ``BlobStore`` instances use :meth:`acquire` and pair it with
        :meth:`release`; this helper retains the prior shared-lookup behavior.
        """
        candidate = cls(root)
        if cls._before_registry_insert is not None:
            cls._before_registry_insert(candidate)
        identity = candidate._root_identity
        with cls._instances_guard:
            barrier = cls._instances.get(identity)
            if barrier is None:
                candidate._registry_identity = identity
                cls._instances[identity] = candidate
                return candidate
        candidate._file_ops.close()
        return barrier

    def release(self) -> None:
        """Release one owning-store lease and close the final root descriptor."""
        close_file_ops = False
        with self._instances_guard:
            registry_identity = self._registry_identity
            if registry_identity is None:
                return
            current = self._instances.get(registry_identity)
            if current is not self:
                return
            if self._leases <= 0:
                return
            self._leases -= 1
            if self._leases == 0:
                del self._instances[registry_identity]
                close_file_ops = True
        if close_file_ops:
            self._file_ops.close()

    @contextmanager
    def _advisory_admission(self, *, exclusive: bool) -> Iterator[None]:
        """Hold the root-wide shared/exclusive lock for one admitted operation."""
        with interprocess_file_lock(
            self._file_ops,
            self._lock_locator,
            exclusive=exclusive,
            operation="lifecycle_admission",
        ):
            yield

    @contextmanager
    def ordinary_admission(self) -> Iterator[None]:
        """Admit one normal operation unless a snapshot is being established."""
        with self._condition:
            while self._aggregate_active:
                self._condition.wait()
            self._ordinary_active += 1
        try:
            with self._advisory_admission(exclusive=False):
                yield
        finally:
            with self._condition:
                self._ordinary_active -= 1
                if self._ordinary_active == 0:
                    self._condition.notify_all()

    @contextmanager
    def aggregate_admission(self) -> Iterator[None]:
        """Exclude ordinary work only while creating a finite clear snapshot."""
        with self._condition:
            while self._aggregate_active:
                self._condition.wait()
            self._aggregate_active = True
            while self._ordinary_active:
                self._condition.wait()
        try:
            with self._advisory_admission(exclusive=True):
                yield
        finally:
            with self._condition:
                self._aggregate_active = False
                self._condition.notify_all()


@dataclass
class _KeyCoordinatorEntry:
    """One exact physical key's lock and in-flight acquisition count."""

    lock: Lock = field(default_factory=Lock)
    users: int = 0


class KeyCoordinatorRegistry:
    """Coordinate same-key local operations without serializing unrelated keys.

    The registry is deliberately owned by one ``BlobStore`` instance.  It
    supplies a cheap, deterministic in-process ordering boundary, while the
    manifest repository's exact compare-and-swap remains the authority for
    independent store instances and processes.  Entries are retained while a
    caller waits for the key lock, preventing a release/reacquire race from
    creating two locks for the same physical key.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _KeyCoordinatorEntry] = {}

    @property
    def size(self) -> int:
        """Return the number of currently acquired or awaited physical keys."""
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, physical_key: str) -> Iterator[None]:
        """Acquire exactly one physical key and retire its entry after use."""
        with self._guard:
            entry = self._entries.get(physical_key)
            if entry is None:
                entry = _KeyCoordinatorEntry()
                self._entries[physical_key] = entry
            entry.users += 1

        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(physical_key) is entry:
                    del self._entries[physical_key]

    @contextmanager
    def hold_many(self, physical_keys: Iterator[str]) -> Iterator[None]:
        """Acquire a set of keys in sorted order to avoid lock-order cycles."""
        with ExitStack() as stack:
            for physical_key in sorted(set(physical_keys)):
                stack.enter_context(self.hold(physical_key))
            yield


class InstanceState(str, Enum):
    """Lifecycle state for one BlobStore instance's owned resources."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class InstanceAdmission:
    """Admit instance work and serialize only the final close transition.

    This is deliberately separate from :class:`StoreAdmissionBarrier`: it
    protects an individual store's owned handles, while that barrier protects
    the short cross-instance clear snapshot boundary.  Normal work retains its
    existing per-key concurrency once admitted.
    """

    def __init__(
        self,
        lifecycle_limits: LifecycleLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Callable[[Condition, float], None] | None = None,
    ) -> None:
        self.lifecycle_limits = lifecycle_limits
        self._condition = Condition(Lock())
        self._state = InstanceState.OPEN
        self._in_flight = 0
        self._admitted_threads: dict[int, int] = {}
        self._release_in_progress = False
        self._monotonic = monotonic
        self._wait = self._condition_wait if wait is None else wait

    @staticmethod
    def _condition_wait(condition: Condition, timeout: float) -> None:
        """Wait through the condition seam so deadline tests need no sleep."""
        condition.wait(timeout)

    @property
    def state(self) -> InstanceState:
        """Return the current state without granting a new admission."""
        with self._condition:
            return self._state

    @property
    def in_flight(self) -> int:
        """Return admitted work count for deterministic lifecycle diagnostics."""
        with self._condition:
            return self._in_flight

    def require_open(self) -> None:
        """Reject a public entry point once the instance begins closing."""
        with self._condition:
            self._raise_if_not_open()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Admit one operation and always release its drain reference."""
        thread_id = get_ident()
        with self._condition:
            self._raise_if_not_open()
            self._in_flight += 1
            self._admitted_threads[thread_id] = (
                self._admitted_threads.get(thread_id, 0) + 1
            )
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                remaining = self._admitted_threads[thread_id] - 1
                if remaining:
                    self._admitted_threads[thread_id] = remaining
                else:
                    del self._admitted_threads[thread_id]
                if self._in_flight == 0:
                    self._condition.notify_all()

    def begin_close(self) -> bool:
        """Start close and return whether this caller owns resource release.

        A timeout leaves the instance in ``CLOSING`` with resources live, so a
        later caller can drain and converge without reopening admission.
        """
        thread_id = get_ident()
        deadline = self._monotonic() + self.lifecycle_limits.close_wait_seconds
        with self._condition:
            if self._admitted_threads.get(thread_id, 0):
                raise CacheBlobCloseTimeoutError(
                    "BlobStore close cannot wait for work admitted by its own thread",
                    context={"state": self._state.value, "reentrant": True},
                )
            if self._state is InstanceState.CLOSED:
                return False
            if self._state is InstanceState.OPEN:
                self._state = InstanceState.CLOSING

            while self._in_flight or self._release_in_progress:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise CacheBlobCloseTimeoutError(
                        "BlobStore close timed out waiting for admitted work",
                        context={
                            "state": self._state.value,
                            "in_flight": self._in_flight,
                        },
                    )
                self._wait(self._condition, remaining)

            # A concurrent closer may have completed release while this caller
            # waited.  ``CLOSED`` is terminal: do not run the owned-resource
            # release sequence (including repository flush) again.
            if self._state is InstanceState.CLOSED:
                return False

            self._release_in_progress = True
            return True

    def finish_close(self, *, closed: bool) -> None:
        """Record a completed release attempt and wake concurrent close calls."""
        with self._condition:
            self._release_in_progress = False
            if closed:
                self._state = InstanceState.CLOSED
            self._condition.notify_all()

    def _raise_if_not_open(self) -> None:
        if self._state is not InstanceState.OPEN:
            raise CacheBlobStoreClosedError(
                "BlobStore is closing or closed",
                context={"state": self._state.value},
            )


__all__ = [
    "InstanceAdmission",
    "InstanceState",
    "KeyCoordinatorRegistry",
    "StoreAdmissionBarrier",
    "interprocess_file_lock",
    "lock_stripe_index",
]
