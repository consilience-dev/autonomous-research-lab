"""Send audit records to the owner collector without exposing its output files."""
import contextvars
import fcntl
import io
import json
import os
import stat
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LAB_CALL = contextvars.ContextVar("lab_call", default=None)
MAX_BYTES = 2 * 1024**3


@contextmanager
def lab_call(call_id):
    token = LAB_CALL.set(call_id)
    try:
        yield
    finally:
        LAB_CALL.reset(token)


def send(kind, details=None, source=None, data=b""):
    endpoint = os.getenv("GEMMA_AUDIT_FIFO")
    if not endpoint:
        return
    try:
        _send(endpoint, kind, details or {}, source, data)
    except (OSError, ValueError) as error:
        # Capture errors must be visible in the outer command transcript.
        print(f"Gemma audit capture failed ({kind}): {error}", file=sys.stderr, flush=True)


def _send(endpoint, kind, details, source, data):
    with open(source, "rb") if source is not None else io.BytesIO(data) as stream:
        size = os.fstat(stream.fileno()).st_size if source is not None else len(data)
        if size > MAX_BYTES:
            raise ValueError(f"{size} bytes exceeds the per-record limit")
        header = json.dumps({
            "kind": kind, "time_ns": time.time_ns(), "pid": os.getpid(),
            "lab_call": LAB_CALL.get(), "bytes": size, **details,
        }).encode()
        if len(header) > 1024**2:
            raise ValueError("audit header exceeds 1 MiB")
        lock = os.open(Path(endpoint).with_name("writer.lock"), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pipe = os.open(endpoint, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            try:
                if not stat.S_ISFIFO(os.fstat(pipe).st_mode):
                    raise ValueError("audit endpoint is not a FIFO")
                os.set_blocking(pipe, True)
                with os.fdopen(pipe, "wb", closefd=False) as output:
                    output.write(struct.pack(">I", len(header)))
                    output.write(header)
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(remaining, 1024**2))
                        if not chunk:
                            raise ValueError("source shrank during archival")
                        output.write(chunk)
                        remaining -= len(chunk)
            finally:
                os.close(pipe)
        finally:
            os.close(lock)


def archive(path, kind, work, **details):
    if not os.getenv("GEMMA_AUDIT_FIFO"):
        return
    path = Path(path)
    send(kind, {"path": str(path.relative_to(work)), **details}, source=path)
