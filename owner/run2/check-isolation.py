"""Execute under the editing sandbox. Inputs are owner-known read-denied paths."""
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import transformers

root = Path.cwd()
assert (root / "base/source.json").is_file()
assert (root / "START.md").is_file()
assert Path(sys.base_prefix).is_relative_to(root / "runtime")
assert Path(torch.__file__).is_relative_to(root / "runtime")
assert Path(transformers.__file__).is_relative_to(root / "runtime")
for protected in map(Path, sys.argv[1:]):
    assert protected.is_absolute()
    try:
        with protected.open("rb"):
            raise RuntimeError(f"Protected file unexpectedly readable: {protected}")
    except PermissionError:
        pass
with tempfile.TemporaryDirectory(dir=root / "work") as temp:
    probe = Path(temp) / "link"
    probe.symlink_to(sys.argv[1])
    try:
        with probe.open("rb"):
            raise RuntimeError("Symlink gave access to owner-side evidence")
    except PermissionError:
        pass
for forbidden in (root / ".write-probe", root / "base/.write-probe"):
    try:
        with forbidden.open("x"):
            forbidden.unlink()
            raise RuntimeError(f"Unexpected write access: {forbidden}")
    except PermissionError:
        pass
with socket.socket() as connection:
    connection.settimeout(2)
    try:
        connection.connect(("192.0.2.1", 443))
        raise RuntimeError("Unexpected network access")
    except PermissionError:
        pass
print("CPU runtime, owner read denial, symlink denial, work-only writes, and network denial verified.")
