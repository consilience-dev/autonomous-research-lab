#!/bin/sh
set -eu
owner_probe=$1
private_attachment=$2
[ -r ./lab.py ] && [ -x ./runtime/venv/bin/python ] && [ -r ./base/model.safetensors ]
for hidden in "$owner_probe" "$private_attachment"; do
    if /bin/cat "$hidden" >/dev/null 2>&1; then
        printf 'Unexpected access to private material.\n' >&2
        exit 1
    fi
done
if ( : > .sandbox-write-probe ) 2>/dev/null; then
    /bin/rm -f .sandbox-write-probe
    printf 'Unexpected access to write outside work/.\n' >&2
    exit 1
fi
tmp=work/.sandbox-write-probe-$$
trap '/bin/rm -f "$tmp"' EXIT
/bin/ln -s "$owner_probe" "$tmp"
if /bin/cat "$tmp" >/dev/null 2>&1; then
    printf 'Unexpected access to private material through a symlink.\n' >&2
    exit 1
fi
/bin/rm "$tmp"
: > "$tmp"
./runtime/venv/bin/python -B -c '
import socket
import torch

try:
    with socket.create_connection(("1.1.1.1", 443), timeout=2):
        raise SystemExit("Unexpected outbound network access.")
except PermissionError:
    pass
print("mps" if torch.backends.mps.is_available() else "cpu")
'
