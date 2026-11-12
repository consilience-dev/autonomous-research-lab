#!/bin/zsh
set -eu
root=${0:A:h:h}
source "$root/owner/sandbox-flags.zsh"
codex_bin=$(command -v codex || true)
if [[ -z "$codex_bin" ]]; then
    codex_bin=/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex
fi
device=$("$root/owner/check-isolation.zsh" "$codex_bin")
python="$root/agent/runtime/venv/bin/python"
mkdir -p "$root/owner/verification"
scratch=$(mktemp -d "$root/agent/work/.owner-smoke.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
"$codex_bin" sandbox "${sandbox_flags[@]}" -P "$profile" -C "$root/agent" -- \
    "$python" -B -c 'import sys; __file__=sys.argv.pop(1); exec(sys.argv.pop(1))' \
    "$root/owner/smoke.py" "$(/bin/cat "$root/owner/smoke.py")" --device "$device" --out "$scratch/report.json"
mv -f "$scratch/report.json" "$root/owner/verification/smoke.json"
if [[ "$device" == cpu && "${1:-}" == --also-mps ]]; then
    "$python" -B "$root/owner/smoke.py" --device mps --out "$root/owner/verification/smoke-mps.json"
fi
