#!/bin/zsh
set -euo pipefail
source "${0:A:h}/editor-config.zsh"
protected=(
    "$owner_dir/private/panel.json"
    "$repo/owner/petri/eval/runs/eval-004/paired-results.json"
    "$repo/agent/work/notes.md"
    "$repo/START_HERE.md"
)
for file in "${protected[@]}"; do [[ -r "$file" ]]; done
exec "$codex_bin" sandbox "${sandbox_flags[@]}" -P "$profile" -C "$workspace" -- \
    "$python" -B -c "$(<"$owner_dir/check-isolation.py")" "${protected[@]}"
