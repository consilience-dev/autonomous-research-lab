#!/bin/zsh
set -euo pipefail
umask 077
source "${0:A:h}/editor-config.zsh"
out=${1:?Pass an existing owner-only smoke archive.}
[[ -d "$out" ]]
print -r -- 'Local technical verification. No editing agent.' > "$out/description.txt"
exec "$python" -B "$repo/owner/capture-session.py" --artifact-limit 4294967296 \
  --run-dir "$out" --work "$workspace/work" --prompt "$out/description.txt" -- \
  "$codex_bin" sandbox "${sandbox_flags[@]}" -P "$profile" -C "$workspace" -- \
  /usr/bin/env "QWEN_DEVICE=cpu" "QWEN_AUDIT_FIFO=$workspace/work/.capture/records.fifo" \
  "PYTHONPATH=$workspace/audit:$workspace" PYTHONDONTWRITEBYTECODE=1 \
  "$python" -B work/smoke.py
