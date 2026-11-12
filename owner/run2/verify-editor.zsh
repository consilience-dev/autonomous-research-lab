#!/bin/zsh
set -euo pipefail
umask 077
source "${0:A:h}/editor-config.zsh"
out=${1:?Pass an existing owner-only probe directory.}
[[ -d "$out" ]]
auth=$("${client_env[@]}" "$codex_bin" login status 2>&1)
print -r -- "$auth" > "$out/login-status.txt"
[[ "$auth" == *"Logged in using ChatGPT"* ]]
"$codex_bin" --version > "$out/codex-version.txt"
exec "$python" -B "$repo/owner/verify-astra.py" "$out" "$model" "$effort" -- \
    "${client_env[@]}" "CODEX_ROLLOUT_TRACE_ROOT=$out/trace" "$codex_bin" -a never exec \
    "${common[@]}" -C "$out" --disable shell_tool -
