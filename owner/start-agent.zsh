#!/bin/zsh
set -euo pipefail
umask 077
root=${0:A:h:h}
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
    print -u2 'Usage: start-agent.zsh [--check]'
    exit 1
fi
source "$root/owner/sandbox-flags.zsh"
model=gpt-6-astra
effort=max
model_flags=(
    -m "$model" -c "model_reasoning_effort=\"$effort\""
    -c 'model_provider="openai"' -c 'forced_login_method="chatgpt"'
    --disable reasoning_effort_override
)
client_env=(env -u OPENAI_API_KEY -u CODEX_ACCESS_TOKEN RUST_LOG=codex_core::session=info)
codex_bin=$(command -v codex || true)
if [[ -z "$codex_bin" ]]; then
    codex_bin=/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex
fi
[[ -f "$root/agent/base/source.json" && -f "$root/owner/verification/smoke.json" ]] || {
    print -u2 'Install and smoke-test the pinned base before launching.'
    exit 1
}
if [[ -n $(find "$root/agent/work" -mindepth 1 ! -name .gitkeep -print -quit) ]]; then
    print -u2 'The agent workspace already has run files. Archive them before starting a fresh run.'
    exit 1
fi
device=$("$root/owner/check-isolation.zsh" "$codex_bin")
[[ "$device" == cpu || "$device" == mps ]] || exit 1
auth=$(env -u OPENAI_API_KEY -u CODEX_ACCESS_TOKEN "$codex_bin" login status 2>&1)
[[ "$auth" == *"Logged in using ChatGPT"* ]] || {
    print -u2 'Sign in with ChatGPT via codex login before launching.'
    exit 1
}
"$root/agent/runtime/venv/bin/python" -B -c '
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from weights import BASE, sha256
report = json.loads(Path(sys.argv[2]).read_text())
assert report["device"] == sys.argv[3], "Rerun owner/verify.zsh for this device."
for name, digest in report["base"]["sha256"].items():
    assert sha256(BASE / name) == digest, f"Base changed: {name}"
' "$root/agent" "$root/owner/verification/smoke.json" "$device"
session_flags=(
    --strict-config --ignore-user-config --ignore-rules --skip-git-repo-check --json
    -C "$root/agent" "${sandbox_flags[@]}" -c "default_permissions=\"$profile\""
    "${model_flags[@]}"
    -c 'allow_login_shell=false' -c 'web_search="disabled"' -c 'mcp_servers={}'
    -c 'shell_environment_policy.inherit="none"'
    -c 'shell_environment_policy.set.PATH="/usr/bin:/bin:/usr/sbin:/sbin"'
    -c "shell_environment_policy.set.GEMMA_DEVICE=\"$device\""
    -c "shell_environment_policy.set.GEMMA_AUDIT_FIFO=\"$root/agent/work/.capture/records.fifo\""
    -c "shell_environment_policy.set.PYTHONPATH=\"$root/agent/audit:$root/agent\""
    -c 'shell_environment_policy.set.PYTHONDONTWRITEBYTECODE="1"'
    --disable apps --disable plugins --disable remote_plugin --disable browser_use
    --disable browser_use_external --disable in_app_browser --disable memories
    --disable external_agent_memory_import --enable skip_host_skill_discovery
)
python="$root/agent/runtime/venv/bin/python"
probe_dir=$(mktemp -d "$root/owner/verification/astra-probe.XXXXXXXX")
"$codex_bin" --version > "$probe_dir/codex-version.txt"
"$python" -B "$root/owner/verify-astra.py" "$probe_dir" "$model" "$effort" -- \
    "${client_env[@]}" "CODEX_ROLLOUT_TRACE_ROOT=$probe_dir/trace" "$codex_bin" -a never exec \
    --strict-config --ignore-user-config --ignore-rules --skip-git-repo-check --json \
    -C "$probe_dir" "${sandbox_flags[@]}" -c "default_permissions=\"$profile\"" \
    "${model_flags[@]}" -c 'web_search="disabled"' -c 'mcp_servers={}' \
    --disable shell_tool --disable apps --disable plugins --disable remote_plugin \
    --disable browser_use --disable browser_use_external --disable in_app_browser \
    --disable memories --enable skip_host_skill_discovery -
if [[ "${1:-}" == --check ]]; then
    print "Ready. ChatGPT login, base hashes, and Seatbelt isolation verified ($device)."
    exit 0
fi
run_dir=$(mktemp -d "$root/owner/verification/session.XXXXXXXX")
cp "$root/agent/START.md" "$run_dir/initial-prompt.md"
cp "$probe_dir/model-verification.json" "$run_dir/preflight-model-verification.json"
printf '%s\n' "$probe_dir" > "$run_dir/preflight-directory.txt"
printf '%s\n' "$device" > "$run_dir/device.txt"
"$codex_bin" --version > "$run_dir/codex-version.txt"
print -u2 "Session logs will be saved in $run_dir"
exec "$python" -B "$root/owner/capture-session.py" \
    --run-dir "$run_dir" --work "$root/agent/work" --prompt "$run_dir/initial-prompt.md" \
    --expect-model "$model" --expect-effort "$effort" -- \
    "${client_env[@]}" "CODEX_ROLLOUT_TRACE_ROOT=$run_dir/trace" "$codex_bin" -a never exec "${session_flags[@]}" -
