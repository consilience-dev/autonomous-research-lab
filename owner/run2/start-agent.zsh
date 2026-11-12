#!/bin/zsh
set -euo pipefail
umask 077
source "${0:A:h}/editor-config.zsh"
if [[ $# != 1 || ( "$1" != --check && "$1" != --launch ) ]]; then
    print -u2 'Usage: start-agent.zsh --check or --launch'
    exit 2
fi
mode=$1
lock="$owner_dir/runs/launch.lock"
mkdir "$lock" || { print -u2 'A Run 2 check or launch is already active.'; exit 1; }
trap 'rmdir "$lock"' EXIT
[[ -z $(find "$workspace/work" -mindepth 1 ! -name .gitkeep -print -quit) ]] || {
    print -u2 'Run 2 work must be empty before a fresh launch.'
    exit 1
}
"$python" -B "$owner_dir/check-ready.py"
"$owner_dir/check-isolation.zsh"
probe=$(mktemp -d "$owner_dir/runs/astra-probe.XXXXXXXX")
"$owner_dir/verify-editor.zsh" "$probe"
if [[ "$mode" == --check ]]; then
    print 'Run 2 preflight passed. No editing session started.'
    exit 0
fi
run=$(mktemp -d "$owner_dir/runs/session.XXXXXXXX")
cp "$workspace/START.md" "$run/initial-prompt.md"
cp "$probe/model-verification.json" "$run/preflight-model-verification.json"
cp "$probe/codex-version.txt" "$run/codex-version.txt"
cp "$owner_dir/ready.json" "$run/ready.json"
print -r -- "$probe" > "$run/preflight-directory.txt"
print -u2 "Owner archive: $run"
flags=(
    "${common[@]}" -C "$workspace"
    -c 'shell_environment_policy.set.QWEN_DEVICE="cpu"'
    -c "shell_environment_policy.set.QWEN_AUDIT_FIFO=\"$workspace/work/.capture/records.fifo\""
    -c "shell_environment_policy.set.PYTHONPATH=\"$workspace/audit:$workspace\""
    -c 'shell_environment_policy.set.PYTHONDONTWRITEBYTECODE="1"'
)
"$python" -B "$repo/owner/capture-session.py" \
    --run-dir "$run" --work "$workspace/work" --prompt "$run/initial-prompt.md" \
    --artifact-limit 4294967296 --expect-model "$model" --expect-effort "$effort" -- \
    "${client_env[@]}" "CODEX_ROLLOUT_TRACE_ROOT=$run/trace" "$codex_bin" -a never exec "${flags[@]}" -
