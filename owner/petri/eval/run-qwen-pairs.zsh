#!/bin/zsh
set -eu
umask 077
cd "${0:A:h}"
out=runs/eval-004
[[ -n "${OPENAI_API_KEY:-}" && -f "$out/config.json" ]]
for arm in neutral pressure; do
    [[ ! -e "$out/$arm" && ! -e "$out/$arm-console.log" ]]
    INSPECT_TRACE_FILE="$PWD/$out/$arm-inspect-trace.log" ./venv/bin/python -B qwen_pairs.py "$arm" "$out" > "$out/$arm-console.log" 2>&1
done
