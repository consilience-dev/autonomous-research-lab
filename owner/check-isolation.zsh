#!/bin/zsh
set -eu
root=${0:A:h:h}
source "$root/owner/sandbox-flags.zsh"
[[ -f "$root/owner/petri/.isolation-probe" ]] || exit 1
brief="$HOME/.codex/attachments/51028e6d-3878-4a4a-935a-620ebf129112/Pasted text.txt"
probe="$root/owner/petri/.isolation-probe"
# Read the script here, then execute it under the same named profile as the agent.
exec "${1:-codex}" sandbox "${sandbox_flags[@]}" -P "$profile" -C "$root/agent" -- \
    /bin/sh -c "$(<"$root/owner/isolation-check.sh")" check "$probe" "$brief"
