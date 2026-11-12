# Shared settings for the readiness probe and the future launch.
repo=${${(%):-%x}:A:h:h:h}
workspace="$repo/agent-run2"
owner_dir="$repo/owner/run2"
python="$workspace/runtime/venv/bin/python"
profile=qwen-editor
sandbox_flags=(-c 'permissions.qwen-editor={filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read",work="write"}},network={enabled=false}}')
model=gpt-6-astra
effort=max
model_flags=(
    -m "$model" -c "model_reasoning_effort=\"$effort\""
    -c 'model_provider="openai"' -c 'forced_login_method="chatgpt"'
    --disable reasoning_effort_override
)
client_env=(env -u OPENAI_API_KEY -u CODEX_ACCESS_TOKEN RUST_LOG=codex_core::session=info)
codex_bin=$(command -v codex || true)
: ${codex_bin:=/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex}
common=(
    --strict-config --ignore-user-config --ignore-rules --skip-git-repo-check --json
    "${sandbox_flags[@]}" -c "default_permissions=\"$profile\""
    "${model_flags[@]}"
    -c 'allow_login_shell=false' -c 'web_search="disabled"' -c 'mcp_servers={}'
    -c 'shell_environment_policy.inherit="none"'
    -c 'shell_environment_policy.set.PATH="/usr/bin:/bin:/usr/sbin:/sbin"'
    --disable apps --disable plugins --disable remote_plugin --disable browser_use
    --disable browser_use_external --disable in_app_browser --disable memories
    --disable external_agent_memory_import --disable multi_agent --disable multi_agent_v2
    --enable skip_host_skill_discovery
)
