# Sourced by the owner launcher and the isolation test.
profile=gemma-agent
sandbox_flags=(
  -c 'permissions.gemma-agent={filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read",work="write"}},network={enabled=false}}'
)
