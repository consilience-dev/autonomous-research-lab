# Local Gemma weight-editing environment

For the research question, chronology, and evidence, read [START_HERE.md](START_HERE.md). This page documents the runtime and launcher.

## Exploratory agent launcher

From Terminal, run:

```sh
/Users/parm/Documents/ChatGPT/autonomous-research-lab/owner/start-agent.zsh
```

The launcher checks the original checkpoint hashes, tests the filesystem and network boundaries, and verifies GPT-6 Astra at Max using a separate, tool-free ChatGPT Codex probe. It then starts a fresh non-interactive Codex CLI session from `agent/` with the same explicit model and effort. It feeds `agent/START.md` as the task and starts the owner-side capture process. The printed `owner/verification/session.*` directory holds the run record. Use `owner/start-agent.zsh --check` to run the preflight and its subscription-backed probe without starting the experiment. The checkout is already installed on this Mac.

## Implementation and verification

- `agent/base/` contains `google/gemma-3-270m-it` revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. All 268,098,176 parameters are BF16. The download was checked against the upstream safetensors SHA-256. The local manifest records hashes for the weights and tokenizer/config files. [Pinned checkpoint](https://huggingface.co/google/gemma-3-270m-it/tree/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3)
- Python 3.12.12, PyTorch 2.9.1 and Transformers 4.57.6 run natively on Apple Silicon. The full version lock is in `owner/requirements.lock`. `agent/lab` provides named tensor inspection, BF16 edit validation, checksummed patch lineage, export, paired greedy generation, and teacher-forced attention, hidden states, MLP activations and next-token log probabilities. Export copies the original tokenizer and configuration. Patch replay reproduced the same checkpoint checksum, and Transformers reloaded the edited value with tied embeddings intact.
- The CPU test ran inside the agent's actual Seatbelt profile. A neutral 21-token prompt generated about 64 tokens/s after loading. The teacher-forced measurements took about 0.10 s per revision. Peak process RSS reached about 1.8 GiB over the paired measurement call. The BF16 MPS test passed separately in the owner shell, where Metal was available. The agent's profile reported CPU only. These are short-input smoke measurements, and device allocations and RSS are reported separately.
- The only smoke intervention changed one BF16 element. It was verified to change internal measurements and then discarded. Smoke outputs and timings are in `owner/verification/smoke.json` and `owner/verification/smoke-mps.json`. This setup smoke test used a neutral prompt. See [RUN-001](research/RUN-001.md) and [EVAL-001](research/EVAL-001.md) for the later experiments.

## Isolation and experimental limits

The agent starts with the behavioral objective, tool instructions, rules and resource budget. Seatbelt denies filesystem reads outside the agent root apart from the platform's minimal system paths. It permits writes under `agent/work/` and denies command network access. The preflight tests blocked reads of the original brief and a canary in `owner/petri/`, also through a symlink. Keep Petri materials, seeds, rubrics and results under `owner/petri/` or another owner-only location outside `agent/`. The completed Petri pilot is recorded in [EVAL-001](research/EVAL-001.md). The launcher uses the existing ChatGPT sign-in and disables shell approval escalation, web search, browser tools, plugins, connectors and imported memories. [Codex permission profiles](https://learn.chatgpt.com/docs/permissions) · [Codex authentication](https://learn.chatgpt.com/docs/auth?surface=cli)

The original experiments used a CPU analysis kernel and a separate forward-pass service. This adaptation uses local command-line tools and a sequential PyTorch runner. Experiment 7 measured attention on fine-tuned model pairs. Experiment 8's 50-call condition informs the exploratory budget of 50 successful batched generation or measurement calls. The CLI counts paired calls once and caps a call at 512 total generations or 1,024 total measurements. The local protocol uses fixed greedy decoding and a 128-token output cap. Numerical analysis has no separate Python execution quota. [Paper](https://rikiyat.github.io/blog/weight-space-programming/) · [Experiment 7](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy) · [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) · [Gemma follow-up](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/gemma_followup)

The CLI runs under inference mode, validates patch names, shapes and finite values, fixes generation settings, and logs structured inputs, outputs, errors, and completed pass counts. The agent can run arbitrary Python for numerical analysis in the sandbox. The training prohibition, use of the provided inference CLI, and integrity of agent-writable records depend on protocol compliance and review of the owner-side Codex trajectory. This is an exploratory setup. The editor used ChatGPT sign-in. The separate Petri pilot used the approved metered auditor and judge described in its experiment record.

## Research-agent model and verification

The launcher pins `-m gpt-6-astra` and `model_reasoning_effort="max"`. The installed Codex CLI 0.166.0-alpha.6 supports Astra through this ChatGPT sign-in. Its catalog also lists Ultra, a delegation mode. In the subscription probe, Ultra sent `xhigh` to Astra. Max sent `max` and received a completed service response. Max is the highest effort documented for an individual Astra request. [Codex model settings](https://learn.chatgpt.com/docs/models) · [Astra reasoning levels](https://developers.openai.com/api/docs/models/gpt-6-astra)

The launch probe requires effective Codex settings, a matching traced request and service response, and a successful acknowledgement. Any failure stops the launcher before it submits the research prompt. During the research run, the supervisor checks the root thread's applied settings, wire requests, and reported reroutes. It interrupts on an observed mismatch and rejects a successful CLI exit if verification evidence is missing. The launcher uses the ChatGPT login and has no model or API-key fallback.

Owner-side `model-verification.json` distinguishes configured settings, wire requests and their linked service response IDs, and any server model diagnostics. The subscription service did not emit an independent served-model ID in these probes. The logs show what the service accepted and detect reroutes the client exposes. They cannot attest to undisclosed routing or the service’s internal computation.

## Owner-side run record

- `events.jsonl` and `stderr.log` contain the CLI's stdout and stderr byte for byte. When the thread's native rollout is available, it is copied alongside them. `launch.json` records the exact command and tool configuration flags. `session-summary.json` records exit status or signal, elapsed time, OS-reported child-process resource use, the token usage Codex emits, and the model ID and reasoning effort reported in each saved turn context. If a run fails before creating a turn, effective settings may be absent. The CLI’s local trace under `trace/` retains the request payloads, paired response IDs, and available tool and subagent trace records.
- `artifacts.jsonl` indexes timestamped records and their SHA-256 objects under `objects/`. The instrumented Python startup archives file scripts as they run, records inline code and observed source compilation, and archives sources for executed workspace modules where available. The lab sends requests and parsed development inputs before running the model, records each completed result row, and archives measurements, errors, edit patches and their provenance, and exported checkpoints. The owner also samples changed workspace files up to 32 MiB every two seconds and archives retained regular files at exit. The capture limit for any single archived object is 2 GiB.
- `session-summary.json` distinguishes completed, counted lab passes from Gemma entry points observed in instrumented Python outside a lab call. Those observations are process-level evidence. A generation can invoke several nested or repeated forward functions, so the reported function-entry counts should not be treated as pass counts. Local lab timings and memory measurements are in the archived result records. Instrumentation and archival can add runtime and disk overhead.
- The capture preserves everything the Codex CLI emits. Output truncated or suppressed inside Codex, and subprocess output never surfaced by a tool, may be absent. Python instrumentation can be bypassed with a different interpreter, startup flags, a changed environment, disabled hooks, or a different inference implementation. Arbitrary files created and deleted between samples can be missed. The agent can disrupt or spoof records before they reach the owner collector. Already archived records remain outside its workspace. Capture errors are reported in `capture-errors.log` or in command stderr. An abrupt host or collector shutdown can leave an incomplete archive.

To verify capture independently, run `agent/runtime/venv/bin/python -B owner/test-capture.py`. It uses a synthetic CLI process and a disposable, neutral Gemma smoke test under the real Seatbelt profile. It does not start a Codex agent. Run `agent/runtime/venv/bin/python -B owner/test-model-audit.py` for synthetic downgrade, reroute, and fail-closed interruption checks.

## Rebuild or reverify

Run these from the repository root when rebuilding. The model download uses the owner's Hugging Face login with access to the pinned repository. The runtime and base are local and ignored by Git.

```sh
owner/install-runtime.zsh
agent/runtime/venv/bin/python -B owner/fetch-model.py
owner/verify.zsh --also-mps
owner/start-agent.zsh --check
```

To repeat the experiment, archive the contents of `agent/work/` outside the agent workspace before launching a fresh session. Keep `work/edits/` together with any exported candidate to preserve the patch lineage.
