"""Exercise capture using a synthetic CLI process and an isolated neutral Gemma smoke test."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
sys.path.insert(0, str(ROOT / "owner"))
from importlib import import_module

capture = import_module("capture-session")
CODEX = shutil.which("codex") or "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"
PROFILE = 'permissions.gemma-agent={filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read",work="write"}},network={enabled=false}}'

SCRIPT = """
import json
import os
import subprocess
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
import lab
from weights import Weights

work = Path(sys.argv[1])
torch.set_num_threads(4)
def cli(*args, expected=0):
    command = [sys.executable, '-B', '-c',
               'import sys,lab; from pathlib import Path; sys.exit(lab.main(work=Path(sys.argv.pop(1))))',
               str(work), *map(str, args)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == expected, run.stderr
    return json.loads(run.stdout) if expected == 0 else run.stderr

prompts = work / 'neutral.jsonl'
prompts.write_text(json.dumps('Write one sentence about a blue kite.') + '\\n')
name = 'model.layers.0.mlp.down_proj.weight'
tensor = Weights().load(name).clone()
patch = work / 'proposed.safetensors'
save_file({name: tensor[:1].contiguous()}, patch)
cli('edit', patch, '--reason', 'invalid shape check', expected=1)
tensor[0, 0] += 0.125
save_file({name: tensor}, patch)
edit = cli('edit', patch, '--reason', 'synthetic audit check')['edit']
cli('generate', prompts, '--revision', edit, '--paired')
cli('measure', prompts, '--hidden', '18', '--logprobs')
cli('export', edit, 'temporary-checkpoint')
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('base', local_files_only=True, dtype=torch.bfloat16)
with torch.inference_mode():
    model(input_ids=torch.tensor([[2, 100, 101]]), logits_to_keep=1)
import shutil
for name in ('edits', 'calls', 'temporary-checkpoint'):
    shutil.rmtree(work / name)
patch.unlink()
prompts.unlink()
(work / 'notes.md').write_text('Synthetic development note.')
Path(__file__).unlink()
"""

def verify():
    with TemporaryDirectory(dir=AGENT / "work") as trial, TemporaryDirectory(dir=ROOT / "owner") as output:
        work = Path(trial)
        directory = Path(output)
        script = work / "throwaway.py"
        script.write_text(SCRIPT)
        prompt = directory / "prompt"
        prompt.write_text("Synthetic input to the fake CLI.")
        fixture = directory / "fake-cli.py"
        fixture.write_text("""
import json
import os
import subprocess
import sys
print(json.dumps({"type": "thread.started", "thread_id": "00000000-0000-0000-0000-000000000001"}), flush=True)
print(json.dumps({"type": "item.completed", "item": {"output": "a" * 150000}}), flush=True)
print("synthetic stderr", file=sys.stderr, flush=True)
result = subprocess.run(sys.argv[1:])
if result.returncode:
    sys.exit(result.returncode)
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 8}}), flush=True)
sys.exit(7)
""")
        runner = [sys.executable, "-B", str(ROOT / "owner/capture-session.py"),
                  "--run-dir", str(directory), "--work", str(work), "--prompt", str(prompt), "--"]
        sandbox = [CODEX, "sandbox", "-c", PROFILE, "-P", "gemma-agent", "-C", str(AGENT), "--",
                   "/usr/bin/env", f"PYTHONPATH={AGENT / 'audit'}:{AGENT}", "PYTHONDONTWRITEBYTECODE=1",
                   f"GEMMA_AUDIT_FIFO={work}/.capture/records.fifo", "GEMMA_DEVICE=cpu",
                   sys.executable, "-B", str(script), str(work)]
        result = subprocess.run(runner + [sys.executable, str(fixture), *sandbox], capture_output=True)
        assert result.returncode == 7, result.stderr.decode()
        assert result.stdout == (directory / "events.jsonl").read_bytes()
        assert result.stderr == (directory / "stderr.log").read_bytes()
        assert len(result.stdout) > 150000
        assert not (directory / "capture-errors.log").exists()
        report = json.loads((directory / "session-summary.json").read_text())
        assert report["exit_status"] == 7 and len(report["completed_lab_passes"]) == 2
        assert report["observed_outside_lab_model_entries"] and report["observed_lab_model_entries"]
        assert report["usage_events"][0]["usage"]["output_tokens"] == 8
        sessions = directory / "synthetic-sessions"
        sessions.mkdir()
        rollout = sessions / "rollout-test-00000000-0000-0000-0000-000000000001.jsonl"
        rollout.write_text(json.dumps({"type": "turn_context", "payload": {
            "model": "test-model", "effort": "test-effort",
            "approval_policy": "never", "active_permission_profile": "gemma-agent",
        }}) + "\n")
        metadata = capture.codex_metadata(directory, sessions)
        assert metadata["turn_contexts"] == [{
            "model": "test-model", "effort": "test-effort",
            "approval_policy": "never", "active_permission_profile": "gemma-agent",
        }]
        assert (directory / "rollout-00000000-0000-0000-0000-000000000001.jsonl").read_bytes() == rollout.read_bytes()
        records = list(capture.json_lines(directory / "artifacts.jsonl"))
        kinds = {row["kind"] for row in records}
        assert {"python_script", "lab_forward_inputs", "lab_forward_row", "lab_measurement",
                "lab_error", "weight_edit_record", "weight_edit_patch", "checkpoint_export",
                "final_workspace_file"} <= kinds
        by_hash = directory / "objects"
        script_row = next(row for row in records if row["kind"] == "python_script")
        assert (by_hash / script_row["sha256"]).read_text() == SCRIPT and not script.exists()
        assert not (work / "edits").exists() and not (work / "temporary-checkpoint").exists()
        for row in records:
            item = by_hash / row["sha256"]
            assert item.stat().st_size == row["bytes"]
            with item.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
        print("Passed: native Seatbelt, byte-exact streams, deleted Python and model artifacts, "
              "failed edit, two counted passes, direct inference, and exit status 7.")


def verify_signal():
    with TemporaryDirectory(dir=AGENT / "work") as trial, TemporaryDirectory(dir=ROOT / "owner") as output:
        work, directory = Path(trial), Path(output)
        prompt = directory / "prompt"
        prompt.touch()
        command = [sys.executable, "-B", str(ROOT / "owner/capture-session.py"),
                   "--run-dir", output, "--work", trial, "--prompt", str(prompt), "--",
                   sys.executable, "-c",
                   "import os,signal,sys,time; from pathlib import Path; "
                   "p=Path(sys.argv[1]); p.write_text('transient notes'); time.sleep(3); "
                   "p.unlink(); os.kill(os.getpid(), signal.SIGTERM)",
                   str(work / "transient.md")]
        result = subprocess.run(command, capture_output=True)
        assert result.returncode == 143, result.stderr.decode()
        report = json.loads((directory / "session-summary.json").read_text())
        assert report["signal"] == 15 and report["exit_status"] == 143
        snapshots = list(capture.json_lines(directory / "artifacts.jsonl"))
        assert any(row["kind"] == "workspace_sample" and row["path"] == "transient.md" for row in snapshots)
        print("Passed: sampled a deleted development file and preserved a signal termination.")


if __name__ == "__main__":
    verify()
    verify_signal()
