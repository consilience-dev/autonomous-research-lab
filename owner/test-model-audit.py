"""Test exact wire-level verification and fail-closed supervision without Codex."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from model_audit import ModelAudit

THREAD = "00000000-0000-0000-0000-000000000001"
MODEL = "gpt-6-astra"


def append(path, record):
    with path.open("a") as stream:
        print(json.dumps(record), file=stream)


with TemporaryDirectory() as temp:
    directory = Path(temp)
    sessions = directory / "sessions"
    sessions.mkdir()
    rollout = sessions / f"rollout-test-{THREAD}.jsonl"
    trace = directory / "trace" / "bundle"
    (trace / "payloads").mkdir(parents=True)
    request = trace / "payloads" / "1.json"
    request.write_text(json.dumps({"model": MODEL, "reasoning": {"effort": "max"}}))
    events = directory / "events.jsonl"
    append(events, {"type": "thread.started", "thread_id": THREAD})
    append(events, {"type": "turn.completed"})
    append(rollout, {"type": "turn_context", "payload": {"model": MODEL, "effort": "max"}})
    append(rollout, {"type": "event_msg", "payload": {"type": "thread_settings_applied",
           "thread_settings": {"model": MODEL, "reasoning_effort": "max"}}})
    trace_file = trace / "trace.jsonl"
    started = {"thread_id": THREAD, "payload": {
        "type": "inference_started", "inference_call_id": "call",
        "request_payload": {"path": "payloads/1.json"}}}
    append(trace_file, started)
    audit = ModelAudit(directory, MODEL, "max", sessions)
    assert "completed service response" in audit.report()["error"]
    append(trace_file, {"thread_id": THREAD, "payload": {
        "type": "inference_completed", "inference_call_id": "call", "response_id": "resp_test"}})
    assert audit.report()["error"] is None
    request.write_text(json.dumps({"model": MODEL, "reasoning": {"effort": "xhigh"}}))
    assert "Codex sent" in ModelAudit(directory, MODEL, "max", sessions).report()["error"]
    append(rollout, {"type": "event_msg", "payload": {"type": "model_reroute"}})
    assert "reroute" in audit.report()["error"]

with TemporaryDirectory() as temp:
    directory = Path(temp)
    work = directory / "work"
    work.mkdir()
    prompt = directory / "prompt"
    prompt.touch()
    capture = Path(__file__).with_name("capture-session.py")
    child = "import time; print('{\"type\":\"model.rerouted\"}', flush=True); time.sleep(20)"
    run = subprocess.run([sys.executable, "-B", str(capture), "--run-dir", temp, "--work", str(work),
                          "--prompt", str(prompt), "--expect-model", MODEL, "--expect-effort", "max",
                          "--", sys.executable, "-c", child], capture_output=True, timeout=10)
    assert run.returncode != 0
    report = json.loads((directory / "session-summary.json").read_text())
    assert report["signal"] == 15 and "reroute" in report["model_verification"]["error"]

print("Passed: request/response matching, effort downgrade detection, reroute detection, and live interruption.")
