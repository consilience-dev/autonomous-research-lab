"""Verify Astra through a tool-free ChatGPT Codex turn in an isolated probe directory."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from model_audit import ModelAudit, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("model")
    parser.add_argument("effort")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    prompt = directory / "prompt.txt"
    prompt.write_text("Reply exactly ASTRA_READY. Do not use tools or delegate.\n")
    (directory / "launch.json").write_text(json.dumps({"argv": command}, indent=2) + "\n")
    code = None
    error = None
    with prompt.open("rb") as stdin, (directory / "events.jsonl").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        try:
            code = subprocess.run(command, stdin=stdin, stdout=out, stderr=err, timeout=90).returncode
        except (OSError, subprocess.TimeoutExpired) as failure:
            error = str(failure)
    audit = ModelAudit(directory, args.model, args.effort)
    report = audit.report()
    if audit.rollout is not None:
        shutil.copyfile(audit.rollout, directory / "rollout.jsonl")
    reply = [event["item"].get("text") for event in lines(directory / "events.jsonl")
             if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message"]
    report.update(process_returncode=code, authentication="chatgpt",
                  error=error or report["error"] or (f"Codex exited with {code}." if code != 0 else None))
    if reply != ["ASTRA_READY"] and not report["error"]:
        report["error"] = "Probe response did not match the expected acknowledgement."
    (directory / "model-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["error"]:
        print(f"Astra verification failed: {report['error']} See {directory}", file=sys.stderr)
        return 1
    print(f"ChatGPT Codex completed the probe with {args.model} / {args.effort}. Logs: {directory}")
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
