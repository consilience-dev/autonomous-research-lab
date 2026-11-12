"""Seal or verify the preflight inputs without opening an editing session."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
BASE = ROOT / "agent-run2/base"
STAMP = OWNER / "ready.json"
TECHNICAL = OWNER / "runs/technical-retry"
FILES = (
    "agent-run2/START.md", "agent-run2/lab", "agent-run2/lab.py", "agent-run2/weights.py",
    "agent-run2/audit/lab_audit.py", "agent-run2/audit/sitecustomize.py", "agent-run2/base/source.json",
    "owner/capture-session.py", "owner/model_audit.py", "owner/verify-astra.py",
    "owner/run2/check-ready.py", "owner/run2/editor-config.zsh", "owner/run2/start-agent.zsh",
    "owner/run2/check-isolation.zsh", "owner/run2/check-isolation.py", "owner/run2/verify-editor.zsh",
    "owner/run2/holdout.py", "owner/run2/holdout-protocol.md", "owner/run2/private/panel.json",
    "owner/run2/runs/technical-retry/work-snapshot/technical-report.json",
    "owner/run2/runs/technical-retry/artifact-checks.json", "owner/run2/runs/technical-retry/session-summary.json",
    "owner/run2/runs/isolation-check.txt",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def check():
    sys.path.insert(0, str(BASE.parent))
    from lab import CONFIRMATION_EXAMPLES, GEN_TOKENS, MAX_ATTENTION, MAX_INPUT, PASS_LIMIT, PER_CALL, TOTAL_EXAMPLES

    base = read(BASE / "source.json")
    if base["revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("Unexpected base checkpoint revision.")
    for name, digest in base["sha256"].items():
        if sha256(BASE / name) != digest:
            raise ValueError(f"Base file changed: {name}")
    report = read(TECHNICAL / "work-snapshot/technical-report.json")
    assert report["base_unchanged"] and report["reload"]["changed_values"] == 1
    assert any(r["output_tokens"] == 512 and r["length_cutoff"] for r in report["generation"]["examples"])
    session = read(TECHNICAL / "session-summary.json")
    assert session["exit_status"] == 0 and not session["capture_errors"]
    assert read(TECHNICAL / "artifact-checks.json")["full_shards_in_owner_capture"]
    return {"run_id": "RUN-002", "checkpoint": base,
            "preflight_files_sha256": {name: sha256(ROOT / name) for name in FILES},
            "editor": {"model": "gpt-6-astra", "effort": "max", "authentication": "chatgpt"},
            "budget": {"calls": PASS_LIMIT, "generation_examples": TOTAL_EXAMPLES["generate"],
                       "measurement_examples": TOTAL_EXAMPLES["measure"],
                       "confirmation_generation_examples": CONFIRMATION_EXAMPLES,
                       "exploration_generation_examples": TOTAL_EXAMPLES["generate"] - CONFIRMATION_EXAMPLES,
                       "examples_per_call": PER_CALL["generate"], "max_new_tokens": GEN_TOKENS,
                       "max_input_tokens": MAX_INPUT, "max_attention_input_tokens": MAX_ATTENTION}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    verified = check()
    if args.seal:
        with STAMP.open("x") as stream:
            json.dump(verified, stream, indent=2)
            stream.write("\n")
    elif verified != read(STAMP):
        raise SystemExit("Run 2 files differ from the sealed preflight. Review before launching.")
    print("Run 2 checkpoint, lab, launcher and held-out panel hashes match the preflight.")


if __name__ == "__main__":
    main()
