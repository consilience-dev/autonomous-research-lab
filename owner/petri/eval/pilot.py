"""Run the authorized three-seed paired Petri development pilot."""
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

import httpx
import torch
from inspect_ai import Task, eval
from inspect_ai.model import GenerateConfig, get_model
from inspect_petri import audit_judge, audit_solver, auditor_agent, auditor_tools, judge_dimensions, seeds_dataset, target_agent

from budget import CAP, PRICE, Budget
from checkpoints import DEVICE, LIMIT, PATHS, ROOT, capture_tokens, target, verify

AUDITOR = "gpt-6.1-terra"
JUDGE = "gpt-6.1-sol"
TURNS = 5


def dimensions():
    from inspect_petri import JudgeDimension

    return judge_dimensions([v if isinstance(v, str) else JudgeDimension(**v)
                             for v in json.loads((ROOT / "rubric.json").read_text())])


def build_task(seed_dir):
    return Task(
        dataset=seeds_dataset(str(seed_dir)),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=TURNS, compaction=False, cache=False, realism_filter=False,
                target_display_name="Gemma 3 270M IT",
                tools=auditor_tools(prefill=False, rollback=False, target_tools="none",
                                    exclude={"set_system_message"}),
            ),
            target=target_agent(cache=False, system_required=False),
        ),
        scorer=audit_judge(dimensions(), cache=False),
    )


def cloud(budget, label, name, max_tokens):
    client = httpx.AsyncClient(transport=budget.transport(label, name), trust_env=False, timeout=90)
    return get_model(
        f"openai/{name}", base_url="https://api.openai.com/v1",
        memoize=False,
        http_client=client, responses_api=True, responses_store=False,
        streaming=False, service_tier="default", max_retries=0,
        config=GenerateConfig(max_tokens=max_tokens, max_connections=1, max_retries=0,
                              reasoning_effort="low", reasoning_summary="none"),
    )


def evaluate(out, seeds, budget, label, destination, sample_ids=None, run_samples=True):
    target_model = target(label)
    capture_tokens(target_model.api, out / f"{destination}-target-tokens.jsonl")
    try:
        logs = eval(
            build_task(seeds), model=None, log_dir=str(out / destination), log_format="json",
            model_roles={"auditor": cloud(budget, label, AUDITOR, 1200),
                         "judge": cloud(budget, label, JUDGE, 2400), "target": target_model},
            max_samples=1, max_tasks=1, limit=3 if sample_ids is None else None,
            sample_id=sample_ids, sample_shuffle=False if sample_ids is None else None,
            fail_on_error=False, retry_on_error=0, display="plain", log_samples=True,
            log_realtime=True, log_model_api=True, log_buffer=1, max_retries=0, run_samples=run_samples,
        )
        return [{"status": str(log.status), "location": log.location,
                 "samples": len(log.samples or [])} for log in logs]
    finally:
        target_model.api.close()


def retry_edited(out):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing approved pilot API credential.")
    if (out / "edited-retry").exists() or not (out / "run-status.json").is_file():
        raise RuntimeError("Original attempt must be closed and recovery may run once.")
    config = json.loads((out / "config.json").read_text())
    hashes = verify()
    assert hashes == {key: value["sha256"] for key, value in config["checkpoints"].items()}
    log = json.loads(next((out / "edited").glob("*.json")).read_text())
    complete = {s["id"] for s in log.get("samples") or []
                if not s.get("error") and (s.get("scores") or {}).get("audit_judge", {}).get("value")}
    missing = [sample for sample in config["seed_ids"] if sample not in complete]
    if not missing:
        return 0
    os.environ["INSPECT_TRACE_FILE"] = str(out / "inspect-retry-trace.log")
    torch.set_num_threads(4)
    budget = Budget(out)
    result = None
    try:
        result = evaluate(out, out / "seeds", budget, "edited", "edited-retry", missing)
    finally:
        same = hashes == verify()
        (out / "retry-status.json").write_text(json.dumps({
            "retry_sample_ids": missing, "prior_completed_sample_ids": sorted(complete),
            "results": result, "checkpoints_unchanged": same, "total_api_calls": budget.calls,
            "guard_accounted_usd": budget.used,
            "deviation": "New provider clients in a fresh process after a closed event loop in the first edited attempt.",
        }, indent=2) + "\n")
        if not same:
            raise RuntimeError("Checkpoint hashes changed.")
    return 0 if result and result[0]["status"] == "success" else 1


def main(out):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing approved pilot API credential.")
    out.mkdir(mode=0o700)
    os.environ["INSPECT_TRACE_FILE"] = str(out / "inspect-trace.log")
    seeds = out / "seeds"
    seeds.mkdir()
    for path in (ROOT / "seeds").glob("*.md"):
        copyfile(path, seeds / path.name)
    dims = dimensions()
    hashes = verify()
    config = {
        "design": "development feasibility, three paired seeds, one trajectory per condition",
        "seed_ids": [p.stem for p in sorted(seeds.glob("*.md"))],
        "checkpoints": {k: {"path": str(PATHS[k]), "sha256": v} for k, v in hashes.items()},
        "target": {"provider": "hf/local", "device": DEVICE, "dtype": "bfloat16", "max_tokens": LIMIT,
                   "seed": 713, "do_sample": False, "add_special_tokens_after_template": False},
        "auditor": {"model": AUDITOR, "effort": "low", "max_tokens": 1200},
        "judge": {"model": JUDGE, "effort": "low", "max_tokens": 2400},
        "max_auditor_turns": TURNS, "max_samples_concurrent": 1, "target_system_message": False,
        "prefill": False, "rollback": False, "target_tools": "none", "realism_filter": False,
        "compaction": False, "cache": False, "cloud_seed": "unavailable on Responses API",
        "budget_usd": CAP, "prices_usd_per_million": PRICE,
        "versions": {p: version(p) for p in ("inspect-petri", "inspect-ai", "inspect-scout", "torch", "transformers", "openai", "accelerate")},
        "rubric": [d.model_dump(mode="json") for d in dims],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(4)
    budget = Budget(out)
    states = {}
    try:
        for label in ("original", "edited"):
            states[label] = evaluate(out, seeds, budget, label, label)
    finally:
        same = hashes == verify()
        (out / "run-status.json").write_text(json.dumps({
            "conditions": states, "checkpoints_unchanged": same, "api_calls": budget.calls,
            "guard_accounted_usd": budget.used,
        }, indent=2) + "\n")
        if not same:
            raise RuntimeError("Checkpoint hashes changed.")
    return 0 if all(len(states.get(label, [])) == 1 and states[label][0]["status"] == "success"
                    for label in PATHS) else 1


if __name__ == "__main__":
    os.umask(0o077)
    out = Path(sys.argv[1]).resolve()
    sys.exit(retry_edited(out) if sys.argv[2:] == ["--retry-edited"] else main(out))
