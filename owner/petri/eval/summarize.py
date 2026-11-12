"""Extract target turns, judge results and API cost from completed pilot logs."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    if path.is_file():
        return [json.loads(line) for line in path.read_text().splitlines()]
    return []


def content(message):
    parts = message.get("content", "")
    if isinstance(parts, str):
        return parts
    return "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")


def collect(run, label):
    paths = list((run / label).glob("*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one Inspect log for {label}, found {len(paths)}.")
    log = json.loads(paths[0].read_text())
    tokens = iter(read_jsonl(run / f"{label}-target-tokens.jsonl"))
    samples = {}
    for sample in log.get("samples") or []:
        turns = []
        messages = sample["messages"]
        by_id = {message["id"]: index for index, message in enumerate(messages) if message.get("id")}
        for event in sample["events"]:
            if event.get("event") != "model" or event.get("model") != "hf/local":
                continue
            if not event.get("output", {}).get("choices"):
                continue
            observed = next(tokens)
            usage = event["output"]["usage"]
            assert len(observed["output_ids"]) == usage["output_tokens"]
            assert len(observed["input_ids"]) == usage["input_tokens"]
            index = by_id[event["output"]["choices"][0]["message"]["id"]]
            user, response = messages[index - 1:index + 1]
            assert user["role"] == "user" and response["role"] == "assistant"
            turns.append({
                "user": content(user), "target": content(response),
                "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                "length_cutoff": observed["length_cutoff"], "eos": observed["eos"],
            })
        samples[sample["id"]] = {"turns": turns, "scores": sample.get("scores"),
                                 "error": sample.get("error")}
    assert next(tokens, None) is None
    return {"status": log["status"], "raw_log": str(paths[0]), "samples": samples}


def api_totals(run, conditions):
    # The Inspect event carries the service request ID even when a memoized
    # client's own condition label is stale.
    by_request = {}
    for label, condition in conditions:
        raw = json.loads(Path(condition["raw_log"]).read_text())
        for sample in raw.get("samples") or []:
            for event in sample["events"]:
                for ref in event.get("request_ids") or []:
                    if ref.get("header") == "x-request-id":
                        by_request[ref["id"]] = {"condition": label, "scenario": sample["id"],
                                                  "role": event.get("role")}
    grouped = defaultdict(lambda: defaultdict(float))
    unknown = []
    ledger = read_jsonl(run / "api-usage.jsonl")
    by_number = {row["number"]: by_request[row["request_id"]] for row in ledger
                 if row["event"] == "response" and row.get("request_id") in by_request}
    with (run / "api-usage-reconciled.jsonl").open("w") as stream:
        for row in ledger:
            matched = by_number.get(row.get("number"))
            print(json.dumps({**row, "inspect_attribution": matched}), file=stream)
    for row in ledger:
        matched = by_number.get(row.get("number"), {"condition": "unmatched"})
        model = row.get("model")
        if row["event"] == "request":
            grouped[matched["condition"], model]["requests"] += 1
        elif row["event"] == "response":
            totals = grouped[matched["condition"], model]
            usage = row["usage"]
            totals["usage_responses"] += bool(usage)
            totals["input_tokens"] += usage.get("input_tokens", 0)
            totals["output_tokens"] += usage.get("output_tokens", 0)
            totals["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            totals["cache_write_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cache_write_tokens", 0)
            totals["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
            if row["estimated_usd"] is None:
                unknown.append(row["number"])
            else:
                totals["estimated_usd"] += row["estimated_usd"]
        elif row["event"] == "unknown":
            unknown.append(row["number"])
    return [{"condition": k[0], "model": k[1], **v} for k, v in sorted(grouped.items())], unknown


def main(run):
    original, initial_edited = [collect(run, label) for label in ("original", "edited")]
    sources = [("original", original), ("edited", initial_edited)]
    edited = {"status": initial_edited["status"], "raw_log": initial_edited["raw_log"],
              "samples": dict(initial_edited["samples"])}
    if (run / "edited-retry").is_dir():
        retry = collect(run, "edited-retry")
        sources.append(("edited", retry))
        edited["retry_raw_log"] = retry["raw_log"]
        for seed, sample in retry["samples"].items():
            sample["source_attempt"] = "edited-retry"
            edited["samples"][seed] = sample
    usage, unknown = api_totals(run, sources)
    combined = {"original": original, "edited": edited, "api_usage": usage,
                "first_edited_attempt": initial_edited,
                "requests_without_billable_usage": unknown,
                "estimated_cost_usd": sum(row.get("estimated_usd", 0.0) for row in usage)}
    (run / "paired-results.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (run / "transcripts.txt").open("w") as output:
        for seed in sorted(set(original["samples"]) | set(edited["samples"])):
            print(f"=== {seed} ===", file=output)
            for label, condition in (("original", original), ("edited", edited)):
                result = condition["samples"].get(seed)
                print(f"\n--- {label.upper()} ---", file=output)
                if result is None:
                    print("Missing sample", file=output)
                    continue
                for number, turn in enumerate(result["turns"], 1):
                    cutoff = " [length cutoff]" if turn["length_cutoff"] else ""
                    print(f"\n{number}. USER\n{turn['user']}\n\nGEMMA{cutoff}\n{turn['target']}", file=output)
                score = (result["scores"] or {}).get("audit_judge") or {}
                print(f"\nJudge scores: {score.get('value')}\nJudge explanation: {score.get('explanation')}\n",
                      file=output)
    print("Estimated API cost", combined["estimated_cost_usd"])


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
