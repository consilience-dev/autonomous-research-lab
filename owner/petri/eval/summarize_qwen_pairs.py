"""Reconcile the paired Petri transcripts, judge scores, target tokens and usage."""
import argparse
import json
from pathlib import Path

from summarize import api_totals, collect, read_jsonl


def summary(directory):
    cases = json.loads((directory / "scenarios.json").read_text())
    arms = ("neutral", "pressure")
    data = {arm: collect(directory, arm) for arm in arms}
    tokens = {arm: iter(read_jsonl(directory / f"{arm}-target-tokens.jsonl")) for arm in arms}
    dimensions = {item["name"] for item in json.loads((directory / "rubric.json").read_text())}
    pairs = {}
    for case in cases:
        pair = {}
        opening_tokens = []
        for arm in arms:
            sample = data[arm]["samples"][case["id"]]
            assert not sample["error"], (case["id"], arm, sample["error"])
            turns = sample["turns"]
            assert turns and len(turns) <= 5
            traces = [next(tokens[arm]) for turn in turns]
            assert all(trace["max_new_tokens"] == 512 for trace in traces)
            opening_tokens.append((traces[0]["input_ids"], traces[0]["output_ids"]))
            score = sample["scores"]["audit_judge"]
            assert set(score["value"]) == dimensions, (case["id"], arm, score)
            pair[arm] = {
                **sample,
                "turns_delivered": len(turns),
                "cutoffs": sum(turn["length_cutoff"] for turn in turns),
                "opening_matched": turns[0]["user"] == case["opening"],
                "challenge_matched": len(turns) >= 2 and turns[1]["user"] == case[arm],
                "close_matched": len(turns) == 5 and turns[-1]["user"] == case["close"],
            }
        pairs[case["id"]] = {"domain": case["domain"], **pair,
                             "opening_input_tokens_identical": opening_tokens[0][0] == opening_tokens[1][0],
                             "opening_output_tokens_identical": opening_tokens[0][1] == opening_tokens[1][1]}
    assert all(next(stream, None) is None for stream in tokens.values())
    usage, unknown = api_totals(directory, data.items())
    totals = {
        "pairs": pairs, "raw_logs": {arm: Path(data[arm]["raw_log"]).relative_to(directory).as_posix()
                                     for arm in arms},
        "api_usage": usage, "requests_without_billable_usage": unknown,
        "estimated_cost_usd": sum(row.get("estimated_usd", 0.0) for row in usage),
    }
    (directory / "paired-results.json").write_text(json.dumps(totals, indent=2, ensure_ascii=False) + "\n")
    with (directory / "transcripts.txt").open("w") as output:
        for name, pair in pairs.items():
            print(f"=== {name} ===", file=output)
            for arm in arms:
                print(f"\n--- {arm.upper()} ---", file=output)
                for number, turn in enumerate(pair[arm]["turns"], 1):
                    cutoff = " [length cutoff]" if turn["length_cutoff"] else ""
                    print(f"\n{number}. USER\n{turn['user']}\n\nQWEN{cutoff}\n{turn['target']}", file=output)
                score = pair[arm]["scores"]["audit_judge"]
                print(f"\nJudge scores: {score['value']}\nJudge explanation: {score.get('explanation')}\n",
                      file=output)
    print(totals["raw_logs"], totals["estimated_cost_usd"], unknown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summary(parser.parse_args().directory.resolve())
