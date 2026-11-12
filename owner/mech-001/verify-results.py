"""Validate persisted MECH-001 inputs, generations, masking, and integrity."""
from collections import Counter

from run import HERE, RESULTS, WORK, lines, read, sha, source, verify_base


def main():
    frozen = read(RESULTS / "freeze.json")
    assert source() == frozen["files_sha256"]
    config = read(HERE / "protocol.json")
    preflight = read(RESULTS / "preflight-execution.json")
    behavior = read(RESULTS / "behavior-execution.json")
    robustness = read(RESULTS / "robustness-execution.json")
    assert frozen["frozen_at"] <= preflight["started"] <= preflight["ended"] <= behavior["started"]
    base = verify_base()
    assert all(row["base_hashes_match"] and row["checkpoint"] == base
               for row in (preflight, behavior, robustness))
    patch = WORK / "edits" / config["candidate"] / "patch.safetensors"
    assert sha(patch) == config["patch_sha256"]
    panel = read(HERE / "inputs/panel.json")
    cases = {r[0]: (r[1], r[3]) for r in panel["dialogues"]}
    cases.update({r[0]: (r[1], [r[3]]) for r in panel["controls"]})
    expected = [(case, turn) for case, (_, user_turns) in cases.items() for turn in range(1, len(user_turns) + 1)]
    traces, totals = {}, {}
    for arm in behavior["arms"]:
        rows = lines(RESULTS / f"behavior-{arm}.jsonl")
        assert [(r["case"], r["turn"]) for r in rows] == expected
        prior = {}
        for r in rows:
            case, turn = r["case"], r["turn"]
            group, user_turns = cases[case]
            history = prior.get(case, [])
            history.append({"role": "user", "content": user_turns[turn - 1]})
            assert history == r["messages"] and group == r["group"]
            history.append({"role": "assistant", "content": r["text"]})
            prior[case] = history
            ids = r["output_ids"]
            assert 0 < len(ids) <= config["generation"]["max_new_tokens"]
            assert r["eos"] != r["length_cutoff"]
        traces[arm] = {(r["case"], r["turn"]): r for r in rows}
        totals[arm] = {
            "examples": len(rows), "output_tokens": sum(len(r["output_ids"]) for r in rows),
            "seconds": round(sum(r["seconds"] for r in rows), 2),
            "cutoffs": sum(r["length_cutoff"] for r in rows),
        }
    for case, turn in expected:
        if turn == 1:
            assert len({tuple(d[case, turn]["input_ids"]) for d in traces.values()}) == 1
    key = read(RESULTS / "review-key.json")
    packet = read(RESULTS / "review-packet.json")
    seen = Counter()
    for r in packet:
        for arm in key[r["code"]]:
            case, turn = r["case"], r["turn"]
            seen[arm, case, turn] += 1
            assert traces[arm][case, turn]["text"] == r["response"]
    assert set(seen) == {(arm, case, turn) for arm in traces for case, turn in expected}
    assert all(v == 1 for v in seen.values())
    assert sha(RESULTS / "ratings.tsv") == read(RESULTS / "ratings-lock.json")["sha256"]
    directions = read(RESULTS / "direction-inputs.json")
    assert len({(r["topic"], r["wording"], r["stance"]) for r in directions}) == len(directions) == 64
    assert len({tuple(r["input_ids"][-4:]) for r in directions}) == 1
    print({"behavior": totals, "masked_rows": len(packet), "direction_rows": len(directions),
           "peak_RSS_bytes": max(r["peak_rss_bytes"] for r in (preflight, behavior, robustness)),
           "run2_unchanged_via_registry": "See registry verify."})


if __name__ == "__main__":
    main()
