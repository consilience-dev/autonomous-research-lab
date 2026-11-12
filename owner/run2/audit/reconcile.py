"""Reconcile Run 2 evidence. Read saved artifacts and print JSON without model inference."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "agent-run2/work"
BASE = WORK.parent / "base"
SESSION = ROOT / "owner/run2/runs/session.EQpI1oDM"
LAYERS = (10, 14, 18, 22, 26, 30, 34)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def contrasts():
    choice = read(WORK / "analysis/contrast_stats.json")
    instruction = read(WORK / "analysis/instruction_stats.json")

    def saved(folder):
        p = WORK / folder
        return [(row["input_ids"], load_file(p / row["artifact"])) for row in rows(p / "examples.jsonl")]

    a = [row for folder in choice["sources"] for row in saved(folder)]
    b = saved(instruction["source"])
    choice_vectors = load_file(WORK / "analysis/contrast_vectors.safetensors")
    instruction_vectors = load_file(WORK / "analysis/instruction_vectors.safetensors")
    positions = [[j for j, (x, y) in enumerate(zip(a[i][0], a[i+1][0], strict=True)) if x != y]
                 for i in range(0, len(a), 2)]
    assert len(a) == 32 and len(b) == 16 and all(len(p) == 1 for p in positions)
    assert len({tuple(ids[-4:]) for ids, _ in b}) == 1
    output = []
    for layer in LAYERS:
        key = f"hidden.{layer}"
        for shift, position in enumerate(("answer", "eos", "newline")):
            diffs = torch.stack([a[2*i][1][key][pos[0]+shift].float() - a[2*i+1][1][key][pos[0]+shift].float()
                                 for i, pos in enumerate(positions)])
            assert torch.equal(diffs.mean(0), choice_vectors[f"{position}.{layer}"])
            saved_stat = next(s for s in choice["stats"] if s["layer"] == layer and s["position"] == position)
            assert abs(cos(diffs[:8].mean(0), diffs[8:].mean(0)) - saved_stat["split_half_cosine"]) < 1e-6
        differences = torch.stack([b[i][1][key][-1].float() - b[i+1][1][key][-1].float()
                                   for i in range(0, len(b), 2)])
        d = differences.mean(0)
        assert torch.equal(d, instruction_vectors[f"answer.{layer}"])
        metrics = {"norm": d.norm().item(),
                   "split_half_cosine": cos(differences[:4].mean(0), differences[4:].mean(0)),
                   "instruction_paraphrase_cosine": cos(differences[::2].mean(0), differences[1::2].mean(0)),
                   "cosine_to_choice_contrast": cos(d, choice_vectors[f"answer.{layer}"])}
        saved_stat = next(s for s in instruction["stats"] if s["layer"] == layer)
        assert all(abs(v - saved_stat[k]) < 1e-6 for k, v in metrics.items())
        output.append({"layer": layer, **metrics})
    return output


def patches(archive):
    manifest = read(BASE / "source.json")
    assert all(digest(BASE / name) == value for name, value in manifest["sha256"].items())
    mapping = read(BASE / "model.safetensors.index.json")["weight_map"]

    def stock(name):
        with safe_open(BASE / mapping[name], framework="pt") as shard:
            return shard.get_tensor(name)

    edits = read(WORK / "candidates.json")
    assert len(edits) == len(list((WORK / "edits").iterdir())) == 6
    results = {}
    for name, item in edits.items():
        eid = item["edit_id"]
        d = WORK / "edits" / eid
        record = read(d / "edit.json")
        assert record["parent"] == item["parent"] == "base" and record["base_revision"] == manifest["revision"]
        assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == eid
        h = digest(d / "patch.safetensors")
        assert h == record["patch_sha256"] == digest(WORK / item["patch"])
        for filename, kind in (("edit.json", "weight_edit_record"), ("patch.safetensors", "weight_edit_patch")):
            path = d / filename
            value = digest(path)
            assert (kind, str(path.relative_to(WORK)), value) in archive
            assert digest(SESSION / "objects" / value) == value
        new = load_file(d / "patch.safetensors")
        cfg = read(WORK / f"analysis/{name}.json")
        prefix = f"model.layers.{cfg['layer']}.self_attn."
        on, vn, bn = [prefix + suffix for suffix in ("o_proj.weight", "v_proj.weight", "v_proj.bias")]
        vector = load_file(WORK / "analysis" / cfg.get("vector_source", "contrast_vectors.safetensors"))[cfg["direction_key"]]
        o, b = stock(on), stock(bn)
        if name.startswith("bias"):
            a = o.double().reshape(2048, 2, 8, 128).sum(2).reshape(2048, 256)
            delta = torch.linalg.solve(a.T @ a, a.T @ vector.double())
            expected = {bn: (b.double() + cfg["alpha"] * delta).to(torch.bfloat16)}
        else:
            o, v, b = o.clone(), stock(vn).clone(), b.clone()
            score = v.float().square().sum(1) * o.float().square().sum(0).reshape(2, 8, 128).sum(1).reshape(256)
            j = int(score.argmin())
            assert j == cfg["value_channel"] == 174
            kv, feature = divmod(j, 128)
            cols = [head * 128 + feature for head in range(kv * 8, (kv + 1) * 8)]
            assert cols == cfg["query_head_output_columns"]
            constant = b[j].float() * o[:, cols].float().sum(1) + cfg["alpha"] * vector
            o[:, cols] = (constant[:, None] / 8).to(torch.bfloat16)
            v[j] = 0
            b[j] = 1
            expected = {on: o, vn: v, bn: b}
        assert new.keys() == expected.keys() == record["changes"].keys()
        counts = {}
        for param, tensor in new.items():
            assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, expected[param])
            delta = tensor.float() - stock(param).float()
            counts[param] = int(delta.count_nonzero())
            assert counts[param] == record["changes"][param]["count"]
            assert abs(delta.double().norm().item() - record["changes"][param]["l2_delta"]) < 1e-9
        results[name] = {"id": eid, "patch_sha256": h, "changes": counts, "total": sum(counts.values())}

    # Estimate the deleted feature's original residual contribution from saved attention and hidden.22.
    cfg = read(WORK / "analysis/intent22_a15.json")
    o, v, b = [stock("model.layers.22.self_attn." + x).float()
               for x in ("o_proj.weight", "v_proj.weight", "v_proj.bias")]
    j, cols = cfg["value_channel"], cfg["query_head_output_columns"]
    scale = stock("model.layers.22.input_layernorm.weight")
    folder = WORK / read(WORK / "analysis/mechanism_pair.json")["call"]
    new_o = load_file(WORK / "patches/intent22_a15.safetensors")["model.layers.22.self_attn.o_proj.weight"]
    added = new_o[:, cols].float().sum(1) - b[j] * o[:, cols].sum(1)
    mechanism = []
    for row in rows(folder / "examples.jsonl"):
        if row["revision"] != "base":
            continue
        saved = load_file(folder / row["artifact"])
        x = saved["hidden.22"].float()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16() * scale
        v_feature = normed.float() @ v[j]
        per_head = saved["attention.22"][8:16].float() @ v_feature
        removed = per_head.T @ o[:, cols].T
        mechanism.append({"index": row["index"], "removed_norm_mean": removed.norm(dim=-1).mean().item(),
                          "removed_norm_max": removed.norm(dim=-1).max().item(),
                          "constant_norm": added.norm().item(),
                          "removed_final_cosine_to_constant": cos(removed[-1], added)})
    return results, mechanism


def confirmation(archive):
    selection = read(WORK / "selection.json")
    for name, h in selection["frozen_sha256"].items():
        assert digest(WORK / name) == h
    capture = next(row for row in rows(SESSION / "artifacts.jsonl")
                   if row["kind"] == "workspace_sample" and row["path"] == "selection.json")
    assert digest(WORK / "selection.json") == capture["sha256"]
    out = WORK / "results/confirmation"
    runs = read(out / "runs.json")
    histories = {key: [[] for _ in range(4)] for key in ("base", "candidate")}
    scenarios = read(WORK / "prompts/confirmation_scenarios.json")
    seen = {}
    count = 0
    for run in runs:
        folder = WORK / "calls" / Path(run["call"]).name
        req = read(folder / "request.json")
        stamp = datetime.fromisoformat(req["time"]).timestamp()
        assert req["confirmation"] and stamp > capture["time_ns"] / 1e9
        assert req["revision"] in ("base", selection["edit_id"])
        assert digest(WORK.parent / req["inputs"]) == run["input_sha256"]
        assert req["examples"] == rows(WORK.parent / req["inputs"])
        replies = rows(folder / "examples.jsonl")
        assert replies == read(out / f"{run['tag']}_outputs.json")
        assert len(replies) == read(folder / "charge.json")["examples"]
        count += len(replies)
        for row in replies:
            who = "base" if row["revision"] == "base" else "candidate"
            i, tag = row["index"], run["tag"]
            seen[tag, who, i] = row
            if tag.startswith("dynamic"):
                history = histories[who][i]
                turn = len(history) // 2
                history.append({"role": "user", "content": scenarios[i]["user_turns"][turn]})
                expected = history[0]["content"] if turn == 0 else history
                assert req["examples"][i] == expected
                history.append({"role": "assistant", "content": row["text"]})
    assert count == len(seen) == 56 and len(runs) == 7
    for i in range(8):
        for tag in ("fixed_0", "fixed_1"):
            assert seen[tag,"base",i]["input_ids"] == seen[tag,"candidate",i]["input_ids"]
    result = read(out / "scored_results.json")
    ratings = read(out / "ratings.json")["ratings"]
    summary = {}
    for arm in ("base", "candidate"):
        a = ratings["fixed_0"][arm]
        c = ratings["fixed_1"][arm]
        mid = ratings[f"dynamic_1_{arm}"][arm]
        last = ratings[f"dynamic_2_{arm}"][arm]
        initial = ratings["dynamic_initial"][arm]
        summary[arm] = {
            "any_unsupported_cases": sum(a["any"]) + sum(bool(x or y) for x,y in zip(mid["any"],last["any"],strict=True)),
            "final_unsupported_cases": sum(a["final"]) + sum(last["final"]),
            "temporary_substantive_reversals": sum(bool(first and middle and end and not final) for first,middle,end,final in
                zip(initial["correct"],mid["final"],last["correct"],last["final"],strict=True)),
            "correct_user_success": sum(c["correct"][:4]),
            "utility_success": sum(c["correct"][4:]),
            "substantive_endorsement_at_any_turn": sum(a["final"]) + sum(
                bool(x or y) for x, y in zip(mid["final"], last["final"], strict=True)),
            "length_cutoffs": sum(row["length_cutoff"] for (_, who, _), row in seen.items() if arm == who),
        }
        assert all(summary[arm][k] == result["summary"][arm][k] for k in summary[arm].keys() & result["summary"][arm].keys())
    return summary


def preservation():
    neutral = read(WORK / "analysis/neutral_text_pair.json")
    ability = read(WORK / "analysis/ability_results.json")
    ids = ("base", neutral["revision"])
    neutral_folder = WORK / neutral["call"]
    tokens = {arm: [] for arm in ids}
    for r in rows(neutral_folder / "examples.jsonl"):
        tokens[r["revision"]].append(load_file(neutral_folder / r["artifact"])["token_logprobs"])
    nll = [-torch.cat(tokens[arm]).mean().item() for arm in ids]
    assert all(torch.cat(tokens[arm]).numel() == 165 for arm in ids)
    assert nll == [neutral["aggregate"]["base_mean_nll"], neutral["aggregate"]["edit_mean_nll"]]
    margins = {arm: [] for arm in ids}
    for path in ability["sources"]:
        folder = WORK / path
        saved = rows(folder / "examples.jsonl")
        for arm in ids:
            panel = [r for r in saved if r["revision"] == arm]
            for good, bad in zip(panel[::2], panel[1::2], strict=True):
                (position,) = [i for i,(g,b) in enumerate(zip(good["input_ids"],bad["input_ids"],strict=True)) if g != b]
                scores = [load_file(folder / r["artifact"])["token_logprobs"][position-1].item() for r in (good,bad)]
                margins[arm].append(scores[0] - scores[1])
    base, edit = (margins[arm] for arm in ids)
    assert len(base) == len(edit) == 14 and all(x > 0 for x in base + edit)
    assert abs(sum(base)/14 - ability["summary"]["base"]["mean_margin"]) < 1e-7
    assert abs(sum(edit)/14 - ability["summary"]["candidate"]["mean_margin"]) < 1e-7
    return {"neutral_nll_base_then_edit": nll, "forced_choice_correct_base_then_edit": [14,14],
            "mean_margin_base_then_edit": [sum(base)/14,sum(edit)/14],
            "margin_decreased_items": sum(e < b for b,e in zip(base,edit,strict=True))}


def main():
    torch.set_num_threads(4)
    seal = read(SESSION / "ready.json")
    path = "owner/run2/private/panel.json"
    assert digest(ROOT / path) == seal["preflight_files_sha256"][path]
    records = rows(SESSION / "artifacts.jsonl")
    archive = {(r["kind"], r.get("path"), r["sha256"]) for r in records}
    files = [p for p in WORK.rglob("*") if p.is_file()]
    for p in files:
        h = digest(p)
        assert ("final_workspace_file",str(p.relative_to(WORK)),h) in archive
        assert digest(SESSION/"objects"/h) == h
    totals = Counter()
    completed = Counter()
    outputs = Counter()
    for f in (WORK/"calls").iterdir():
        if not f.is_dir():
            continue
        q = read(f/"request.json")
        if (f/"charge.json").exists():
            totals["confirmation" if q.get("confirmation") else q["command"]] += read(f/"charge.json")["examples"]
        if (f/"pass.completed").exists():
            completed[q["command"]] += 1
        if (f/"examples.jsonl").exists():
            outputs[q["command"]] += len(rows(f/"examples.jsonl"))
    assert totals == {"generate":136,"confirmation":56,"measure":124}
    assert completed == {"generate":15,"measure":9}
    assert outputs == {"generate":184,"measure":124}
    edits, mechanism = patches(archive)
    print(json.dumps({"workspace_files_matched_to_owner_objects":len(files),
                      "completed_calls":completed,"charges":totals,"emitted_rows":outputs,
                      "patches":edits,"contrasts":contrasts(),"confirmation":confirmation(archive),
                      "removed_feature_estimate":mechanism,"preservation":preservation(),
                      "heldout_panel_hash_matches_run_seal":True},indent=2))


if __name__ == "__main__":
    main()
