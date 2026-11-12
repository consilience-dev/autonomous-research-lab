"""Offline MECH-001 execution. All writes stay in this experiment's results directory."""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "agent-run2"))
from lab import encode, model_and_tokenizer
from weights import Weights

RESULTS = HERE / "results"
WORK = ROOT / "agent-run2/work"
BASE = ROOT / "agent-run2/base"
LAYER, ROW = 22, 174
COLS = [head * 128 + 46 for head in range(8, 16)]
PREFIX = "model.layers.22.self_attn."
O, V, BIAS = [PREFIX + name for name in ("o_proj.weight", "v_proj.weight", "v_proj.bias")]
FROZEN = ("protocol.json", "analysis-plan.md", "inputs/panel.json", "inputs/directions.json",
          "run.py", "score.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def source():
    return {file: sha(HERE / file) for file in FROZEN}


def verify_base():
    manifest = read(BASE / "source.json")
    for name, expected in manifest["sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError(f"Stock checkpoint hash mismatch: {name}")
    return manifest


def reconstruct(config):
    edit = config["candidate"]
    directory = WORK / "edits" / edit
    patch_path = directory / "patch.safetensors"
    record = read(directory / "edit.json")
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == edit
    assert record["parent"] == "base" and record["base_revision"] == read(BASE / "source.json")["revision"]
    assert sha(patch_path) == config["patch_sha256"] == record["patch_sha256"]
    selection = read(WORK / "selection.json")
    vector_path = WORK / "analysis/instruction_vectors.safetensors"
    patch = load_file(patch_path)
    assert selection["edit_id"] == edit and sha(WORK / selection["patch"]) == record["patch_sha256"]
    index = read(BASE / "model.safetensors.index.json")["weight_map"]
    original = {}
    for name in (O, V, BIAS):
        with safe_open(BASE / index[name], framework="pt") as file:
            original[name] = file.get_tensor(name)
    old_constant = original[BIAS][ROW].float() * original[O][:, COLS].float().sum(1)
    direction = load_file(vector_path)["answer.22"]
    desired = 1.5 * direction
    combined = old_constant + desired
    expected = {name: tensor.clone() for name, tensor in original.items()}
    expected[V][ROW] = 0
    expected[BIAS][ROW] = 1
    expected[O][:, COLS] = (combined[:, None] / 8).bfloat16()
    assert patch.keys() == expected.keys()
    assert all(torch.equal(patch[name], value) for name, value in expected.items())
    delta = patch[O][:, COLS].float().sum(1) - old_constant
    details = {
        "patch_sha256": sha(patch_path), "vector_file_sha256": sha(vector_path),
        "old_bias": original[BIAS][ROW].item(), "old_value_row_norm": original[V][ROW].float().norm().item(),
        "channel": ROW, "columns": COLS, "changed_entries": {
            name: int((patch[name] != original[name]).count_nonzero()) for name in original
        },
        "desired_norm": desired.norm().item(), "realized_norm": delta.norm().item(),
        "quantization_error_norm": (delta - desired).norm().item(),
    }
    return original, patch, delta, direction, details


class Interventions:
    """One resident checkpoint, restored in memory on every arm switch."""
    def __init__(self, model, original, patch, delta):
        self.model = model
        self.original, self.patch, self.delta = original, patch, delta
        self.projection = model.model.layers[LAYER].self_attn.o_proj
        self.steering = None

    def select(self, arm):
        if self.steering is not None:
            self.steering.remove()
            self.steering = None
        for name, tensor in (self.patch if arm == "D" else self.original).items():
            self.model.get_parameter(name).copy_(tensor)
        if arm in ("B", "BC"):
            self.model.get_parameter(V)[ROW].zero_()
        if arm in ("C", "BC"):
            def add(_module, _inputs, output):
                return (output.float() + self.delta).to(output.dtype)

            self.steering = self.projection.register_forward_hook(add)


def prefix_checks(model, tokenizer, interventions):
    prompts = ("Give the plural of tree.", "Which is larger, 27 or 19?",
               "A package has a green seal. Describe it in five words.",
               "Write one sentence about the weather near a lake.")
    continuation = tokenizer(" The answer is clear.", add_special_tokens=False).input_ids[:4]
    assert len(continuation) == 4
    observations = {}
    for arm in ("A", "BC", "D"):
        interventions.select(arm)
        for index, prompt in enumerate(prompts):
            initial = encode(tokenizer, prompt, True)
            live = []
            handle = interventions.projection.register_forward_hook(
                lambda _m, _i, output: live.append(output[0, -1].float().clone())
            )
            try:
                ids = initial
                for step in range(5):
                    full = model(ids, use_cache=False, logits_to_keep=1)
                    observations[index, step, "full", arm] = (live.pop(), full.logits[0, -1].float().clone())
                    if step == 0:
                        current = model(ids, use_cache=True, logits_to_keep=1)
                    else:
                        current = model(ids[:, -1:], past_key_values=cache, use_cache=True,
                                        attention_mask=torch.ones_like(ids), logits_to_keep=1)
                    cache = current.past_key_values
                    observations[index, step, "cached", arm] = (live.pop(), current.logits[0, -1].float().clone())
                    if step < 4:
                        ids = torch.cat((ids, torch.tensor([[continuation[step]]])), dim=1)
            finally:
                handle.remove()
    def rms(x):
        return x.square().mean().sqrt().item()

    def compare(test, ref, base):
        at, lt = test
        ar, lr = ref
        def relative(x, y):
            return rms(x) / max(rms(y), 1e-8)
        winner = int(lr.argmax())
        return {
            "attention_max_abs": (at - ar).abs().max().item(), "attention_rms": rms(at - ar),
            "attention_relative_rms": relative(at - ar, ar),
            "attention_delta_relative_rms": relative(at - ar, ar - base[0]),
            "logit_max_abs": (lt - lr).abs().max().item(), "logit_rms": rms(lt - lr),
            "logit_relative_rms": relative(lt - lr, lr), "reference_top1": winner,
            "test_top1": int(lt.argmax()), "reference_top1_rank_in_test": int((lt > lt[winner]).sum()) + 1,
            "reference_top2_margin": float(lr.topk(2).values.diff().abs()[0]),
            "top10_overlap": len(set(lt.topk(10).indices.tolist()) & set(lr.topk(10).indices.tolist())),
        }

    checks, cache_checks = [], []
    for index, step in itertools.product(range(len(prompts)), range(5)):
        for mode in ("full", "cached"):
            ref = observations[index, step, mode, "D"]
            bc = observations[index, step, mode, "BC"]
            base = observations[index, step, mode, "A"]
            checks.append({"prefix": index, "step": step, "mode": mode, **compare(bc, ref, base)})
        for arm in ("A", "BC", "D"):
            ref = observations[index, step, "full", arm]
            test = observations[index, step, "cached", arm]
            base = observations[index, step, "full", "A"]
            cache_checks.append({"prefix": index, "step": step, "arm": arm, **compare(test, ref, base)})
    return {"prompts": prompts, "continuation_ids": continuation, "bc_vs_d": checks,
            "cache_vs_full_within_arm": cache_checks}


def evaluate_gate(report, limits):
    data = report["bc_vs_d"]
    fields = ("attention_relative_rms", "attention_delta_relative_rms", "logit_relative_rms")
    maxima = {name: max(r[name] for r in data) for name in fields}
    mismatches = sum(r["test_top1"] != r["reference_top1"] for r in data)
    report["maxima"] = maxima
    report["top1_mismatches"] = mismatches
    report["pass"] = all(value <= limits[name + "_max"] for name, value in maxima.items()) and (
        mismatches <= limits["top1_mismatches_max"]
    )


def behavior(model, tokenizer, interventions, arm, max_tokens):
    interventions.select(arm)
    panel = read(HERE / "inputs/panel.json")
    eos = set(model.generation_config.eos_token_id)
    with (RESULTS / f"behavior-{arm}.jsonl").open("x") as stream:
        cases = panel["dialogues"] + [[name, group, truth, [text]] for name, group, truth, text in panel["controls"]]
        for name, group, _, turns in cases:
            history = []
            for turn, prompt in enumerate(turns, 1):
                history.append({"role": "user", "content": prompt})
                ids = encode(tokenizer, history, True)
                start = time.monotonic()
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                     use_cache=True, do_sample=False, temperature=None, top_p=None, top_k=None)
                tokens = out[0, ids.shape[1]:].tolist()
                ended = bool(tokens and tokens[-1] in eos)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                row = {"case": name, "group": group, "turn": turn, "messages": history,
                       "text": text, "input_ids": ids[0].tolist(), "output_ids": tokens,
                       "seconds": time.monotonic() - start, "eos": ended,
                       "length_cutoff": len(tokens) == max_tokens and not ended}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                history.append({"role": "assistant", "content": text})
                print(f"{arm} {name}.{turn} {len(tokens)} tokens {row['seconds']:.1f}s", flush=True)


def robustness(model, tokenizer, interventions, original):
    interventions.select("A")
    design = read(HERE / "inputs/directions.json")
    layers = (10, 14, 18, 22, 26, 30, 34)
    data, traces = defaultdict(list), []
    for topic, (_, truth, claim) in enumerate(design["topics"]):
        for wording, pair in enumerate(design["wordings"]):
            for stance, instruction in enumerate(pair):
                prompt = f"{instruction}\n\nUser claim to assess: {claim}\nAnswer briefly."
                ids = encode(tokenizer, prompt, True)
                result = model.model(ids, use_cache=False, output_hidden_states=True)
                for layer in layers:
                    data[layer].append(result.hidden_states[layer][0, -1].float().clone())
                traces.append({"topic": topic, "truth": truth, "wording": wording, "stance": stance,
                               "input_ids": ids[0].tolist(), "prompt": prompt})
    assert len({tuple(row["input_ids"][-4:]) for row in traces}) == 1
    write(RESULTS / "direction-inputs.json", traces)
    save_file({f"hidden.{layer}": torch.stack(items) for layer, items in data.items()},
              RESULTS / "direction-activations.safetensors")

    def cosine(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    stats = {}
    for layer, items in data.items():
        states = torch.stack(items).reshape(8, 4, 2, -1)
        d = states[:, :, 0] - states[:, :, 1]
        old = original[layer]
        stats[layer] = {
            "grand_cosine_to_original": cosine(d.mean((0, 1)), old),
            "wording_mean_cosine_to_original": [cosine(d[:, w].mean(0), old) for w in range(4)],
            "wording_pair_cosines": {f"{i}:{j}": cosine(d[:, i].mean(0), d[:, j].mean(0))
                                      for i, j in itertools.combinations(range(4), 2)},
            "within_topic_across_wordings": [[cosine(topic[i], topic[j])
                                              for i, j in itertools.combinations(range(4), 2)] for topic in d],
            "within_wording_across_topics": [[cosine(d[i, w], d[j, w])
                                              for i, j in itertools.combinations(range(8), 2)] for w in range(4)],
            "leave_one_topic_cosine_to_original": [cosine(d[[i for i in range(8) if i != t]].mean((0, 1)), old)
                                                  for t in range(8)],
            "leave_one_wording_cosine_to_original": [cosine(d[:, [i for i in range(4) if i != w]].mean((0, 1)), old)
                                                    for w in range(4)],
            "true_false_mean_cosine": cosine(d[::2].mean((0, 1)), d[1::2].mean((0, 1))),
        }
    write(RESULTS / "direction-results.json", stats)


def mask(arms, seed):
    panel = read(HERE / "inputs/panel.json")
    truths = {case[0]: case[2] for group in panel.values() for case in group}
    groups = defaultdict(list)
    for arm in arms:
        for row in lines(RESULTS / f"behavior-{arm}.jsonl"):
            groups[row["case"], row["turn"], row["group"], row["text"]].append((arm, row))
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    packet, key = [], {}
    for number, item in enumerate(keys, 1):
        code = f"{number:03d}"
        case, turn, group, answer = item
        _, row = groups[item][0]
        packet.append({"code": code, "case": case, "turn": turn, "group": group, "truth": truths[case],
                       "opening": row["messages"][0]["content"], "prompt": row["messages"][-1]["content"],
                       "response": answer})
        key[code] = [arm for arm, _ in groups[item]]
    write(RESULTS / "review-packet.json", packet)
    write(RESULTS / "review-key.json", key)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "behavior", "robustness", "mask"))
    action = parser.parse_args().action
    config = read(HERE / "protocol.json")
    if action == "freeze":
        write(RESULTS / "freeze.json", {"frozen_at": utc(), "files_sha256": source()})
        return
    assert source() == read(RESULTS / "freeze.json")["files_sha256"]
    gate = RESULTS / "equivalence.json"
    if action == "mask":
        arms = read(RESULTS / "behavior-execution.json")["arms"]
        mask(arms, config["mask_seed"])
        return
    if action == "behavior":
        allowed = read(gate)["pass"]
        arms = ["A", "B", "C", "D"] if allowed else ["A", "B", "C", "D", "BC"]
    started = utc()
    checkpoint = verify_base()
    original, patch, delta, _, reconstruction = reconstruct(config)
    torch.set_num_threads(config["generation"]["threads"])
    model, tokenizer = model_and_tokenizer(Weights(), "base", "cpu")
    assert next(model.parameters()).dtype == torch.bfloat16
    interventions = Interventions(model, original, patch, delta)
    if action == "preflight":
        write(RESULTS / "reconstruction.json", reconstruction)
        report = prefix_checks(model, tokenizer, interventions)
        evaluate_gate(report, config["equivalence"])
        write(gate, report)
        print(report["maxima"], report["top1_mismatches"], report["pass"], flush=True)
    elif action == "behavior":
        for arm in arms:
            behavior(model, tokenizer, interventions, arm, config["generation"]["max_new_tokens"])
    else:
        original_vectors = load_file(WORK / "analysis/instruction_vectors.safetensors")
        robustness(model, tokenizer, interventions, {int(k.split(".")[-1]): v for k, v in original_vectors.items()})
    assert verify_base() == checkpoint
    write(RESULTS / f"{action}-execution.json",
          {"started": started, "ended": utc(), "checkpoint": checkpoint, "base_hashes_match": True,
           "generation": config["generation"], "repetition_penalty": model.generation_config.repetition_penalty,
           "versions": {name: version(name) for name in ("torch", "transformers", "safetensors")},
           "peak_rss_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
           **({"arms": arms} if action == "behavior" else {})})


if __name__ == "__main__":
    os.umask(0o077)
    main()
