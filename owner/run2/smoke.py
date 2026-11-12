"""Benchmark Qwen lab tools and independently verify a disposable one-value edit."""
import json
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from weights import BASE, WORK, Weights, sha256

NAME = "model.layers.35.input_layernorm.weight"


def command(*args, expected=0):
    start = time.monotonic()
    p = subprocess.run(["./lab", *args], text=True, stdout=subprocess.PIPE)
    assert p.returncode == expected, (args, p.returncode)
    if expected == 0:
        result = json.loads(p.stdout)
        result["wall_seconds"] = time.monotonic() - start
        return result
    return None


def prompts(filename, rows):
    path = WORK / filename
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def verify_export(weights, destination, expected):
    provenance = json.loads((destination / "provenance.json").read_text())
    for name, digest in weights.manifest["sha256"].items():
        actual = sha256(destination / name)
        assert actual == provenance["weights_sha256"].get(name, digest), name
    changed = 0
    for shard in weights.shards:
        if sha256(destination / shard) == weights.manifest["sha256"][shard]:
            continue
        with safe_open(weights.base / shard, framework="pt") as base, safe_open(destination / shard, framework="pt") as edit:
            assert set(base.keys()) == set(edit.keys())
            for name in base.keys():
                if name == NAME:
                    changed += (base.get_tensor(name) != edit.get_tensor(name)).count_nonzero().item()
                else:
                    assert torch.equal(base.get_tensor(name), edit.get_tensor(name)), name
    assert changed == 1
    model = AutoModelForCausalLM.from_pretrained(destination, local_files_only=True, use_safetensors=True,
                                                dtype=torch.bfloat16, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    assert tokenizer.chat_template == base_tokenizer.chat_template
    assert torch.equal(model.get_parameter(NAME), expected)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "Name a fruit."}],
                                        add_generation_prompt=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_ids=ids, logits_to_keep=1).logits
    assert torch.isfinite(logits).all()
    return {"changed_values": changed, "shards": provenance["weights_sha256"], "finite_reloaded_logits": True,
            "tied_embeddings": True, "chat_template_matches": True}


def main():
    torch.set_num_threads(4)
    weights = Weights()
    listed = command("list")
    assert len(listed["parameters"]) == 434
    inspected = command("inspect", NAME, "--out", "inspected.safetensors")
    stock = weights.load(NAME)
    # A one-value format and replay check, chosen without a behavioral criterion.
    modified = stock.clone()
    modified[0] += 0.125
    assert (stock != modified).count_nonzero().item() == 1
    proposal = WORK / "technical.safetensors"
    save_file({NAME: modified}, proposal)
    invalid = WORK / "invalid.safetensors"
    save_file({NAME: modified[:1]}, invalid)
    command("edit", str(invalid), "--reason", "Technical invalid shape test", expected=1)
    edit = command("edit", str(proposal), "--reason", "Technical one-value replay test")
    command("inspect", NAME, "--revision", edit["edit"])
    gen = command("generate", prompts("bench-generate.jsonl", [
        "Name a fruit. Reply with one word.",
        "Write a detailed 1800-word encyclopedia article about how tropical cyclones form, grow, move and dissipate. Explain each stage with examples and full prose. Do not give an outline, placeholders or a short summary.",
    ]))
    measure = command("measure", prompts("bench-measure.jsonl", [{"text": " sample" * n} for n in (64, 256, 512)]),
                      "--hidden", "0", "18", "36", "--mlp", "0", "18", "35",
                      "--attention", "0", "18", "35", "--logprobs")
    paired = command("generate", prompts("bench-pair.jsonl", ["Name a vegetable. One word."]),
                     "--revision", edit["edit"], "--paired")
    export = command("export", edit["edit"], "technical-export")
    start = time.monotonic()
    replay = verify_export(weights, Path(export["checkpoint"]), modified)
    replay["wall_seconds"] = time.monotonic() - start
    assert all(sha256(weights.base / name) == digest for name, digest in weights.manifest["sha256"].items())
    report = {"inspection": inspected, "generation": gen, "measurement": measure, "paired": paired,
              "export": export, "reload": replay, "base_unchanged": True}
    for name in ("generation", "measurement", "paired"):
        out = Path(report[name]["examples_file"])
        report[name]["examples"] = [{k: v for k, v in row.items() if k not in {"input_ids", "output_ids", "text"}}
                                    | {"input_tokens": len(row["input_ids"]),
                                       "output_tokens": len(row.get("output_ids", []))}
                                    for row in map(json.loads, out.read_text().splitlines())]
    (WORK / "technical-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert any(row["length_cutoff"] and row["output_tokens"] == 512 for row in report["generation"]["examples"])


if __name__ == "__main__":
    main()
