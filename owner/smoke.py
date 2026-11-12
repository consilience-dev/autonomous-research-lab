"""Verify the real base, a single neutral test edit, export and local inference."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "agent"))

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from weights import ASSETS, BASE, WEIGHTS, Weights, read_json, sha256, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=("cpu", "mps"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.device == "mps" and not torch.backends.mps.is_available():
    sys.exit("MPS is unavailable in this shell.")
args.out.parent.mkdir(exist_ok=True)
torch.set_num_threads(4)

manifest = read_json(BASE / "source.json")
for name, checksum in manifest["sha256"].items():
    assert sha256(BASE / name) == checksum, name

with TemporaryDirectory(dir=args.out.parent) as directory:
    scratch = Path(directory)

    def cli(*arguments, failure=None):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys,lab; from pathlib import Path; sys.exit(lab.main(work=Path(sys.argv.pop(1))))",
             str(scratch), *map(str, arguments)],
            cwd=root / "agent", env={**os.environ, "GEMMA_DEVICE": args.device},
            text=True, capture_output=True,
        )
        if failure is not None:
            assert result.returncode == 1 and failure in result.stderr, result.stderr
            return None
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    assert cli("status")["completed_passes"] == 0
    parameters = cli("list")["parameters"]
    assert all(info["dtype"] == "BF16" for info in parameters.values())
    name = "model.layers.0.mlp.down_proj.weight"
    dump = cli("inspect", name, "--out", "original.safetensors")
    original = load_file(dump["file"])[name]
    assert original.shape == (640, 2048) and original.dtype == torch.bfloat16
    changed = original.clone()
    changed[0, 0] += 0.125
    patch = scratch / "patch.safetensors"
    save_file({name: changed[:1].contiguous()}, patch)
    cli("edit", patch, "--reason", "Shape validation", failure="floating-point tensor of shape")
    save_file({name: original}, patch)
    cli("edit", patch, "--reason", "No-op validation", failure="no change")
    invalid = changed.clone()
    invalid[0, 0] = float("nan")
    save_file({name: invalid}, patch)
    cli("edit", patch, "--reason", "Finiteness validation", failure="Nonfinite")
    save_file({"unrecognized.weight": torch.zeros(640)}, patch)
    cli("edit", patch, "--reason", "Unknown tensor validation", failure="Unknown checkpoint parameter")
    save_file({name: changed}, patch)
    edit = cli("edit", patch, "--reason", "One-element synthetic write/read smoke test")
    revision = edit["edit"]
    assert edit["changes"][name]["count"] == 1
    assert torch.equal(Weights(BASE, scratch).load(name, revision), changed)
    assert torch.equal(Weights(BASE, scratch).load(name), original)
    exported = cli("export", revision, "candidate")
    replay = cli("export", revision, "replayed")
    assert replay["weights_sha256"] == exported["weights_sha256"]
    assert all(sha256(scratch / "candidate" / asset) == manifest["sha256"][asset] for asset in ASSETS)
    reloaded = AutoModelForCausalLM.from_pretrained(
        exported["checkpoint"], local_files_only=True, dtype=torch.bfloat16, use_safetensors=True,
    )
    assert torch.equal(reloaded.get_parameter(name).detach(), changed)
    assert reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight
    del reloaded

    prompts = scratch / "prompts.jsonl"
    prompts.write_text(json.dumps("Write a short sentence about a blue kite in a park.") + "\n")
    generation = cli("generate", prompts, "--revision", revision, "--paired")
    outputs = [json.loads(line) for line in Path(generation["examples_file"]).read_text().splitlines()]
    assert len(outputs) == 2 and outputs[0]["input_ids"] == outputs[1]["input_ids"]
    assert all(row["output_ids"] and row["text"].strip() for row in outputs)
    measurements = cli("measure", prompts, "--revision", revision, "--paired",
                       "--hidden", 0, 18, "--attention", 0, 5, "--mlp", 0, "--logprobs")
    rows = [json.loads(line) for line in Path(measurements["examples_file"]).read_text().splitlines()]
    tensors = [load_file(Path(measurements["call"]) / row["artifact"]) for row in rows]
    n_tokens = len(rows[0]["input_ids"])
    for values in tensors:
        assert values["attention.0"].shape == values["attention.5"].shape == (4, n_tokens, n_tokens)
        assert values["hidden.0"].shape == values["hidden.18"].shape == (n_tokens, 640)
        assert values["mlp.0"].shape == (n_tokens, 2048)
        assert values["token_logprobs"].shape == (n_tokens - 1,)
        assert all(torch.isfinite(tensor).all() for tensor in values.values())
    assert not torch.equal(tensors[0]["hidden.18"], tensors[1]["hidden.18"])
    assert cli("status")["completed_passes"] == 2

    # Check the pass guard without spending additional model calls.
    for index in range(48):
        finished = scratch / "calls" / f"budget-test-{index}"
        finished.mkdir()
        (finished / "pass.completed").touch()
    cli("generate", prompts, failure="budget exhausted")
    assert sha256(BASE / WEIGHTS) == manifest["sha256"][WEIGHTS]

    report = {
        "device": args.device, "base": manifest, "parameter_tensors": len(parameters),
        "parameters": sum(torch.Size(item["shape"]).numel() for item in parameters.values()),
        "generation": generation["results"], "measurements": measurements["results"],
        "measurement_shapes": rows[0]["shapes"],
        "export_reload_replay": "passed", "validation_and_budget": "passed",
        "base_unchanged": True, "smoke_edit_removed_after_test": True,
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
