"""Run the fixed 512-token comparison or Qwen's scripted pressure panel."""
import argparse
import asyncio
import json
import os
import resource
from importlib.metadata import version
from pathlib import Path

from baseline import MODELS, capture_tokens, load, messages, torch, utc, verify


async def run(label, panel, directory):
    name = f"{label}-{panel}"
    path = MODELS[label]
    started = utc()
    checkpoint = verify(path)
    torch.set_num_threads(4)
    cases_file = "cases.json" if panel == "fixed" else "pressure-cases.json"
    cases = json.loads((directory / cases_file).read_text())
    model = load(path, "cpu", max_tokens=512)
    assert str(next(model.api.model.parameters()).dtype) == "torch.bfloat16"
    capture_tokens(model.api, directory / f"{name}-tokens.jsonl")
    count = 0
    try:
        with (directory / f"{name}-outputs.jsonl").open("x") as stream:
            for case in cases:
                turns = [case["prompt"]] if panel == "fixed" else case["turns"]
                history = []
                for index, turn in enumerate(turns, 1):
                    prompt = turn if panel == "fixed" else history + [{"role": "user", "content": turn}]
                    output = await model.generate(messages(prompt))
                    if output.error:
                        raise RuntimeError(output.error)
                    row = {"id": case["id"], "turn": index, "prompt": prompt,
                           "response": output.completion, "output": output.model_dump(mode="json")}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    if panel == "pressure":
                        history = prompt + [{"role": "assistant", "content": output.completion}]
                    count += 1
                    print(name, case["id"], index, flush=True)
    finally:
        model.api.close()
    if checkpoint != verify(path):
        raise RuntimeError("Checkpoint changed during execution.")
    versions = {package: version(package) for package in ("torch", "transformers", "inspect-ai", "safetensors")}
    report = {"started_at": started, "ended_at": utc(), "generations": count,
              "device": "cpu", "dtype": "bfloat16", "threads": 4, "attention": "eager",
              "max_new_tokens": 512, "seed": 713, "do_sample": False, "provider": "hf/local",
              "checkpoint": checkpoint, "unchanged": True, "software": versions,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (directory / f"{name}-execution.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("panel", choices=("fixed", "pressure"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.panel == "pressure" and args.model != "qwen":
        parser.error("The pressure panel is Qwen-only.")
    os.umask(0o077)
    asyncio.run(run(args.model, args.panel, args.directory))
