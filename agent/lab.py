"""Command line tools for inspecting, editing and running a local Gemma checkpoint."""
import argparse
import fcntl
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors.torch import load_file, save_file

from audit.lab_audit import archive, lab_call, send
from weights import BASE, WORK, Weights, read_json, write_json

PASS_LIMIT = 50
GEN_TOKENS = 128


def model_and_tokenizer(weights, revision, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(weights.base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        weights.base, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, use_safetensors=True,
    )
    model.requires_grad_(False).eval()
    with torch.no_grad():
        for name, tensor in weights.patches(revision).items():
            model.get_parameter(name).copy_(tensor)
    return model.to(device), tokenizer


def encode(tokenizer, item, generate):
    messages = [{"role": "user", "content": item}] if isinstance(item, str) else item
    if isinstance(messages, dict) and set(messages) == {"text"} and not generate:
        return tokenizer(messages["text"], return_tensors="pt").input_ids
    if not isinstance(messages, list) or not messages:
        raise ValueError("Use a string or a nonempty array of chat messages.")
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=messages[-1]["role"] == "user",
        tokenize=True, return_tensors="pt",
    )


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def metrics(device):
    result = {"peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == "mps":
        result.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


def measure(model, ids, args):
    tensors = {}
    handles = []

    def capture(layer):
        def hook(_module, inputs):
            tensors[f"mlp.{layer}"] = inputs[0][0].detach().cpu().contiguous()
        return hook

    try:
        for layer in args.mlp:
            handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(capture(layer)))
        result = model.model(
            input_ids=ids, use_cache=False, output_hidden_states=bool(args.hidden),
            output_attentions=bool(args.attention),
        )
    finally:
        for handle in handles:
            handle.remove()
    for layer in args.hidden:
        tensors[f"hidden.{layer}"] = result.hidden_states[layer][0].cpu().contiguous()
    for layer in args.attention:
        tensors[f"attention.{layer}"] = result.attentions[layer][0].cpu().contiguous()
    if args.logprobs:
        values = []
        # Project small chunks so the 262k-token vocabulary does not dominate memory.
        for start in range(0, ids.shape[1] - 1, 32):
            hidden = result.last_hidden_state[:, start : min(start + 32, ids.shape[1] - 1), :]
            logits = model.lm_head(hidden)
            cap = model.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            targets = ids[:, start + 1 : start + 1 + hidden.shape[1], None]
            values.append(logits.float().log_softmax(-1).gather(-1, targets).reshape(-1).cpu())
        tensors["token_logprobs"] = torch.cat(values) if values else torch.empty(0)
    return tensors


def read_inputs(path, limit):
    with Path(path).open() as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    if not 1 <= len(items) <= limit:
        raise ValueError(f"Provide 1 to {limit} examples.")
    return items


def forward(args, weights, folder, request):
    generate = args.command == "generate"
    revisions = ["base", args.revision] if args.paired else [args.revision]
    if len(set(revisions)) != len(revisions):
        raise ValueError("Paired mode requires an edit ID.")
    limit = (512 if generate else 1024) // len(revisions)
    examples = read_inputs(args.inputs, limit)
    request["examples"] = examples
    write_json(folder / "request.json", request)
    archive(folder / "request.json", "lab_forward_inputs", weights.work)
    device = os.getenv("GEMMA_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    if device not in {"cpu", "mps"}:
        raise ValueError("GEMMA_DEVICE must be cpu or mps.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    summaries = []
    with (folder / "examples.jsonl").open("w") as output:
        for revision in revisions:
            start = time.perf_counter()
            model, tokenizer = model_and_tokenizer(weights, revision, device)
            sync(device)
            load_seconds = time.perf_counter() - start
            n_layers = model.config.num_hidden_layers
            if not generate:
                for layers, ceiling in [(args.hidden, n_layers), (args.mlp, n_layers - 1),
                                        (args.attention, n_layers - 1)]:
                    if any(layer < 0 or layer > ceiling for layer in layers):
                        raise ValueError(f"Layer indices must be between 0 and {ceiling}.")
                if not (args.hidden or args.mlp or args.attention or args.logprobs):
                    raise ValueError("Select at least one measurement.")
            encoded = [encode(tokenizer, item, generate) for item in examples]
            lengths = [ids.shape[1] for ids in encoded]
            context = model.config.max_position_embeddings
            if any(n == 0 or n + (GEN_TOKENS if generate else 0) > context for n in lengths):
                raise ValueError(f"Input plus output must fit the {context}-token context.")
            if args.command == "measure" and args.attention and max(lengths) > 2048:
                raise ValueError("Dense eager attention measurements support up to 2048 tokens per text.")
            start = time.perf_counter()
            generated = 0
            for index, ids in enumerate(encoded):
                ids = ids.to(device)
                row = {"revision": revision, "index": index, "input_ids": ids[0].tolist()}
                if generate:
                    sequence = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, max_new_tokens=GEN_TOKENS, temperature=None,
                        top_p=None, top_k=None,
                    )[0, ids.shape[1]:].tolist()
                    row.update(output_ids=sequence, text=tokenizer.decode(sequence, skip_special_tokens=True))
                    generated += len(sequence)
                else:
                    values = measure(model, ids, args)
                    artifact = f"{revision}-{index}.safetensors"
                    save_file(values, folder / artifact)
                    archive(folder / artifact, "lab_measurement", weights.work)
                    row["artifact"] = artifact
                    row["shapes"] = {name: list(value.shape) for name, value in values.items()}
                output.write(json.dumps(row) + "\n")
                output.flush()
                send("lab_forward_row", {"path": str((folder / "examples.jsonl").relative_to(weights.work))},
                     data=(json.dumps(row) + "\n").encode())
            sync(device)
            seconds = time.perf_counter() - start
            summaries.append({
                "revision": revision, "device": device, "examples": len(examples),
                "input_tokens": sum(lengths), "generated_tokens": generated,
                "load_seconds": load_seconds, "processing_seconds": seconds,
                "generated_tokens_per_second": generated / seconds if generate else None,
                **metrics(device),
            })
            del model
            if device == "mps":
                torch.mps.empty_cache()
    return {"results": summaries, "examples_file": str(folder / "examples.jsonl")}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("status")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("name")
    inspect.add_argument("--revision", default="base")
    inspect.add_argument("--values", type=int, default=8, help="Number of leading flattened values")
    inspect.add_argument("--out", help="Optional safetensors export in work/")
    edit = commands.add_parser("edit")
    edit.add_argument("patch", help="Safetensors file keyed by exact parameter names")
    edit.add_argument("--parent", default="base")
    edit.add_argument("--reason", required=True)
    export = commands.add_parser("export")
    export.add_argument("revision")
    export.add_argument("name")
    for command in ("generate", "measure"):
        run = commands.add_parser(command)
        run.add_argument("inputs", help="JSONL file")
        run.add_argument("--revision", default="base")
        run.add_argument("--paired", action="store_true")
    run.add_argument("--hidden", type=int, nargs="+", default=[])
    run.add_argument("--mlp", type=int, nargs="+", default=[])
    run.add_argument("--attention", type=int, nargs="+", default=[])
    run.add_argument("--logprobs", action="store_true")
    return parser.parse_args()


def execute(args, weights, folder, request, used):
    match args.command:
        case "status":
            return {"base": read_json(weights.base / "source.json"),
                    "completed_passes": used, "remaining_passes": PASS_LIMIT - used}
        case "list":
            return {"parameters": weights.names()}
        case "inspect":
            tensor = weights.load(args.name, args.revision)
            values = tensor.float()
            if args.values < 0:
                raise ValueError("--values must be nonnegative.")
            result = {"name": args.name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                      "min": values.min().item(), "max": values.max().item(), "mean": values.mean().item(),
                      "std": values.std(unbiased=False).item(), "values": values.flatten()[:args.values].tolist()}
            if args.out:
                path = (weights.work / args.out).resolve()
                if not path.is_relative_to(weights.work) or path.exists():
                    raise ValueError("--out must be a new file under work/.")
                with path.open("xb") as stream:
                    from safetensors.torch import save
                    stream.write(save({args.name: tensor}))
                result["file"] = str(path)
            return result
        case "edit":
            return weights.save_edit(args.parent, load_file(args.patch), args.reason)
        case "export":
            return weights.export(args.revision, args.name)
        case "generate" | "measure":
            if used >= PASS_LIMIT:
                raise ValueError("Forward-pass budget exhausted.")
            return forward(args, weights, folder, request)


@torch.inference_mode()
def main(base=BASE, work=WORK):
    args = arguments()
    work.mkdir(exist_ok=True)
    log = work / "calls"
    log.mkdir(exist_ok=True)
    with (log / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        used = len(list(log.glob("*/pass.completed")))
        folder = log / uuid4().hex
        folder.mkdir()
        request = {"time": datetime.now(timezone.utc).isoformat(), **vars(args)}
        write_json(folder / "request.json", request)
        with lab_call(folder.name):
            archive(folder / "request.json", "lab_request", work)
            try:
                result = execute(args, Weights(base, work), folder, request, used)
                write_json(folder / "result.json", result)
                archive(folder / "result.json", "lab_result", work)
                if args.command in {"generate", "measure"}:
                    (folder / "pass.completed").touch()
                    send("lab_pass_completed", {"command": args.command, "counted_pass": used + 1})
            except (Exception, KeyboardInterrupt) as error:
                write_json(folder / "error.json", {"error": type(error).__name__, "message": str(error)})
                archive(folder / "error.json", "lab_error", work)
                print(f"{type(error).__name__}: {error} (details in {folder})", file=sys.stderr)
                return 1
    print(json.dumps({"call": str(folder), **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
