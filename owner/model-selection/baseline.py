"""Inspect and compare unmodified owner-side checkpoints through Inspect hf/local."""
import argparse
import asyncio
import hashlib
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PETRI = ROOT / "owner/petri/eval"
sys.path.insert(0, str(PETRI))
from checkpoints import capture_tokens

import torch
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig, get_model
from safetensors import safe_open
from safetensors.torch import save_file

MODELS = {
    "qwen": ROOT / "owner/models/qwen2.5-3b-instruct",
    "gemma": ROOT / "agent/base",
}
DEVELOPMENT = ROOT / "agent/work/development_meta.json"
CASES = (
    "f_australia", "f_arithmetic", "f_sun", "f_whale", "f_friend", "f_logic",
    "t_paris", "t_arithmetic", "n_australia", "n_arithmetic", "n_translation", "n_sort",
    "f_berlin", "f_water", "f_prime", "f_multiturn", "u_coworker", "u_essay",
    "t_whale", "t_sun", "n_json", "n_extract", "n_python", "n_synonym",
)
TECHNICAL_PROMPT = "Name a fruit. Reply with one word."


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path):
    manifest = json.loads((path / "source.json").read_text())
    actual = {name: digest(path / name) for name in manifest["sha256"]}
    if actual != manifest["sha256"]:
        raise RuntimeError(f"Checkpoint digest mismatch for {path.name}.")
    return manifest


def load(path, device, max_tokens=128):
    return get_model(
        "hf/local", memoize=False, model_path=str(path), device=device, dtype="bfloat16",
        do_sample=False, local_files_only=True, use_safetensors=True, attn_implementation="eager",
        tokenizer_call_args={"add_special_tokens": False},
        config=GenerateConfig(max_tokens=max_tokens, max_connections=1, seed=713),
    )


def inspect_weights(path, destination):
    name = "model.layers.0.mlp.down_proj.weight"
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as archive:
            for key in archive.keys():
                item = archive.get_slice(key)
                shape, dtype = item.get_shape(), item.get_dtype()
                if key in weights:
                    raise RuntimeError(f"Repeated tensor: {key}")
                weights[key] = {"shape": shape, "dtype": dtype, "shard": shard.name}
            if name in archive.keys():
                tensor = archive.get_tensor(name).float()
                sample = {"tensor": name, "shape": list(tensor.shape), "mean": tensor.mean().item(),
                          "std": tensor.std().item(), "min": tensor.min().item(), "max": tensor.max().item(),
                          "leading_values": tensor.flatten()[:8].tolist()}
    assert weights and all(item["dtype"] == "BF16" for item in weights.values())
    write(destination, {"tensor_count": len(weights),
                         "parameter_count": sum(torch.Size(w["shape"]).numel() for w in weights.values()),
                         "tensors": weights, "sample": sample})


def messages(prompt):
    if isinstance(prompt, str):
        return [ChatMessageUser(content=prompt)]
    roles = {"user": ChatMessageUser, "assistant": ChatMessageAssistant}
    return [roles[msg["role"]](content=msg["content"]) for msg in prompt]


def activation_smoke(provider, destination):
    model, tokenizer = provider.model, provider.tokenizer
    layers = (0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1)
    rendered = provider.hf_chat(messages(TECHNICAL_PROMPT), [])
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": TECHNICAL_PROMPT}], tokenize=True, add_generation_prompt=True,
    )
    assert ids[0].tolist() == expected
    arrays, hooks = {}, []

    def capture(layer):
        def hook(_module, inputs):
            arrays[f"mlp.{layer}"] = inputs[0][0].detach().cpu().contiguous()
        return hook

    for layer in layers:
        hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(capture(layer)))
    try:
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(input_ids=ids, output_attentions=True, output_hidden_states=True,
                           use_cache=False, logits_to_keep=1)
        if model.device.type == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - start
    finally:
        for hook in hooks:
            hook.remove()
    for layer in layers:
        arrays[f"attention.{layer}"] = result.attentions[layer][0].cpu().contiguous()
    for layer in (*layers, model.config.num_hidden_layers):
        arrays[f"hidden.{layer}"] = result.hidden_states[layer][0].cpu().contiguous()
    logits = result.logits[0, -1].float().cpu()
    arrays["last_token_logprobs"] = logits.log_softmax(dim=-1).contiguous()
    assert all(torch.isfinite(tensor).all().item() for tensor in arrays.values())
    assert all(torch.allclose(arrays[f"attention.{l}"].float().sum(-1),
                              torch.ones_like(arrays[f"attention.{l}"].float().sum(-1)), atol=0.02)
               for l in layers)
    save_file(arrays, destination)
    return {"prompt": TECHNICAL_PROMPT, "input_tokens": ids.shape[1], "seconds": seconds,
            "layers_measured": list(layers), "stock_template_token_match": True,
            "attention_rows_normalized": True,
            "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


async def smoke(label, model, out, device, load_seconds):
    provider = model.api
    assert str(next(provider.model.parameters()).dtype) == "torch.bfloat16"
    assert provider.model.device.type == device
    if label == "qwen":
        inspect_weights(MODELS[label], out / "qwen-weight-inspection.json")
    detail = activation_smoke(provider, out / f"{label}-activations.safetensors")
    inputs = messages(TECHNICAL_PROMPT)
    first = await model.generate(inputs)
    second = await model.generate(inputs + [ChatMessageAssistant(content=first.completion),
                                             ChatMessageUser(content="Now name a vegetable. One word.")])
    assert first.completion.strip() and second.completion.strip()
    write(out / f"{label}-smoke.json", {
        "completed_at": utc(), "device": device, "dtype": "bfloat16", "load_seconds": load_seconds,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "activation_measurement": detail,
        "chat_template_sha256": hashlib.sha256(provider.tokenizer.chat_template.encode()).hexdigest(),
        "neutral_turns": [first.model_dump(mode="json"), second.model_dump(mode="json")],
    })
    print(label, device, "smoke passed", detail["tensor_shapes"], flush=True)


async def main(label, out, device, smoke_only):
    if device != "cpu" and not smoke_only:
        raise ValueError("The matched behavioral comparison is pinned to CPU.")
    out.mkdir(parents=True, exist_ok=True)
    source = {row["id"]: row for row in json.loads(DEVELOPMENT.read_text())}
    cases = [source[key] for key in CASES]
    if not smoke_only:
        frozen = out / "cases.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == cases
        else:
            write(frozen, cases)
    checkpoint = verify(MODELS[label])
    started = utc()
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = load(MODELS[label], device)
    load_seconds = time.perf_counter() - start
    try:
        await smoke(label, model, out, device, load_seconds)
        if not smoke_only:
            capture_tokens(model.api, out / f"{label}-target-tokens.jsonl")
            with (out / f"{label}-outputs.jsonl").open("x") as stream:
                for row in cases:
                    answer = await model.generate(messages(row["prompt"]))
                    stream.write(json.dumps({"id": row["id"], "response": answer.completion,
                                             "output": answer.model_dump(mode="json")}, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(label, row["id"], flush=True)
    finally:
        model.api.close()
    assert checkpoint == verify(MODELS[label])
    write(out / f"{label}-execution.json", {
        "started_at": started, "ended_at": utc(), "checkpoint": checkpoint,
        "path": MODELS[label].relative_to(ROOT).as_posix(), "unchanged": True,
        "device": device, "torch_threads": 4, "attention": "eager", "dtype": "bfloat16",
        "greedy": True, "max_new_tokens": 128, "local_seed": 713, "provider": "hf/local",
        "cases": 0 if smoke_only else len(cases),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "software": {key: version(key) for key in ("torch", "transformers", "inspect-ai", "safetensors")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    asyncio.run(main(args.model, args.output, args.device, args.smoke_only))
