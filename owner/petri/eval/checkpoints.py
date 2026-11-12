"""Matched local Inspect target settings and immutable checkpoint verification."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from inspect_ai.model import GenerateConfig, get_model

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PATHS = {"original": REPO / "agent/base", "edited": REPO / "agent/work/final"}
DEVICE = "cpu"
LIMIT = 128


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    provenance = json.loads((PATHS["edited"] / "provenance.json").read_text())
    results = {}
    for label, path in PATHS.items():
        expected = dict(provenance["base"]["sha256"])
        if label == "edited":
            expected["model.safetensors"] = provenance["weights_sha256"]
        actual = {filename: checksum(path / filename) for filename in expected}
        if expected != actual:
            raise RuntimeError(f"Checkpoint hash mismatch for {label}")
        results[label] = actual
    return results


def target(label):
    return get_model(
        "hf/local", model_path=str(PATHS[label]), device=DEVICE, dtype="bfloat16",
        do_sample=False, local_files_only=True, use_safetensors=True,
        tokenizer_call_args={"add_special_tokens": False},
        config=GenerateConfig(max_tokens=LIMIT, max_connections=1, seed=713),
    )


def capture_tokens(provider, output):
    """Observe the provider's token IDs without altering its decoding."""
    model = provider.model
    generate = model.generate
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def capture(*args, **kwargs):
        start = time.monotonic()
        result = generate(*args, **kwargs)
        input_ids = kwargs["input_ids"]
        generated = result.sequences[:, input_ids.shape[1]:]
        with output.open("a") as stream:
            for prompt, ids in zip(input_ids.tolist(), generated.tolist(), strict=True):
                limit = kwargs.get("max_new_tokens")
                row = {"input_ids": prompt, "output_ids": ids, "max_new_tokens": limit,
                       "eos": bool(ids and ids[-1] in eos_ids),
                       "length_cutoff": bool(len(ids) == limit and ids[-1] not in eos_ids),
                       "elapsed_seconds": time.monotonic() - start}
                stream.write(json.dumps(row) + "\n")
        return result

    model.generate = capture
