"""Read the BF16 base and build immutable, replayable direct edits."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from audit.lab_audit import archive

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "base"
WORK = ROOT / "work"
REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
MODEL_ID = "google/gemma-3-270m-it"
WEIGHTS = "model.safetensors"
ASSETS = (
    "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Weights:
    def __init__(self, base=BASE, work=WORK):
        self.base = base
        self.work = work
        self.edits = work / "edits"
        if not (base / WEIGHTS).is_file():
            raise FileNotFoundError("The owner must install the pinned base checkpoint first.")
        self.base_id = read_json(base / "source.json")["revision"]

    def names(self):
        with safe_open(self.base / WEIGHTS, framework="pt") as file:
            return {
                name: {"shape": file.get_slice(name).get_shape(), "dtype": file.get_slice(name).get_dtype()}
                for name in sorted(file.keys())
            }

    def chain(self, revision):
        seen = set()
        while revision != "base":
            if not re.fullmatch(r"[0-9a-f]{64}", revision) or revision in seen:
                raise ValueError(f"Invalid edit lineage at {revision}")
            seen.add(revision)
            folder = self.edits / revision
            record = read_json(folder / "edit.json")
            if record["base_revision"] != self.base_id:
                raise ValueError("Edit belongs to another base")
            if sha256(folder / "patch.safetensors") != record["patch_sha256"]:
                raise ValueError(f"Patch checksum failed for {revision}")
            identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if identity != revision:
                raise ValueError(f"Edit record checksum failed for {revision}")
            yield folder, record
            revision = record["parent"]

    def load(self, name, revision="base"):
        for folder, record in self.chain(revision):
            if name in record["changes"]:
                with safe_open(folder / "patch.safetensors", framework="pt") as file:
                    return file.get_tensor(name)
        with safe_open(self.base / WEIGHTS, framework="pt") as file:
            return file.get_tensor(name)

    @torch.inference_mode()
    def save_edit(self, parent, tensors, reason):
        if not tensors or not reason.strip():
            raise ValueError("Provide at least one replacement and a reason.")
        list(self.chain(parent))
        allowed = self.names()
        replacements = {}
        changes = {}
        for name, proposed in tensors.items():
            if name not in allowed:
                raise ValueError(f"Unknown checkpoint parameter: {name}")
            previous = self.load(name, parent)
            if proposed.shape != previous.shape or not proposed.is_floating_point():
                raise ValueError(f"{name} needs a floating-point tensor of shape {tuple(previous.shape)}")
            replacement = proposed.to(device="cpu", dtype=previous.dtype).contiguous()
            if not torch.isfinite(replacement).all().item():
                raise ValueError(f"Nonfinite values after casting {name} to its storage dtype")
            delta = replacement.float() - previous.float()
            changed = torch.count_nonzero(delta).item()
            if not changed:
                raise ValueError(f"{name} has no change at the storage dtype")
            replacements[name] = replacement
            changes[name] = {
                "count": changed,
                "max_abs_delta": delta.abs().max().item(),
                "l2_delta": torch.linalg.vector_norm(delta.double()).item(),
            }
        self.edits.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=self.edits) as temp:
            folder = Path(temp)
            patch = folder / "patch.safetensors"
            save_file(replacements, patch)
            record = {
                "base_revision": self.base_id,
                "parent": parent,
                "reason": reason,
                "patch_sha256": sha256(patch),
                "changes": changes,
            }
            revision = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            write_json(folder / "edit.json", record)
            destination = self.edits / revision
            try:
                folder.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                list(self.chain(revision))
        archive(destination / "edit.json", "weight_edit_record", self.work)
        archive(destination / "patch.safetensors", "weight_edit_patch", self.work)
        return {"edit": revision, **record}

    def patches(self, revision):
        patches = {}
        for folder, _ in reversed(list(self.chain(revision))):
            patches.update(load_file(folder / "patch.safetensors"))
        return patches

    def export(self, revision, name):
        if revision == "base" or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError("Export requires an edit ID and a simple new directory name.")
        destination = self.work / name
        if destination.exists():
            raise FileExistsError(destination)
        chain = list(self.chain(revision))
        with TemporaryDirectory(dir=self.work) as temp:
            folder = Path(temp)
            for asset in ASSETS:
                shutil.copyfile(self.base / asset, folder / asset)
            tensors = load_file(self.base / WEIGHTS)
            tensors.update(self.patches(revision))
            save_file(tensors, folder / WEIGHTS, metadata={"format": "pt"})
            checksum = sha256(folder / WEIGHTS)
            write_json(folder / "provenance.json", {
                "base": read_json(self.base / "source.json"),
                "edit": revision,
                "weights_sha256": checksum,
                "lineage": [path.name for path, _ in reversed(chain)],
            })
            folder.rename(destination)
        for asset in (*ASSETS, WEIGHTS, "provenance.json"):
            archive(destination / asset, "checkpoint_export", self.work)
        return {"checkpoint": str(destination), "weights_sha256": checksum, "edit": revision}
