"""Download and verify the unmodified BF16 Qwen checkpoint."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL = Path(__file__).resolve().parents[1] / "models/qwen2.5-3b-instruct"
SHARDS = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
FILES = [
    "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    *SHARDS,
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    marker = MODEL / "source.json"
    if marker.exists():
        data = json.loads(marker.read_text())
        if data["revision"] != REVISION or data["sha256"] != {name: digest(MODEL / name) for name in FILES}:
            raise RuntimeError("Installed checkpoint does not match the pinned manifest.")
        print(f"Verified existing installation at {MODEL}")
        return
    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True, token=False)
    upstream = {item.rfilename: item for item in info.siblings or []}
    if info.sha != REVISION or any(upstream[name].lfs.sha256 != expected for name, expected in SHARDS.items()):
        raise RuntimeError("Upstream revision or shard digest mismatch.")
    snapshot_download(REPO_ID, revision=REVISION, local_dir=MODEL, allow_patterns=FILES,
                      token=False, max_workers=3)
    hashes = {name: digest(MODEL / name) for name in FILES}
    if any(hashes[name] != expected for name, expected in SHARDS.items()):
        raise RuntimeError("Downloaded weights differ from the published originals.")
    marker.write_text(json.dumps({"model": REPO_ID, "revision": REVISION, "sha256": hashes}, indent=2) + "\n")
    print(f"Installed and verified {REPO_ID}@{REVISION} at {MODEL}")


if __name__ == "__main__":
    main()
