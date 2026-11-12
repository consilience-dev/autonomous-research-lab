"""Install the pinned, licensed Gemma model with the owner's Hugging Face login."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from weights import ASSETS, BASE, MODEL_ID, REVISION, WEIGHTS, sha256, write_json

if BASE.exists():
    sys.exit(f"Base already exists at {BASE}.")
files = (*ASSETS, WEIGHTS)
model = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True, token=True)
upstream = {file.rfilename: file for file in model.siblings}
with TemporaryDirectory(dir=BASE.parent) as directory:
    staged = Path(directory)
    snapshot_download(MODEL_ID, revision=REVISION, token=True,
                      local_dir=staged, allow_patterns=list(files))
    checksums = {name: sha256(staged / name) for name in files}
    if model.sha != REVISION or checksums[WEIGHTS] != upstream[WEIGHTS].lfs.sha256:
        sys.exit("Base revision or upstream weight checksum mismatch.")
    write_json(staged / "source.json", {"model": MODEL_ID, "revision": REVISION, "sha256": checksums})
    import shutil
    if (staged / ".cache").exists():
        shutil.rmtree(staged / ".cache")
    for asset in staged.iterdir():
        asset.chmod(0o444)
    staged.rename(BASE)
print(f"Installed {MODEL_ID}@{REVISION} at {BASE}")
