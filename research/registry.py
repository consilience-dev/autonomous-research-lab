"""Register owner-side experiments and verify their evidence using the standard library."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
INDEX = RESEARCH / "index.json"
MANIFESTS = RESEARCH / "manifests"
PREFIX = {"run": "RUN", "eval": "EVAL", "mech": "MECH"}
OUTCOMES = ("completed", "interrupted", "failed")
DOCS = ("README.md", "START_HERE.md", "TIMELINE.md", "FINDINGS.md", "DECISIONS.md")
LINK = re.compile(r"!?\[[^\]\n]+\]\((<[^>\n]+>|[^()\s]+)\)")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Provide ISO 8601 timestamps with an explicit UTC offset.")
    return parsed


def safe_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError(f"Expected a repository-relative artifact path: {name}")
    result = ROOT / path
    if result.resolve() != result:
        raise ValueError(f"Symlink paths are excluded from evidence manifests: {name}")
    return result


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
        after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError(f"File changed while hashing: {path.relative_to(ROOT)}")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": after.st_size,
            "sha256": checksum.hexdigest()}


def artifact_files(names):
    files = set()
    for name in names:
        path = safe_path(name)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            for entry in path.rglob("*"):
                if ".git" in entry.relative_to(ROOT).parts:
                    raise ValueError("Git internals cannot be captured as experiment artifacts.")
                if entry.is_symlink():
                    raise ValueError(f"Symlink inside artifact directory: {entry.relative_to(ROOT)}")
                if entry.is_file():
                    files.add(entry)
                elif not entry.is_dir():
                    raise ValueError(f"Unsupported artifact type: {entry.relative_to(ROOT)}")
        else:
            raise ValueError(f"Artifact does not exist: {name}")
    return sorted(files)


def atomic_write(path, content):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()


def write_index(index):
    atomic_write(INDEX, json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_index():
    index = json.loads(INDEX.read_text())
    if index["schema_version"] != 1:
        raise ValueError("Unsupported experiment index schema.")
    return index


def snapshot(identifier, sequence, names):
    filename = identifier if sequence == 1 else f"{identifier}-{sequence:02d}"
    path = MANIFESTS / f"{filename}.jsonl"
    if path.exists():
        raise ValueError(f"Manifest already exists: {path.relative_to(ROOT)}")
    entries = [digest(file) for file in artifact_files(names)]
    if not entries:
        raise ValueError("Artifact selection contains no files.")
    content = "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in entries)
    atomic_write(path, content)
    return {**digest(path), "captured_at": now(), "roots": sorted(set(names)),
            "file_count": len(entries)}


def times(started, ended):
    if started:
        start = timestamp(started)
    if ended:
        end = timestamp(ended)
        if started and end < start:
            raise ValueError("End timestamp precedes start timestamp.")


def object_arg(value):
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Resource data must be a JSON object.")
    return result


def register(args, index):
    existing = {entry["id"] for entry in index["experiments"]}
    if set(args.parent) - existing:
        raise ValueError("Parent experiments must already be registered.")
    times(args.started, args.ended)
    prefix = PREFIX[args.kind]
    counters = [int(item.split("-")[1]) for item in existing if item.startswith(prefix + "-")]
    identifier = f"{prefix}-{max(counters, default=0) + 1:03d}"
    report = RESEARCH / f"{identifier}.md"
    if report.exists():
        raise ValueError(f"Record already exists: {report.relative_to(ROOT)}")
    config = digest(safe_path(args.config))
    resources = object_arg(args.resources)
    entry = {
        "id": identifier, "kind": args.kind, "title": args.title, "status": args.status,
        "registered_at": now(), "started_at": args.started, "ended_at": args.ended,
        "parents": args.parent, "record": report.relative_to(ROOT).as_posix(),
        "config": config, "resources": resources,
        "manifests": [snapshot(identifier, 1, [args.config, *args.artifact])],
    }
    template = (RESEARCH / "TEMPLATE.md").read_text()
    atomic_write(report, template.replace("{{ID}}", identifier).replace("{{TITLE}}", args.title))
    index["experiments"].append(entry)
    write_index(index)
    print(f"Registered {identifier}. Complete {entry['record']} and append a timeline entry.")


def seal(args, index):
    entry = next((entry for entry in index["experiments"] if entry["id"] == args.id), None)
    if entry is None or entry["status"] != "planned":
        raise ValueError("Only an existing planned experiment can be sealed.")
    started = args.started or entry["started_at"]
    times(started, args.ended)
    resources = object_arg(args.resources)
    manifest = snapshot(args.id, len(entry["manifests"]) + 1, args.artifact)
    entry.update(started_at=started, ended_at=args.ended, status=args.status, resources=resources)
    entry["manifests"].append(manifest)
    write_index(index)
    print(f"Sealed {args.id}. Update {entry['record']} and append the outcome to TIMELINE.md.")


def verify(index):
    issues = []
    verified = {}
    documents = [ROOT / name for name in DOCS] + [RESEARCH / name for name in ("README.md", "TEMPLATE.md")]
    ids = set()
    records = 0

    def check(entry):
        path = safe_path(entry["path"])
        if path not in verified:
            verified[path] = digest(path)
        actual = verified[path]
        if actual["sha256"] != entry["sha256"] or actual["bytes"] != entry["bytes"]:
            issues.append(f"Checksum or size mismatch: {entry['path']}")
        return path

    for entry in index["experiments"]:
        identifier = entry["id"]
        if identifier in ids or not re.fullmatch(PREFIX[entry["kind"]] + r"-\d{3,}", identifier):
            issues.append(f"Duplicate or invalid experiment ID: {identifier}")
        if set(entry["parents"]) - ids:
            issues.append(f"Unknown or later parent for {identifier}")
        ids.add(identifier)
        times(entry["started_at"], entry["ended_at"])
        documents.append(safe_path(entry["record"]))
        check(entry["config"])
        for manifest in entry["manifests"]:
            path = check(manifest)
            for root in manifest["roots"]:
                if not safe_path(root).exists():
                    issues.append(f"Missing manifest root: {root}")
            lines = path.read_text().splitlines()
            if len(lines) != manifest["file_count"]:
                issues.append(f"Unexpected entry count: {manifest['path']}")
            for line in lines:
                check(json.loads(line))
                records += 1

    link_count = 0
    for doc in documents:
        in_code = False
        for line in doc.read_text().splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                in_code = not in_code
                continue
            if in_code:
                continue
            for match in LINK.finditer(line):
                target = urlsplit(match[1].strip("<>"))
                if target.scheme or target.netloc or not target.path:
                    continue
                linked = (doc.parent / unquote(target.path)).resolve()
                link_count += 1
                if not linked.is_relative_to(ROOT) or not linked.exists():
                    issues.append(f"Broken local link in {doc.relative_to(ROOT)}: {match[1]}")
    if issues:
        raise ValueError("\n".join(issues))
    print(f"Verified {len(ids)} experiments, {records} artifact entries, "
          f"{len(verified)} unique files (including manifests), and {link_count} local links.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("register", help="Allocate the next ID and capture initial evidence.")
    add.add_argument("--kind", choices=PREFIX, required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--config", required=True, help="Existing safe, repository-relative configuration file.")
    add.add_argument("--parent", action="append", default=[])
    add.add_argument("--status", choices=("planned", *OUTCOMES), default="planned")
    add.add_argument("--artifact", action="append", default=[])
    finish = sub.add_parser("seal", help="Attach final evidence to a planned experiment.")
    finish.add_argument("id")
    finish.add_argument("--status", choices=OUTCOMES, required=True)
    finish.add_argument("--artifact", action="append", required=True)
    for cmd in (add, finish):
        cmd.add_argument("--started", help="ISO 8601 timestamp with UTC offset.")
        cmd.add_argument("--ended", help="ISO 8601 timestamp with UTC offset.")
        cmd.add_argument("--resources", default="{}", help="JSON object with resource totals.")
    sub.add_parser("verify", help="Rehash recorded artifacts and check local documentation links.")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify(load_index())
        else:
            with (RESEARCH / ".registry.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                index = load_index()
                (register if args.command == "register" else seal)(args, index)
    except (OSError, ValueError, KeyError) as error:
        print(f"Verification or registration failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
