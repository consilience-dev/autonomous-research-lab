"""Supervise the CLI and retain experiment evidence outside the agent sandbox."""
import argparse
import hashlib
import json
import os
import re
import resource
import select
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

from model_audit import ModelAudit

LIMIT = 2 * 1024**3
LIVE_FILE_LIMIT = 32 * 1024**2


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Archive:
    def __init__(self, directory):
        self.directory = directory
        self.objects = directory / "objects"
        self.objects.mkdir()
        self.index = directory / "artifacts.jsonl"
        self.index_lock = threading.Lock()

    def store(self, header, read):
        remaining = header["bytes"]
        if not isinstance(remaining, int) or not 0 <= remaining <= LIMIT:
            raise ValueError("invalid payload size")
        digest = hashlib.sha256()
        with NamedTemporaryFile(dir=self.objects, delete=False) as temporary:
            path = Path(temporary.name)
            try:
                while remaining:
                    chunk = read(min(remaining, 1024**2))
                    if not chunk:
                        raise EOFError("truncated audit payload")
                    temporary.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
                checksum = digest.hexdigest()
                destination = self.objects / checksum
                if not destination.exists():
                    path.rename(destination)
            finally:
                path.unlink(missing_ok=True)
        with self.index_lock, self.index.open("a") as stream:
            stream.write(json.dumps({**header, "sha256": checksum}, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self, work, observed=None):
        live = observed is not None
        for folder, dirs, names, fd in os.fwalk(work, follow_symlinks=False):
            dirs[:] = [name for name in dirs if Path(folder, name) != work / ".capture"]
            if live and Path(folder) == work:
                # These have their own synchronous archives.
                dirs[:] = [name for name in dirs if name not in {"calls", "edits"}]
            for name in names:
                relative = str(Path(folder, name).relative_to(work))
                try:
                    raw = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
                    with os.fdopen(raw, "rb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                        if live and (info.st_size > LIVE_FILE_LIMIT or observed.get(relative) == identity):
                            continue
                        self.store({"kind": "workspace_sample" if live else "final_workspace_file",
                                    "path": relative, "time_ns": time.time_ns(),
                                    "bytes": info.st_size}, stream.read)
                        if live:
                            observed[relative] = identity
                except (OSError, ValueError, EOFError) as error:
                    self.error(f"final snapshot {relative}: {error}")

    def error(self, message):
        with (self.directory / "capture-errors.log").open("a") as stream:
            print(message, file=stream)


class Collector:
    def __init__(self, fifo, archive):
        self.archive = archive
        self.stop = threading.Event()
        self.fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW)
        self.thread = threading.Thread(target=self.collect, name="artifact-collector")
        self.thread.start()

    def read(self, size):
        chunks = bytearray()
        while len(chunks) < size:
            if select.select([self.fd], [], [], 0.1)[0]:
                chunks.extend(os.read(self.fd, size - len(chunks)))
            elif self.stop.is_set():
                break
        return bytes(chunks)

    def collect(self):
        try:
            while prefix := self.read(4):
                length, = struct.unpack(">I", prefix)
                if not 0 < length <= 1024**2:
                    raise ValueError("invalid audit header length")
                header = json.loads(self.read(length))
                self.archive.store(header, self.read)
        except (OSError, ValueError, EOFError, KeyError, TypeError, struct.error) as error:
            self.archive.error(f"audit channel: {error}")
        finally:
            os.close(self.fd)

    def close(self):
        self.stop.set()
        self.thread.join()


def copy_output(stream, destination, display, archive):
    try:
        with stream, destination.open("wb") as target:
            while chunk := stream.read1(64 * 1024):
                target.write(chunk)
                target.flush()
                if display is not None:
                    try:
                        display.write(chunk)
                        display.flush()
                    except (OSError, ValueError):
                        display = None
    except OSError as error:
        archive.error(f"{destination.name}: {error}")


def json_lines(path):
    if path.is_file():
        with path.open("rb") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        yield event
                except (UnicodeError, ValueError):
                    continue


def codex_metadata(directory, sessions=None):
    thread_ids = []
    usage = []
    item_types = set()
    for event in json_lines(directory / "events.jsonl"):
        if event.get("type") == "thread.started":
            thread_ids.append(event.get("thread_id"))
        if "usage" in event:
            usage.append({"type": event.get("type"), "usage": event["usage"]})
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            item_types.add(item["type"])
    contexts = []
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions = sessions if sessions is not None else codex_home / "sessions"
    for thread in dict.fromkeys(thread_ids):
        if not isinstance(thread, str) or not re.fullmatch(r"[0-9a-f-]{36}", thread):
            continue
        rollout = next(sessions.rglob(f"rollout-*-{thread}.jsonl"), None)
        if rollout is None:
            continue
        saved = directory / f"rollout-{thread}.jsonl"
        shutil.copyfile(rollout, saved)
        for row in json_lines(saved):
            if row.get("type") != "turn_context":
                continue
            payload = row["payload"]
            contexts.append({key: payload[key] for key in (
                "turn_id", "model", "effort", "reasoning_effort", "approval_policy",
                "sandbox_policy", "permission_profile", "active_permission_profile",
                "file_system_sandbox_policy", "disabled_plugin_ids", "cwd",
            ) if key in payload})
    return {"thread_ids": thread_ids, "usage_events": usage, "turn_contexts": contexts,
            "observed_codex_item_types": sorted(item_types)}


def artifact_counts(index):
    completed = []
    outside = []
    lab_entries = []
    for item in json_lines(index):
        if item.get("kind") == "lab_pass_completed":
            completed.append(item)
        elif item.get("kind") == "model_entry":
            (outside if item.get("classification") == "outside_lab" else lab_entries).append(item)
    return {"completed_lab_passes": completed, "observed_outside_lab_model_entries": outside,
            "observed_lab_model_entries": lab_entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--expect-model")
    parser.add_argument("--expect-effort")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    if bool(args.expect_model) != bool(args.expect_effort):
        parser.error("provide both expected model and expected effort")

    os.umask(0o077)
    directory = args.run_dir
    work = args.work.resolve()
    endpoint = work / ".capture"
    endpoint.mkdir()
    lock = endpoint / "writer.lock"
    lock.touch()
    fifo = endpoint / "records.fifo"
    os.mkfifo(fifo)
    endpoint_fd = os.open(endpoint, os.O_DIRECTORY | os.O_NOFOLLOW)
    archive = Archive(directory)
    audit = ModelAudit(directory, args.expect_model, args.expect_effort) if args.expect_model else None
    write_json(directory / "launch.json", {"argv": command, "launcher_cwd": os.getcwd(),
                                             "prompt": str(args.prompt), "started_at": time.time()})
    collector = Collector(fifo, archive)
    stop_sampling = threading.Event()

    def sample_work():
        observed = {}
        while not stop_sampling.wait(2):
            try:
                archive.snapshot(work, observed)
            except OSError as error:
                archive.error(f"workspace sample: {error}")

    sampler = threading.Thread(target=sample_work, name="workspace-sampler")
    sampler.start()
    started = time.monotonic()
    code = None
    error_text = None
    process = None
    readers = []

    def forward_signal(signum, _frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, forward_signal)
    try:
        with args.prompt.open("rb") as prompt:
            process = subprocess.Popen(command, stdin=prompt, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            for stream, name, display in (
                (process.stdout, "events.jsonl", sys.stdout.buffer),
                (process.stderr, "stderr.log", sys.stderr.buffer),
            ):
                reader = threading.Thread(target=copy_output,
                                          args=(stream, directory / name, display, archive))
                reader.start()
                readers.append(reader)
            if audit is not None:
                while process.poll() is None:
                    audit.read()
                    if audit.problem:
                        print(f"Codex model verification failed: {audit.problem}", file=sys.stderr)
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                        break
                    time.sleep(0.2)
            code = process.wait()
    except OSError as error:
        error_text = str(error)
        archive.error(f"launch: {error}")
    finally:
        for reader in readers:
            reader.join()
        collector.close()
        stop_sampling.set()
        sampler.join()
        elapsed = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        # The workspace endpoint is disposable. Leave any unexpected entry untouched.
        for path in (fifo, lock):
            try:
                os.unlink(path.name, dir_fd=endpoint_fd)
            except FileNotFoundError:
                pass
            except OSError as error:
                archive.error(f"audit endpoint cleanup: {error}")
        os.close(endpoint_fd)
        try:
            endpoint.rmdir()
        except OSError as error:
            archive.error(f"audit endpoint cleanup: {error}")
        try:
            archive.snapshot(work)
        except OSError as error:
            archive.error(f"final workspace snapshot: {error}")
        summary = {
            "process_returncode": code,
            "exit_status": 1 if code is None else code if code >= 0 else 128 - code,
            "signal": -code if code is not None and code < 0 else None,
            "launch_error": error_text, "elapsed_seconds": elapsed, "finished_at": time.time(),
            "child_rusage": {"user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
                             "peak_rss_bytes_on_macos": usage.ru_maxrss},
            **artifact_counts(archive.index),
        }
        try:
            summary.update(codex_metadata(directory))
        except (OSError, KeyError, TypeError) as error:
            archive.error(f"Codex rollout metadata: {error}")
        if audit is not None:
            model_report = audit.report(require_completion=code == 0)
            summary["model_verification"] = model_report
            write_json(directory / "model-verification.json", model_report)
            if model_report["error"]:
                archive.error(model_report["error"])
                if code == 0:
                    summary["exit_status"] = 78
        summary["capture_errors"] = (directory / "capture-errors.log").is_file()
        write_json(directory / "session-summary.json", summary)
    return summary["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
