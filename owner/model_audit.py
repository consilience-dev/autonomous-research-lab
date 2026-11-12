"""Read effective Codex settings and detect model substitutions from local evidence."""
import json
import os
import re
from pathlib import Path

THREAD_ID = re.compile(r"[0-9a-f-]{36}")
SERVER_MODEL = re.compile(r"server reported model ([\w.-]+)(?: while requested model was ([\w.-]+))?")
REROUTE_TYPES = {"model_reroute", "model.rerouted", "model/rerouted"}


def sessions_dir():
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"


def find_rollout(thread, sessions):
    if isinstance(thread, str) and THREAD_ID.fullmatch(thread):
        return next(sessions.rglob(f"rollout-*-{thread}.jsonl"), None)
    return None


def lines(path):
    if path.is_file():
        with path.open("rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if isinstance(row, dict):
                    yield row


class ModelAudit:
    def __init__(self, directory, model, effort, sessions=None):
        self.directory = directory
        self.model = model
        self.effort = effort
        self.sessions = sessions if sessions is not None else sessions_dir()
        self.offsets = {}
        self.thread = None
        self.rollout = None
        self.turns = []
        self.applied = []
        self.server_models = []
        self.reroutes = []
        self.requests = {}
        self.responses = {}
        self.completed = False
        self.problem = None

    def tail(self, path):
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            return
        with stream:
            stream.seek(self.offsets.get(path, 0))
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                self.offsets[path] = stream.tell()
                yield line

    def records(self, path):
        for line in self.tail(path):
            try:
                row = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            if isinstance(row, dict):
                yield row

    def settings(self, record):
        model = record.get("model")
        effort = record.get("reasoning_effort", record.get("effort"))
        if model != self.model or effort != self.effort:
            self.problem = f"Effective Codex settings were {model!r} / {effort!r}; expected {self.model} / {self.effort}."

    def read(self):
        for event in self.records(self.directory / "events.jsonl"):
            kind = event.get("type")
            if kind == "thread.started":
                self.thread = event.get("thread_id")
            if kind == "turn.completed":
                self.completed = True
            if kind in REROUTE_TYPES:
                self.reroutes.append(event)
                self.problem = "Codex reported a model reroute."
        if self.thread and self.rollout is None:
            self.rollout = find_rollout(self.thread, self.sessions)
        if self.rollout is not None:
            for event in self.records(self.rollout):
                payload = event.get("payload", {})
                if not isinstance(payload, dict):
                    continue
                if event.get("type") == "turn_context":
                    self.turns.append({k: payload[k] for k in ("model", "effort", "reasoning_effort", "turn_id") if k in payload})
                    self.settings(payload)
                kind = payload.get("type")
                if kind == "thread_settings_applied":
                    settings = payload.get("thread_settings", {})
                    self.applied.append({k: settings[k] for k in ("model", "reasoning_effort", "model_provider_id") if k in settings})
                    self.settings(settings)
                if kind in REROUTE_TYPES:
                    self.reroutes.append(payload)
                    self.problem = "Codex reported a model reroute."
        for line in self.tail(self.directory / "stderr.log"):
            # Codex diagnostics include the owning thread. Ignore unrelated worker traces.
            text = line.decode(errors="replace")
            if not self.thread or f"thread_id={self.thread}" not in text:
                continue
            match = SERVER_MODEL.search(text)
            if match:
                served, requested = match.groups()
                self.server_models.append(served)
                if served != self.model or (requested is not None and requested != self.model):
                    self.problem = f"The service reported {served!r}; expected {self.model}."
        if self.thread:
            self.read_trace()

    def read_trace(self):
        for path in (self.directory / "trace").glob("*/trace.jsonl"):
            for record in self.records(path):
                if record.get("thread_id") != self.thread:
                    continue
                event = record.get("payload", {})
                if event.get("type") == "protocol_event_observed" and event.get("event_type") in REROUTE_TYPES:
                    self.reroutes.append(event)
                    self.problem = "Codex reported a model reroute in the native trace."
                call = event.get("inference_call_id")
                if not call:
                    continue
                if event.get("type") == "inference_started":
                    payload = event.get("request_payload", {}).get("path", "")
                    if not re.fullmatch(r"payloads/[0-9]+\.json", payload):
                        self.problem = "Codex did not provide a recognized inference request payload."
                        continue
                    try:
                        request = json.loads((path.parent / payload).read_bytes())
                    except (OSError, ValueError):
                        self.problem = "Codex inference request payload could not be read."
                        continue
                    model = request.get("model")
                    effort = (request.get("reasoning") or {}).get("effort")
                    self.requests[call] = {"model": model, "reasoning_effort": effort,
                                           "trace_payload": str((path.parent / payload).relative_to(self.directory))}
                    if model != self.model or effort != self.effort:
                        self.problem = f"Codex sent {model!r} / {effort!r}; expected {self.model} / {self.effort}."
                elif event.get("type") == "inference_completed" and event.get("response_id"):
                    self.responses[call] = event["response_id"]

    def report(self, require_completion=True):
        self.read()
        problem = self.problem
        if require_completion and not problem:
            if not self.turns or not self.applied or not self.completed:
                problem = "Codex did not expose both applied settings and a completed turn."
            elif not self.requests.keys() & self.responses.keys():
                problem = "Codex did not expose a matching inference request and completed service response."
        return {
            "requested": {"model": self.model, "reasoning_effort": self.effort},
            "thread_id": self.thread, "effective_turn_contexts": self.turns,
            "applied_settings": self.applied, "completed_turn": self.completed,
            "inference_requests": [{"call_id": call, **request, "response_id": self.responses.get(call)}
                                   for call, request in self.requests.items()],
            "server_reported_models": self.server_models, "reroutes": self.reroutes,
            "error": problem,
        }
