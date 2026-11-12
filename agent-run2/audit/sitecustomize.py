"""Best-effort observation for Python started with the experiment environment."""
import atexit
import collections
import os
import sys
import threading
from pathlib import Path

from audit.lab_audit import LAB_CALL, send

if os.getenv("QWEN_AUDIT_FIFO"):
    WORK = Path(os.environ["QWEN_AUDIT_FIFO"]).parent.parent
    guard = threading.local()
    observed = collections.Counter()

    def audit(event, args):
        if getattr(guard, "active", False):
            return
        guard.active = True
        try:
            if event == "cpython.run_file":
                path = Path(args[0]).absolute()
                if path.is_file():
                    send("python_script", {"path": str(path), "argv": sys.argv}, source=path)
            elif event == "cpython.run_command":
                send("python_command", {"argv": sys.argv}, data=args[0].encode())
            elif event in {"cpython.run_module", "cpython.run_stdin"}:
                send("python_entry", {"event": event, "argv": sys.argv, "arguments": list(args)})
            elif event == "compile":
                source, filename = args
                if not isinstance(source, (bytes, str)) or not isinstance(filename, str):
                    return
                if filename in {"<stdin>", "<string>", "<console>"} or (
                    not filename.startswith("<") and Path(filename).absolute().is_relative_to(WORK)
                ):
                    data = source if isinstance(source, bytes) else source.encode()
                    send("python_compile", {"path": filename}, data=data)
            elif event == "exec":
                path = Path(args[0].co_filename)
                if path.is_absolute() and path.is_relative_to(WORK) and path.is_file():
                    send("python_exec_source", {"path": str(path)}, source=path)
        finally:
            guard.active = False

    def profile(frame, event, _arg):
        if event != "call":
            return
        code = frame.f_code
        if code.co_name not in {"forward", "generate"}:
            return
        file = code.co_filename
        if not (file.endswith("/transformers/models/qwen2/modeling_qwen2.py") or
                file.endswith("/transformers/generation/utils.py")):
            return
        model = frame.f_locals.get("self")
        if type(model).__name__ not in {"Qwen2ForCausalLM", "Qwen2Model"}:
            return
        call = LAB_CALL.get()
        key = (call, code.co_qualname)
        observed[key] += 1
        if observed[key] == 1:
            send("model_entry", {"function": code.co_qualname,
                                 "classification": "lab" if call else "outside_lab"})

    def finish():
        sys.setprofile(None)
        threading.setprofile(None)
        for (call, function), count in observed.items():
            send("model_entry_count", {"call_id": call, "function": function, "entries": count,
                                       "classification": "lab" if call else "outside_lab"})

    sys.addaudithook(audit)
    send("python_start", {"executable": sys.executable, "argv": sys.argv, "cwd": os.getcwd()})
    sys.setprofile(profile)
    threading.setprofile(profile)
    atexit.register(finish)
