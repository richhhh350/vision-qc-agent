"""Persistent local subprocess jobs; inference never runs training in its request thread."""
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone


class TrainingJobs:
    def __init__(self, run, state):
        self.run,self.state = Path(run),Path(state)
        self.root = self.state/"jobs"
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = threading.Lock()
        self.processes = {}
        for path in self.root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["status"] in {"queued","running"}:
                value.update(status="interrupted",reason="Service restarted; candidate was not activated")
                self._write(value)

    def _write(self, value):
        path = self.root/f"{value['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
        temporary.replace(path)

    def get(self, identifier):
        if len(identifier)!=32 or any(c not in "0123456789abcdef" for c in identifier):
            raise KeyError("Unknown job")
        path = self.root/f"{identifier}.json"
        if not path.exists():
            raise KeyError("Unknown job")
        value = json.loads(path.read_text(encoding="utf-8"))
        log = self.root/f"{identifier}.log"
        if log.exists():
            value["progress"] = log.read_text(encoding="utf-8",errors="replace")[-2000:]
        return value

    def start(self, steps=120):
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError("A training job is already running")
            identifier = uuid.uuid4().hex
            output = self.state/"candidates"/identifier
            value = dict(id=identifier,status="running",created_at=datetime.now(timezone.utc).isoformat(),
                         output=str(output),reason=None)
            log = (self.root/f"{identifier}.log").open("w",encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0
            try:
                process = subprocess.Popen([sys.executable,"-u","-m","vision_qc.v2.worker",
                    "--run",str(self.run.resolve()),"--state",str(self.state.resolve()),"--output",str(output.resolve()),
                    "--steps",str(steps)],stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            finally:
                log.close()
            self.processes[identifier] = process
            self._write(value)
        def finish():
            code = process.wait()
            with self.lock:
                current = self.get(identifier)
                current.pop("progress",None)
                if current["status"] != "cancelled":
                    current.update(status="completed" if code==0 else "failed",reason=None if code==0 else f"Worker exit {code}; see progress log")
                self._write(current)
        threading.Thread(target=finish,daemon=True).start()
        return value

    def cancel(self, identifier):
        with self.lock:
            value = self.get(identifier)
            process = self.processes.get(identifier)
            if process and process.poll() is None:
                process.terminate()
                value.pop("progress",None)
                value.update(status="cancelled",reason="Cancelled by user; candidate not activated")
                self._write(value)
        return value
