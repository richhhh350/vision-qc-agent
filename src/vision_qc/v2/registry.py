"""Transactional single-machine registry; review-only models may serve safely."""
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from .evidence import check_certificate, check_weights, policy_hash


class Registry:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.database = self.root/"registry.sqlite"
        with sqlite3.connect(self.database) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS models(version TEXT PRIMARY KEY,path TEXT NOT NULL,policy TEXT NOT NULL,metadata_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at TEXT NOT NULL,action TEXT NOT NULL,version TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active(slot INTEGER PRIMARY KEY CHECK(slot=1),version TEXT NOT NULL);
            """)

    def register(self, run):
        run = Path(run).resolve()
        raw = (run/"model.json").read_bytes()
        meta = json.loads(raw)
        check_weights(run,meta)
        record = (meta["version"],str(run),policy_hash(meta),hashlib.sha256(raw).hexdigest())
        with sqlite3.connect(self.database) as db:
            old = db.execute("SELECT version,path,policy,metadata_hash FROM models WHERE version=?",(meta["version"],)).fetchone()
            if old and tuple(old)!=record:
                raise ValueError("Version already registered with different package")
            db.execute("INSERT OR IGNORE INTO models VALUES(?,?,?,?)",record)
        return dict(version=meta["version"],state="approved" if check_certificate(meta)[0] else "evaluated",release_approved=check_certificate(meta)[0])

    def activate(self, version, *, require_release=False, action="activate"):
        # BEGIN IMMEDIATE serializes promotion and rollback across processes.
        with sqlite3.connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT path,policy,metadata_hash FROM models WHERE version=?",(version,)).fetchone()
            if not row:
                raise ValueError("Unknown registered version")
            run = Path(row[0]); raw = (run/"model.json").read_bytes(); meta = json.loads(raw)
            if policy_hash(meta)!=row[1] or hashlib.sha256(raw).hexdigest()!=row[2]:
                raise ValueError("Registered package changed")
            check_weights(run,meta)
            approved,reason = check_certificate(meta)
            if require_release and not approved:
                raise ValueError(f"Cannot enable release: {reason}")
            db.execute("INSERT OR REPLACE INTO active VALUES(1,?)",(version,))
            db.execute("INSERT INTO events(at,action,version) VALUES(?,?,?)",(datetime.now(timezone.utc).isoformat(),action,version))
        return dict(version=version,run=str(run),release_approved=approved,mode="release" if approved else "review_only")

    def rollback(self):
        with sqlite3.connect(self.database) as db:
            current = db.execute("SELECT version FROM active WHERE slot=1").fetchone()
            if not current:
                raise ValueError("No active version")
            row = db.execute("SELECT version FROM events WHERE version != ? ORDER BY id DESC LIMIT 1",current).fetchone()
        if not row:
            raise ValueError("No previous version")
        return self.activate(row[0],action="rollback")

    def active_run(self):
        with sqlite3.connect(self.database) as db:
            row = db.execute("SELECT m.path,m.metadata_hash FROM active a JOIN models m ON a.version=m.version WHERE slot=1").fetchone()
        if row is None:
            raise ValueError("Registry has no active model")
        if hashlib.sha256((Path(row[0])/"model.json").read_bytes()).hexdigest()!=row[1]:
            raise ValueError("Active package changed")
        return Path(row[0])

    def list(self):
        with sqlite3.connect(self.database) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT m.version,m.path,m.policy,a.slot IS NOT NULL AS active FROM models m LEFT JOIN active a ON m.version=a.version")]
