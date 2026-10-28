from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, image_path TEXT NOT NULL,
                    result TEXT NOT NULL, route TEXT NOT NULL, status TEXT NOT NULL,
                    label INTEGER, revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, inspection_id TEXT NOT NULL,
                    label INTEGER NOT NULL CHECK(label IN (0,1)), reviewer TEXT NOT NULL,
                    note TEXT NOT NULL, created_at TEXT NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS masks (
                    inspection_id TEXT PRIMARY KEY, path TEXT NOT NULL, label_revision INTEGER NOT NULL,
                    seconds REAL, spec_version TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def add(self, inspection_id: str, image_path: str, result: dict):
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO inspections(id,created_at,image_path,result,route,status) VALUES (?,?,?,?,?,?)",
                       (inspection_id, datetime.now(timezone.utc).isoformat(), image_path, json.dumps(result, allow_nan=False),
                        result["route"], "pending" if result["route"] == "review" else "auto"))
        return self.get(inspection_id)

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError("Inspection not found")
        value = dict(row)
        value["result"] = json.loads(value["result"])
        return value

    def get(self, inspection_id: str):
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone())

    def list(self, status: str | None = None, limit: int = 100):
        with self.connection() as db:
            if status:
                rows = db.execute("SELECT * FROM inspections WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = db.execute("SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.decode(row) for row in rows]

    def feedback(self, inspection_id: str, label: int, reviewer: str, note: str, revision: int):
        if label not in (0, 1) or not reviewer.strip():
            raise ValueError("A binary label and reviewer are required")
        with self.connection() as db:
            cursor = db.execute("UPDATE inspections SET label=?,status='labeled',revision=revision+1 WHERE id=? AND revision=?",
                                (label, inspection_id, revision))
            if cursor.rowcount != 1:
                if db.execute("SELECT 1 FROM inspections WHERE id=?", (inspection_id,)).fetchone() is None:
                    raise KeyError("Inspection not found")
                raise ValueError("Revision conflict; reload before labeling")
            db.execute("INSERT INTO feedback(inspection_id,label,reviewer,note,created_at,revision) VALUES (?,?,?,?,?,?)",
                       (inspection_id, label, reviewer.strip(), note, datetime.now(timezone.utc).isoformat(), revision + 1))
        return self.get(inspection_id)

    def summary(self):
        with self.connection() as db:
            return {row["status"]: row["n"] for row in db.execute("SELECT status,COUNT(*) n FROM inspections GROUP BY status")}

    def history(self, inspection_id: str):
        self.get(inspection_id)
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT label,reviewer,note,created_at,revision FROM feedback WHERE inspection_id=? ORDER BY revision", (inspection_id,))]

    def refresh_result(self, inspection_id, result):
        """Refresh time-dependent release decisions without discarding human labels."""
        with self.connection() as db:
            db.execute("UPDATE inspections SET result=?,route=?,status=CASE WHEN label IS NOT NULL THEN 'labeled' WHEN ?='review' THEN 'pending' ELSE 'auto' END WHERE id=?",
                       (json.dumps(result,allow_nan=False),result["route"],result["route"],inspection_id))
        return self.get(inspection_id)

    def save_mask(self, inspection_id, path, revision, seconds=None, spec_version="1"):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT label,revision FROM inspections WHERE id=?",(inspection_id,)).fetchone()
            if row is None:
                raise KeyError("Inspection not found")
            if row["label"] != 1 or row["revision"] != revision:
                raise ValueError("Mask requires the current defect label revision")
            db.execute("INSERT OR REPLACE INTO masks VALUES(?,?,?,?,?)",(inspection_id,str(path),revision,seconds,spec_version))

    def masks(self):
        with self.connection() as db:
            return {r["inspection_id"]:dict(r) for r in db.execute("SELECT m.* FROM masks m JOIN inspections i ON i.id=m.inspection_id WHERE i.label=1 AND i.revision=m.label_revision")}
