"""
A local, no-network field notebook for bird-sound identifications.

Every time identify.py classifies a recording, it writes an entry here. This is
the "keep your observations off a server you don't control" piece: your
recordings, your locations, your sightings stay on your device unless you
choose to export them.

Two backends are supported, pick one:
  - SQLite (default): field_log.sqlite, one row per identification.
  - JSONL: field_log.jsonl, one JSON object per line.

SQLite is the default. Set the environment variable FIELD_LOG_BACKEND=jsonl to
use JSONL. The JSONL path is easy to version-control or copy; the SQLite path is
easy to query.

No network. No account. Your data stays local.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = (Path(__file__).resolve().parent / "..").resolve()
LOG_SQLITE = PROJECT_ROOT / "field_log.sqlite"
LOG_JSONL = PROJECT_ROOT / "field_log.jsonl"
BACKEND = os.environ.get("FIELD_LOG_BACKEND", "sqlite").lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_id() -> str:
    # A stable-but-anonymous per-device tag so entries from the same device can
    # be grouped, without collecting anything identifiable. Derived from the
    # project root path + a salt; not sent anywhere.
    import hashlib
    raw = f"{PROJECT_ROOT.as_posix()}:field-log-salt:{BACKEND}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def init_sqlite() -> None:
    conn = sqlite3.connect(str(LOG_SQLITE))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created TEXT NOT NULL,
                device_tag TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                audio_exists INTEGER NOT NULL,
                top_species TEXT,
                top_support REAL,
                top_confidence REAL,
                shortlist_json TEXT,
                confidence_flag TEXT,
                model_name TEXT,
                model_params INTEGER,
                note TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def append_sqlite(entry: dict) -> int:
    conn = sqlite3.connect(str(LOG_SQLITE))
    try:
        cur = conn.execute("""
            INSERT INTO sightings
                (created, device_tag, audio_path, audio_exists,
                 top_species, top_support, top_confidence,
                 shortlist_json, confidence_flag, model_name, model_params, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["created"], entry["device_tag"], entry["audio_path"],
            int(entry["audio_exists"]), entry.get("top_species"), entry.get("top_support"),
            entry.get("top_confidence"), json.dumps(entry.get("shortlist", [])),
            entry.get("confidence_flag"), entry.get("model_name"),
            entry.get("model_params"), entry.get("note"),
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def append_jsonl(entry: dict) -> int:
    entry.setdefault("id", None)
    line = json.dumps(entry, ensure_ascii=False)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return None


def append(entry: dict) -> int | None:
    if BACKEND == "jsonl":
        LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
        return append_jsonl(entry)
    init_sqlite()
    return append_sqlite(entry)


def list_entries(limit: int = 50, backend: str | None = None) -> list[dict]:
    backend = backend or BACKEND
    if backend == "jsonl":
        rows: list[dict] = []
        if LOG_JSONL.exists():
            for line in LOG_JSONL.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return rows[-limit:]
    init_sqlite()
    conn = sqlite3.connect(str(LOG_SQLITE))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM sightings ORDER BY created DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def print_entries(limit: int = 50, backend: str | None = None) -> None:
    rows = list_entries(limit, backend)
    if not rows:
        print("Field log is empty. Run identify.py on a recording to log a sighting.")
        return
    print(f"Field log ({BACKEND}, {len(rows)} shown):")
    print(f"{'#':>3}  {'created (UTC)':<25} {'top_species':<18} {'support':>6} {'conf':>6}  flag")
    for i, r in enumerate(rows, 1):
        top = r.get("top_species") or "(none)"
        sup = f"{r.get('top_support'):.2f}" if r.get("top_support") is not None else "  -  "
        conf = f"{r.get('top_confidence'):.2f}" if r.get("top_confidence") is not None else "  -  "
        print(f"{i:>3}  {r.get('created',''):<25} {top:<18} {sup:>6} {conf:>6}  {r.get('confidence_flag','')}")


if __name__ == "__main__":
    # Quick CLI: print the log, optionally with --backend jsonl and --limit N.
    import argparse
    p = argparse.ArgumentParser(description="Read the local field log.")
    p.add_argument("--backend", choices=["sqlite", "jsonl"], help="which backend to read")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    if args.backend:
        os.environ["FIELD_LOG_BACKEND"] = args.backend
    # Re-read backend after possible env change.
    print_entries(limit=args.limit, backend=args.backend or os.environ.get("FIELD_LOG_BACKEND", "sqlite"))
