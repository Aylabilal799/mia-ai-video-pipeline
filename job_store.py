"""job_store.py — persistent job record store for the YouTube upload stage.

Deliberately NOT placed inside PUBLIC_SHARE_DIR: that directory is served
publicly by file_server.py, and job records may reference local paths /
Discord user ids that shouldn't be web-exposed.

One row per Mia video job, keyed by the Celery task id (the same id used as
the job's output subfolder under PUBLIC_SHARE_DIR). This is intentionally
plain sqlite3 (stdlib only) instead of a new dependency.
"""

import os
import sqlite3
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Keep the DB next to the app code, not in the public downloads folder.
JOB_DB_PATH = os.getenv(
    "JOB_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jobs.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,   -- Celery task id from generate_mia_video_task
    status           TEXT NOT NULL,      -- generated | upload_pending | uploading |
                                          -- uploaded | upload_failed
    video_path       TEXT NOT NULL,      -- local VPS path to final_video.mp4 (from tasks.py)
    seo_path         TEXT,               -- local VPS path to mia_seo.txt (from tasks.py)
    work_dir         TEXT,               -- /tmp/mia_<task_id> -- intermediate files only
    topic            TEXT,
    category         TEXT,
    script           TEXT,
    discord_user_id  TEXT,
    youtube_video_id TEXT,
    youtube_url      TEXT,
    privacy_status   TEXT,
    publish_at       TEXT,               -- RFC3339 UTC, e.g. 2026-08-20T09:30:00Z
    last_error       TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
"""


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(JOB_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(JOB_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(job_id: str, video_path: str, seo_path: str, topic: str,
               category: str, script: str, discord_user_id: str,
               work_dir: str = None, status: str = "generated",
               publish_at: str = None) -> None:
    """Called once, right after tasks.py finishes moving the final MP4 into
    PUBLIC_SHARE_DIR. video_path MUST be the exact local path the file was
    moved to (tasks.py's `target_video`) -- this store never derives or
    guesses that path itself.

    publish_at: optional RFC3339 UTC timestamp from !miaschedule. None for
    !mia / !miascript (immediate publish, no schedule)."""
    now = time.time()
    with _conn() as c:
        c.execute(
            """INSERT INTO jobs
               (job_id, status, video_path, seo_path, work_dir, topic, category,
                script, discord_user_id, publish_at, attempts, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                   video_path=excluded.video_path,
                   seo_path=excluded.seo_path,
                   work_dir=excluded.work_dir,
                   publish_at=excluded.publish_at,
                   updated_at=excluded.updated_at""",
            (job_id, status, video_path, seo_path, work_dir, topic, category,
             script, str(discord_user_id), publish_at, now, now),
        )


def get_job(job_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?",
                   (*fields.values(), job_id))


def increment_attempts(job_id: str) -> int:
    with _conn() as c:
        c.execute("UPDATE jobs SET attempts = attempts + 1, updated_at = ? WHERE job_id = ?",
                   (time.time(), job_id))
        row = c.execute("SELECT attempts FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row["attempts"] if row else 0


def list_by_status(status: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at",
                          (status,)).fetchall()
        return [dict(r) for r in rows]
