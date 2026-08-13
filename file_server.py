"""
file_server.py -- tiny self-hosted temp file server for finished videos.

Why this exists: Discord's own upload limit (25MB on non-boosted servers,
higher on boosted ones) is way smaller than a good-quality YouTube Shorts
clip. Instead of attaching the file to a Discord message, the bot moves the
finished video into SHARE_DIR and this server exposes it at a plain HTTP(S)
URL, e.g. http://your-host:8766/files/<id>.mp4 -- Discord then just posts
that link as text, so there's no size limit at all.

A background thread deletes anything older than FILE_SHARE_MAX_AGE_HOURS
(default 24h) so the disk doesn't fill up.

Run directly (`python file_server.py`) or under systemd -- see
systemd/shorts-fileserver.service.
"""
import os
import time
import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHARE_DIR = Path(os.getenv('FILE_SHARE_DIR', os.path.join(os.path.dirname(__file__), 'shared_videos')))
SHARE_DIR.mkdir(parents=True, exist_ok=True)

MAX_AGE_HOURS = float(os.getenv('FILE_SHARE_MAX_AGE_HOURS', '24'))
CLEANUP_INTERVAL_SECONDS = int(os.getenv('FILE_SHARE_CLEANUP_INTERVAL_SECONDS', '600'))  # every 10 min
HOST = os.getenv('FILE_SHARE_HOST', '0.0.0.0')
PORT = int(os.getenv('FILE_SHARE_PORT', '8766'))

app = FastAPI(title="Shorts Bot File Share")


@app.get("/healthz")
def healthz():
    return {"ok": True, "share_dir": str(SHARE_DIR), "max_age_hours": MAX_AGE_HOURS}


# StaticFiles supports HTTP Range requests, so links also work fine for
# streaming/previewing the video in a browser before downloading, at full
# original quality (files are served byte-for-byte, no re-encoding).
app.mount("/files", StaticFiles(directory=str(SHARE_DIR)), name="files")


def _cleanup_loop():
    max_age_seconds = MAX_AGE_HOURS * 3600
    while True:
        try:
            now = time.time()
            removed = 0
            for f in SHARE_DIR.iterdir():
                if not f.is_file():
                    continue
                age = now - f.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError as e:
                        logger.warning("Could not delete expired file %s: %s", f, e)
            if removed:
                logger.info("Cleanup: removed %d file(s) older than %.1fh", removed, MAX_AGE_HOURS)
        except Exception:
            logger.exception("Cleanup loop error")
        time.sleep(CLEANUP_INTERVAL_SECONDS)


def start_cleanup_thread():
    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()


start_cleanup_thread()

if __name__ == "__main__":
    logger.info("Serving %s at http://%s:%s/files/ (files expire after %.1fh)", SHARE_DIR, HOST, PORT, MAX_AGE_HOURS)
    uvicorn.run(app, host=HOST, port=PORT)
