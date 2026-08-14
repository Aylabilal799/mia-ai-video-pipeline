"""tasks.py — Celery background tasks for standard video and Mia video generation.
"""
import os
import shutil
import logging
import subprocess

from celery import Celery

from mia_pipeline import generate_mia_video
import job_store
from youtube_tasks import upload_to_youtube_task

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PUBLIC_SHARE_DIR = os.getenv("PUBLIC_SHARE_DIR", "/var/www/downloads")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000/downloads")

# Set to "false" in .env to disable automatic YouTube upload after generation
# (e.g. while testing) without touching any other code.
AUTO_UPLOAD_YOUTUBE = os.getenv("AUTO_UPLOAD_YOUTUBE", "true").lower() == "true"
YOUTUBE_DEFAULT_PRIVACY = os.getenv("YOUTUBE_DEFAULT_PRIVACY", "private")

app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)


def extract_video_thumbnail(video_path: str, output_dir: str, task_id: str) -> str:
    """Extracts a non-black frame around 2.5 seconds into the final generated video."""
    thumb_filename = f"thumb_{task_id}.jpg"
    thumb_path = os.path.join(output_dir, thumb_filename)
    timestamps = ["00:00:02.5", "00:00:01.5", "00:00:03.5"]
    for ts in timestamps:
        try:
            cmd = [
                "ffmpeg", "-y", "-ss", ts, "-i", video_path,
                "-vframes", "1", "-q:v", "2", thumb_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 5000:
                logger.info(f"[Thumbnail] Extracted valid thumbnail frame at {ts} -> {thumb_filename}")
                return thumb_filename
        except Exception as e:
            logger.warning(f"[Thumbnail] Extraction attempt at {ts} failed: {e}")
    try:
        cmd = ["ffmpeg", "-y", "-ss", "00:00:00.5", "-i", video_path, "-vframes", "1", "-q:v", "2", thumb_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        if os.path.exists(thumb_path):
            return thumb_filename
    except Exception as e:
        logger.error(f"[Thumbnail] Fallback frame extraction failed: {e}")
    return ""


@app.task(bind=True)
def generate_video_task(self, script: str, user_id: str):
    """Standard video task for non-Mia manuscript requests."""
    work_dir = os.path.join("/tmp", f"video_{self.request.id}")
    os.makedirs(work_dir, exist_ok=True)
    try:
        target_dir = os.path.join(PUBLIC_SHARE_DIR, self.request.id)
        os.makedirs(target_dir, exist_ok=True)
        target_video = os.path.join(target_dir, "final_video.mp4")
        raw_path = os.path.join(work_dir, "final_video.mp4")
        if os.path.exists(raw_path):
            shutil.move(raw_path, target_video)
            return f"{PUBLIC_BASE_URL}/{self.request.id}/final_video.mp4"
        return None
    except Exception as exc:
        logger.exception(f"[Video Task] Failed: {exc}")
        raise exc


@app.task(bind=True)
def generate_mia_video_task(self, topic_or_script: str, category: str, user_id: str, is_raw_script: bool = False):
    """Celery task executing Mia's mini-vlog generation pipeline."""
    work_dir = os.path.join("/tmp", f"mia_{self.request.id}")
    os.makedirs(work_dir, exist_ok=True)
    try:
        final_video_path, seo_path, topic, category, script = generate_mia_video(
            topic_or_script=topic_or_script,
            category=category,
            work_dir=work_dir,
            task=self,
            is_raw_script=is_raw_script,
        )
        target_dir = os.path.join(PUBLIC_SHARE_DIR, self.request.id)
        os.makedirs(target_dir, exist_ok=True)
        target_video = os.path.join(target_dir, "final_video.mp4")
        target_seo = os.path.join(target_dir, "mia_seo.txt")
        shutil.move(final_video_path, target_video)
        shutil.move(seo_path, target_seo)

        video_url = f"{PUBLIC_BASE_URL}/{self.request.id}/final_video.mp4"
        seo_url = f"{PUBLIC_BASE_URL}/{self.request.id}/mia_seo.txt"

        thumb_filename = extract_video_thumbnail(target_video, target_dir, self.request.id)
        thumbnail_url = f"{PUBLIC_BASE_URL}/{self.request.id}/{thumb_filename}" if thumb_filename else None

        # --- YouTube upload stage -------------------------------------------------
        # target_video is the EXACT local VPS filesystem path the finished MP4 was
        # just moved to above -- this is the only path the uploader ever receives.
        # Persist it to the durable job store (Celery/Redis results expire) and
        # hand off to a separate Celery task so upload retries never re-run
        # generation. This does not change anything above this point.
        job_store.create_job(
            job_id=self.request.id,
            video_path=target_video,
            seo_path=target_seo,
            topic=topic,
            category=category,
            script=script,
            discord_user_id=user_id,
            work_dir=work_dir,
            status="generated",
        )
        if AUTO_UPLOAD_YOUTUBE:
            upload_to_youtube_task.delay(self.request.id, privacy_status=YOUTUBE_DEFAULT_PRIVACY)
        # --- end YouTube upload stage ----------------------------------------------

        return {
            "video_url": video_url,
            "thumbnail_url": thumbnail_url,
            "seo_url": seo_url,
            "topic": topic,
            "category": category,
            "script": script,
        }
    except Exception as exc:
        logger.exception(f"[Mia Task] Failed: {exc}")
        raise exc
