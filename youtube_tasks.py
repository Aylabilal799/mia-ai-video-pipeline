"""youtube_tasks.py — Celery tasks for the YouTube upload stage.

Deliberately separate from tasks.py's video-generation tasks: this stage
runs AFTER a job already exists in job_store, and only ever touches the
local MP4 path that's already recorded there. It never talks to Agnes,
never re-generates anything, and never changes the generation pipeline.
"""

import os
import shutil
import logging

from celery import Celery

import job_store
from youtube_uploader import (
    verify_video_file,
    parse_mia_seo_txt,
    build_youtube_metadata,
    upload_video,
    VideoNotReadyError,
)

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("youtube_tasks", broker=REDIS_URL, backend=REDIS_URL)

MAX_UPLOAD_ATTEMPTS = int(os.getenv("YOUTUBE_MAX_UPLOAD_ATTEMPTS", "5"))


@app.task(bind=True)
def upload_to_youtube_task(self, job_id: str, privacy_status: str = None, publish_at: str = None):
    """Uploads the MP4 belonging to `job_id`. The local path is read straight
    out of the persistent job record (job_store), which tasks.py populated
    with the exact same path it just moved the finished file to."""
    job = job_store.get_job(job_id)
    if not job:
        logger.error("[YouTube Upload] No job record found for job_id=%s", job_id)
        return

    video_path = job["video_path"]
    seo_path = job["seo_path"]
    attempts = job_store.increment_attempts(job_id)
    job_store.update_job(job_id, status="uploading")

    try:
        # 1 & 2. Confirm the existing job's final MP4 is present and complete.
        verify_video_file(video_path)

        # 3. SEO -- reuse what the pipeline already generated, don't re-invent it.
        seo = parse_mia_seo_txt(seo_path)
        metadata = build_youtube_metadata(seo, job.get("topic") or "", job.get("category") or "")

        # 4 & 5. Upload via YouTube Data API with title/description/tags/privacy.
        result = upload_video(
            video_path=video_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            privacy_status=privacy_status or job.get("privacy_status"),
            publish_at=publish_at or job.get("publish_at"),
        )

        job_store.update_job(
            job_id,
            status="uploaded",
            youtube_video_id=result["video_id"],
            youtube_url=result["url"],
            last_error=None,
        )
        logger.info("[YouTube Upload] job=%s -> %s", job_id, result["url"])

        # 6. Clean up ONLY the temp/intermediate work dir, never the job
        # record and never the archived final MP4 in PUBLIC_SHARE_DIR.
        work_dir = job.get("work_dir")
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info("[YouTube Upload] Cleaned up temp dir %s", work_dir)

        return result

    except VideoNotReadyError as e:
        # File missing/incomplete: preserve everything, mark for retry.
        logger.warning("[YouTube Upload] job=%s not ready: %s", job_id, e)
        job_store.update_job(job_id, status="upload_pending", last_error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=MAX_UPLOAD_ATTEMPTS)

    except Exception as e:
        # 7. Any other failure (auth, quota, network): keep the MP4 and job
        # record, mark upload_pending for retry instead of losing the video.
        logger.exception("[YouTube Upload] job=%s failed: %s", job_id, e)
        if attempts >= MAX_UPLOAD_ATTEMPTS:
            job_store.update_job(job_id, status="upload_failed", last_error=str(e))
        else:
            job_store.update_job(job_id, status="upload_pending", last_error=str(e))
            raise self.retry(exc=e, countdown=120, max_retries=MAX_UPLOAD_ATTEMPTS)


@app.task
def retry_pending_uploads_task():
    """Re-enqueues every job currently marked upload_pending. Call this
    periodically (cron/celery-beat) or manually after fixing an auth/quota
    issue -- it never re-runs video generation, only the upload step."""
    pending = job_store.list_by_status("upload_pending")
    for job in pending:
        logger.info("[YouTube Upload] Re-queuing pending job %s", job["job_id"])
        upload_to_youtube_task.delay(job["job_id"])
    return {"requeued": len(pending)}
