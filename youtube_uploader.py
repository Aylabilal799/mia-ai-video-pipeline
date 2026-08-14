"""youtube_uploader.py — uploads an already-generated local MP4 to YouTube
via the YouTube Data API v3 resumable upload.

Takes the video's LOCAL VPS FILESYSTEM PATH directly (never a URL). The path
must come from the existing job record (job_store), which in turn was
populated verbatim from tasks.py's `target_video` -- the same variable the
generation pipeline already uses to move the finished file into
PUBLIC_SHARE_DIR. This module does not construct or guess that path.
"""

import os
import logging

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from youtube_auth import get_credentials

logger = logging.getLogger(__name__)

DEFAULT_PRIVACY_STATUS = os.getenv("YOUTUBE_DEFAULT_PRIVACY", "private")  # private|unlisted|public
DEFAULT_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "22")  # 22 = "People & Blogs"

# YouTube limits
_TITLE_MAX = 100
_DESCRIPTION_MAX = 5000
_TAGS_MAX_TOTAL_CHARS = 450  # keep headroom under the 500-char API limit


class VideoNotReadyError(Exception):
    """Raised when the local MP4 doesn't exist yet or looks incomplete."""


def verify_video_file(video_path: str, min_bytes: int = 100_000) -> None:
    """Confirms the final MP4 the job record points to is actually present
    and not a truncated/in-progress write before we hand it to the API."""
    if not video_path or not os.path.isfile(video_path):
        raise VideoNotReadyError(f"Final MP4 not found at expected job path: {video_path}")

    size = os.path.getsize(video_path)
    if size < min_bytes:
        raise VideoNotReadyError(
            f"Final MP4 at {video_path} is only {size} bytes -- looks incomplete."
        )

    # A quick, dependency-free sanity check that this is really an MP4/MOV
    # container (ftyp box), to catch a half-written or corrupt file.
    with open(video_path, "rb") as f:
        header = f.read(12)
    if len(header) < 8 or header[4:8] != b"ftyp":
        raise VideoNotReadyError(
            f"File at {video_path} does not look like a valid MP4 (missing ftyp box)."
        )


def parse_mia_seo_txt(seo_path: str) -> dict:
    """Parses the LABEL / value block format written by
    mia_llm.seo_package_to_txt() in the existing pipeline. Reuses the
    pipeline's own SEO generation instead of generating new SEO here."""
    if not seo_path or not os.path.isfile(seo_path):
        return {}

    label_to_key = {
        "VIDEO TITLE": "video_title",
        "DESCRIPTION": "description",
        "HOOK": "hook",
        "KEYWORDS": "keywords",
        "TAGS": "tags",
        "HASHTAGS": "hashtags",
        "SEO KEYWORDS": "seo_keywords",
        "TARGET SEARCH TERMS": "target_search_terms",
        "THUMBNAIL TEXT": "thumbnail_text",
        "VIDEO CATEGORY": "video_category",
        "CONTENT ANGLE": "content_angle",
        "SHORT DESCRIPTION": "short_description",
        "LONG DESCRIPTION": "long_description",
        "CALL TO ACTION": "call_to_action",
    }

    lines = [ln.rstrip("\n") for ln in open(seo_path, encoding="utf-8").readlines()]
    data, i = {}, 0
    while i < len(lines):
        label = lines[i].strip()
        if label in label_to_key:
            value = lines[i + 1].strip() if i + 1 < len(lines) else ""
            data[label_to_key[label]] = value
            i += 2
        else:
            i += 1
    return data


def build_youtube_metadata(seo: dict, topic: str, category: str) -> dict:
    """Maps the pipeline's SEO dict onto YouTube's title/description/tags."""
    title = (seo.get("video_title") or f"{topic} | Mia Mini Vlog").strip()[:_TITLE_MAX]

    description_parts = [
        seo.get("long_description") or seo.get("description") or "",
        "",
        seo.get("call_to_action") or "",
        "",
        seo.get("hashtags") or "",
    ]
    description = "\n".join(p for p in description_parts if p).strip()[:_DESCRIPTION_MAX]

    raw_tags = [t.strip() for t in (seo.get("tags") or "").split(",") if t.strip()]
    tags, total = [], 0
    for t in raw_tags:
        if total + len(t) + 1 > _TAGS_MAX_TOTAL_CHARS:
            break
        tags.append(t)
        total += len(t) + 1

    return {"title": title or "Mia Mini Vlog", "description": description, "tags": tags}


def upload_video(video_path: str, title: str, description: str, tags: list,
                  privacy_status: str = None, publish_at: str = None,
                  category_id: str = None) -> dict:
    """Uploads video_path (a LOCAL filesystem path) to YouTube via resumable
    upload. Returns {'video_id', 'url'}. Raises on failure -- the caller
    (youtube_tasks.py) is responsible for marking the job upload_pending."""
    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    status = {"privacyStatus": privacy_status or DEFAULT_PRIVACY_STATUS}
    if publish_at:
        # Scheduling requires privacyStatus == "private" plus publishAt (RFC3339 UTC)
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id or DEFAULT_CATEGORY_ID,
        },
        "status": status,
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True, chunksize=1024 * 1024 * 8)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status_chunk, response = request.next_chunk()
        if status_chunk:
            logger.info("[YouTube Upload] %s%% uploaded", int(status_chunk.progress() * 100))

    video_id = response["id"]
    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}"}
