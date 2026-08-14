"""mia_pipeline.py — end-to-end orchestration for a single Mia video:

    topic -> script -> Agnes manuscript pipeline (identity-anchored scenes,
    fixed voice, 38px green karaoke captions) -> Shorts postprocess
    -> YouTube SEO .txt

Reuses the low-level Agnes-server plumbing already in video_generator.py
(server startup, API-key failover pool, task polling, video download,
Shorts postprocessing) instead of duplicating it -- only what's specific to
Mia (anchor image, manuscript payload, script/SEO generation) lives here.
"""
import logging
import os
import time
from pathlib import Path

import requests

import mia_config as cfg
import mia_llm
from video_generator import (
    AGNES_API_KEYS,
    AGNES_BASE_URL,
    ensure_agnes_server_running,
    _push_api_key,
    wait_for_task,
    download_video,
    postprocess_for_shorts,
)

logger = logging.getLogger(__name__)

_IMAGE_VALID_SIZES = {
    "1024x1024", "768x1152", "1152x768", "768x1344",
    "1344x768", "1792x1024", "1024x1792",
}

# Extra instructions layered on top of MIA_CHARACTER_DESCRIPTION specifically
# for the ONE-TIME master reference image, so it's a clean i2i identity
# anchor (front-facing, unobstructed, evenly lit) -- see Agnes's own
# characters.py, which asks for the same qualities in a reference image.
_ANCHOR_IMAGE_SUFFIX = (
    ", clear front-facing face with eyes and mouth fully visible, "
    "no hands or hair blocking the face, even soft studio lighting with no "
    "harsh shadows, neutral confident expression, full-body or three-quarter "
    "view, plain neutral background, high detail, photorealistic"
)


def ensure_mia_anchor_image(api_key: str, force: bool = False) -> str:
    """Makes sure assets/mia_anchor.png exists. Generates it ONCE via Agnes's
    image endpoint if missing; never regenerates an existing anchor unless
    `force=True` is explicitly passed (e.g. a manual /mia-reset-face command),
    since silently regenerating it is exactly what breaks consistency."""
    out_path = Path(cfg.MIA_REFERENCE_IMAGE)
    if out_path.exists() and not force:
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_agnes_server_running(api_key)
    _push_api_key(api_key)

    prompt = cfg.MIA_CHARACTER_DESCRIPTION + _ANCHOR_IMAGE_SUFFIX
    resp = requests.post(
        f"{AGNES_BASE_URL}/api/image/generate",
        data={"prompt": prompt, "size": "768x1344"},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Mia anchor image generation failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Agnes image response didn't include a task id: {data}")

    img_resp = requests.get(f"{AGNES_BASE_URL}/api/image/{task_id}", timeout=60)
    if img_resp.status_code >= 400:
        raise RuntimeError(f"Failed to download Mia anchor image ({img_resp.status_code}): {img_resp.text}")

    with open(out_path, "wb") as f:
        f.write(img_resp.content)

    logger.info("[Mia] Generated permanent anchor image -> %s", out_path)
    return str(out_path)


def submit_mia_manuscript_task(script: str, category: str) -> str:
    """Submits Mia's script to Agnes's manuscript pipeline with every
    identity/voice/caption setting locked to Mia's fixed configuration."""
    logger.info(f"[MIA] Reference image: {cfg.MIA_REFERENCE_IMAGE}")
    logger.info(f"[MIA] Character seed: {cfg.MIA_SEED}")
    logger.info(f"[TTS] Voice: {cfg.MIA_VOICE}")
    logger.info(f"[CAPTIONS] Font size: {cfg.MIA_CAPTION_FONT_SIZE}")
    logger.info(f"[CAPTIONS] Bottom margin: {cfg.MIA_CAPTION_BOTTOM_MARGIN_PX}px "
                f"(final) / {cfg.MIA_CAPTION_BOTTOM_MARGIN_PX_GEN}px (generation res)")

    # Explicit negative prompt to reinforce single-character stability across scenes
    identity_negative_prompt = (
        "different person, altered face, morphing features, identity shift, "
        "changing facial structure, distorted face, inconsistent character"
    )

    payload = {
        "manuscript_text": script,
        "reference_image": cfg.MIA_REFERENCE_IMAGE,
        "character_style": cfg.MIA_CHARACTER_STYLE_FOR_SCENES,
        "character_seed": cfg.MIA_SEED,
        "negative_prompt": identity_negative_prompt,
        "video_width": cfg.MIA_GENERATION_WIDTH,
        "video_height": cfg.MIA_GENERATION_HEIGHT,
        "audio_voice": cfg.MIA_VOICE,
        "audio_rate": cfg.MIA_VOICE_RATE,
        "audio_lang": cfg.MIA_VOICE_LANG,
        "subtitle_enabled": True,
        "subtitle_style_mode": "fixed",
        "subtitle_font": cfg.MIA_CAPTION_FONT,
        "subtitle_fontsize": cfg.MIA_CAPTION_FONT_SIZE,
        "subtitle_color": cfg.MIA_CAPTION_COLOR,
        "subtitle_position": cfg.MIA_CAPTION_POSITION,
        "subtitle_stroke_color": cfg.MIA_CAPTION_STROKE_COLOR,
        "subtitle_stroke_width": cfg.MIA_CAPTION_STROKE_WIDTH,
        "subtitle_bg_color": cfg.MIA_CAPTION_BG_COLOR,
        "subtitle_highlight_color": cfg.MIA_CAPTION_HIGHLIGHT_COLOR,
        "subtitle_max_width_ratio": cfg.MIA_CAPTION_MAX_WIDTH_RATIO,
    }
    resp = requests.post(f"{AGNES_BASE_URL}/api/tasks/manuscript", data=payload, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Agnes rejected Mia's manuscript task ({resp.status_code}): {resp.text}")
    data = resp.json()
    task_id = data.get("task_id") or data.get("id") or (data.get("task") or {}).get("id")
    if not task_id:
        raise RuntimeError(f"Agnes response didn't include a task id: {data}")
    return task_id


def _generate_with_key(topic, category, script, work_dir, api_key, key_label, task=None):
    if task:
        task.update_state(state="PROGRESS", meta={
            "stage": f"Starting Agnes AI server ({key_label})...", "progress": 2,
        })
    ensure_agnes_server_running(api_key)
    _push_api_key(api_key)

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Preparing Mia's reference image...", "progress": 4})
    ensure_mia_anchor_image(api_key)

    if task:
        task.update_state(state="PROGRESS", meta={
            "stage": f"Submitting script to Agnes AI ({key_label})...", "progress": 6,
        })
    task_id = submit_mia_manuscript_task(script, category)

    if task:
        task.update_state(state="PROGRESS", meta={
            "stage": "Generating Mia's video (this can take a while)...", "progress": 8,
        })
    wait_for_task(task_id, task)

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Downloading finished video...", "progress": 90})
    raw_path = Path(work_dir) / "raw_video.mp4"
    download_video(task_id, raw_path)

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Enhancing video for Shorts...", "progress": 95})
    final_path = Path(work_dir) / "final_video.mp4"
    postprocess_for_shorts(raw_path, final_path)

    return str(final_path)


def generate_mia_video(topic_or_script: str, category: str, work_dir: str, task=None, is_raw_script: bool = False):
    """High-level entry point: produces (video_path, seo_txt_path, topic, category, script).

    - `topic_or_script`: either a short topic ("morning yoga stretches") or,
      if `is_raw_script=True`, a full ready-to-speak script.
    - `category`: one of mia_config.MIA_CONTENT_CATEGORIES, or None to pick
      one automatically using the configured content-mix weights.
    """
    if not AGNES_API_KEYS:
        raise RuntimeError(
            "No Agnes API key configured. Set AGNES_API_KEYS or AGNES_API_KEY in .env."
        )

    category = category or cfg.pick_category()
    if category not in cfg.MIA_CONTENT_CATEGORIES:
        category = cfg.pick_category()

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Writing Mia's script...", "progress": 1})

    if is_raw_script and topic_or_script.strip():
        script = topic_or_script.strip()
        topic = topic_or_script.strip()[:80]
    else:
        topic = topic_or_script.strip() if topic_or_script and topic_or_script.strip() else mia_llm.generate_topic(category)
        script = mia_llm.generate_script(topic, category)

    last_error = None
    final_path = None
    for i, api_key in enumerate(AGNES_API_KEYS):
        key_label = f"key {i + 1}/{len(AGNES_API_KEYS)}"
        try:
            final_path = _generate_with_key(topic, category, script, work_dir, api_key, key_label, task)
            break
        except Exception as e:
            logger.warning("[Mia] Generation failed with %s: %s", key_label, e)
            last_error = e
            if i < len(AGNES_API_KEYS) - 1:
                if task:
                    task.update_state(state="PROGRESS", meta={
                        "stage": f"{key_label} failed, retrying with next key...", "progress": 2,
                    })
                time.sleep(3)
            continue

    if final_path is None:
        raise RuntimeError(f"All {len(AGNES_API_KEYS)} Agnes API key(s) failed. Last error: {last_error}")

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Generating YouTube SEO package...", "progress": 98})
    seo = mia_llm.generate_seo_package(topic, category, script)
    seo_txt = mia_llm.seo_package_to_txt(seo)
    seo_path = Path(work_dir) / "mia_seo.txt"
    seo_path.write_text(seo_txt, encoding="utf-8")

    if task:
        task.update_state(state="PROGRESS", meta={"stage": "Done!", "progress": 100})

    return final_path, str(seo_path), topic, category, script
