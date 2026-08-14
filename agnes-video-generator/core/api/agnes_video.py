"""core.api.agnes_video — Agnes Video API 封装（从 core/video_generator.py 迁移）"""

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
from typing import List, Optional

import requests

from core.api.error_collector import collect_error, collect_error_from_exception
from core.api.rate_limiter import get_rate_limiter
from core.config import get_agnes_base_url, get_agnes_api_root
from utils.video import download_video

logger = logging.getLogger(__name__)

DURATION_PRESETS = {
    5: (121, 24),
    10: (241, 24),
    15: (361, 24),
    18: (409, 24),   # capped at 409 (API max for 720p); actual ~17s
    20: (409, 24),   # capped at 409 (API max for 720p); actual ~17s
}

# 图片上传 429 退避间隔基数（秒）：delay = 基数 * (attempt + 1)
_UPLOAD_RETRY_BASE_DELAY_SECONDS = 30


class VideoOutput:
    def __init__(self, fmt: str, ext: str, data: str):
        self.fmt = fmt
        self.ext = ext
        self.data = data

    def save(self, path: str) -> None:
        if self.fmt == "url":
            download_video(self.data, path)
        else:
            with open(path, "wb") as f:
                f.write(self.data if isinstance(self.data, bytes) else self.data.encode())


class AgnesVideoAPI:
    """Agnes Video 生成 API 封装（t2v / i2v / ti2vid / keyframes）。"""

    def __init__(
        self,
        api_key: str,
        model: str = "agnes-video-v2.0",
        default_duration: int = 5,
        max_retries: int = 5,
        retry_base_delay: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.default_duration = default_duration
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.shutdown_event = None
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _path_to_b64(self, path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _sha256_of_local_path(path: str) -> Optional[str]:
        """Best-effort SHA256 of a local reference image file, for identity
        traceability logging (IDENTITY TRACE requirement). Returns None for
        non-local refs (http(s) URL, data: URI) or a path that doesn't
        exist -- this is diagnostic only and must never block a real
        submission.
        """
        try:
            if not path or path.startswith(("http://", "https://", "data:")):
                return None
            if not os.path.exists(path):
                return None
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    async def _resolve_image_ref(self, ref: str) -> str:
        if ref.startswith(("http://", "https://")):
            return ref
        if ref.startswith("data:"):
            return ref
        if os.path.exists(ref):
            url_file = ref + ".url"
            _URL_CACHE_MAX_AGE = 3600  # 1 小时
            if os.path.exists(url_file):
                try:
                    with open(url_file, "r", encoding="utf-8") as f:
                        cache_data = json.loads(f.read())
                    cached_url = cache_data.get("url", "")
                    cached_ts = cache_data.get("ts", 0)
                    age = time.time() - cached_ts
                    if cached_url and age < _URL_CACHE_MAX_AGE:
                        logger.info(
                            f"[AgnesVideo] Using cached hosted URL (age={age:.0f}s): "
                            f"{cached_url[:80]}..."
                        )
                        return cached_url
                    if cached_url:
                        logger.info(
                            f"[AgnesVideo] Cached URL expired (age={age:.0f}s), re-uploading"
                        )
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug(f"[AgnesVideo] Failed to read cached URL: {e}")
                except Exception as e:
                    logger.debug(f"[AgnesVideo] Failed to read legacy URL cache: {e}")
            url = await self._upload_image_to_url(ref)
            if url:
                try:
                    cache_data = {"url": url, "ts": time.time()}
                    tmp_file = url_file + ".tmp"
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_file, url_file)
                except Exception as e:
                    logger.debug(f"[AgnesVideo] Failed to cache URL: {e}")
                return url
            logger.warning("[AgnesVideo] Image upload failed, falling back to base64.")
            return self._path_to_b64(ref)
        return ref

    async def _upload_image_to_url(self, image_path: str, retries: int = 3) -> Optional[str]:
        for attempt in range(retries):
            if self.shutdown_event and self.shutdown_event.is_set():
                logger.info("[AgnesVideo] Image upload cancelled by shutdown")
                return None
            try:
                b64_data = self._path_to_b64(image_path)
                payload = {
                    "model": "agnes-image-2.1-flash",
                    "prompt": "Keep the image exactly as it is",
                    "n": 1,
                    "size": "1024x1024",
                    "extra_body": {
                        "response_format": "url",
                        "image": b64_data,
                    },
                }
                logger.info(f"[AgnesVideo] Uploading image to hosted URL (attempt {attempt + 1}/{retries})...")
                await asyncio.to_thread(get_rate_limiter().acquire)
                resp = await asyncio.to_thread(
                    requests.post,
                    f"{get_agnes_base_url()}/images/generations",
                    headers=self.headers,
                    json=payload,
                    timeout=(30, 120),
                )
                if resp.status_code == 429:
                    delay = _UPLOAD_RETRY_BASE_DELAY_SECONDS * (attempt + 1)
                    logger.warning(f"[AgnesVideo] Image upload 429, retry in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                result = resp.json()
                data_list = result.get("data", [])
                if data_list:
                    url = data_list[0].get("url", "")
                    if url:
                        logger.info(f"[AgnesVideo] Image uploaded to hosted URL: {url[:80]}...")
                        return url
            except Exception as e:
                logger.warning(f"[AgnesVideo] Image upload attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(15)
        return None

    _FRAME_LIMITS = {
        "1080p": 169,
        "720p": 409,
        "480p": 961,
    }

    @staticmethod
    def _get_max_frames(width: int, height: int) -> int:
        """Get the maximum allowed num_frames for the given resolution."""
        pixels = width * height
        if pixels > 1280 * 720:
            return 169   # 1080p tier
        elif pixels > 854 * 480:
            return 409   # 720p tier
        else:
            return 961   # 480p tier

    def _get_frame_config(self, duration: Optional[float] = None,
                          width: int = 1152, height: int = 768) -> tuple:
        # Round duration to integer seconds safely for frame calculation
        d = max(1, int(round(duration))) if duration is not None else self.default_duration
        max_nf = self._get_max_frames(width, height)
        if d in DURATION_PRESETS:
            nf, fr = DURATION_PRESETS[d]
            if nf <= max_nf:
                return nf, fr
            logger.warning(
                f"[AgnesVideo] Duration preset {d}s has {nf} frames, "
                f"exceeds {max_nf} for {width}x{height}. Capping."
            )
            return max_nf, fr
        best = None
        for nf in range(9, min(410, max_nf + 1), 8):
            fr = round(nf / d)
            if 1 <= fr <= 60:
                best = (nf, fr)
        return best or DURATION_PRESETS[5]

    async def _poll_task(self, video_id: str, interval: int = 60,
                          max_poll_duration: int = 1800,
                          max_consecutive_failures: int = 10,
                          progress_callback=None) -> dict:
        last_status = ""
        poll_count = 0
        consecutive_failures = 0
        start_time = asyncio.get_event_loop().time()
        curl_cmd = (
            f'curl -s -H "Authorization: Bearer $AGNES_API_KEY" '
            f'"{get_agnes_api_root()}/agnesapi?video_id={video_id}"'
        )
        while True:
            if self.shutdown_event and self.shutdown_event.is_set():
                raise RuntimeError("Video generation cancelled by user")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_poll_duration:
                error_msg = (
                    f"[AgnesVideo] Polling timed out after {max_poll_duration}s "
                    f"for video {video_id[:16]}"
                )
                collect_error(
                    "video", "poll_task",
                    prompt=curl_cmd,
                    error_type="PollingTimeout",
                    error_message=error_msg,
                    extra={"video_id": video_id[:16], "elapsed_s": int(elapsed)},
                )
                raise RuntimeError(error_msg)

            try:
                if poll_count % 10 == 0:
                    logger.info(f"[AgnesVideo] Polling video {video_id[:16]}... (poll #{poll_count + 1}, elapsed {elapsed:.0f}s)")
                await asyncio.to_thread(get_rate_limiter().acquire)
                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        requests.get,
                        f"{get_agnes_api_root()}/agnesapi?video_id={video_id}",
                        headers=self.headers,
                        timeout=15,
                    ),
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status", "")
                progress = result.get("progress", 0)
                poll_count += 1
                consecutive_failures = 0

                if status != last_status:
                    logger.info(f"[AgnesVideo] Video {video_id[:16]}... status={status} progress={progress}%")
                    last_status = status

                if progress_callback:
                    progress_callback(status, progress, curl_cmd)

                if status in ("completed", "COMPLETED"):
                    return result

                if status in ("failed", "FAILED"):
                    err = result.get("error") or "unknown error"
                    error_msg = f"Video generation failed: {err}"
                    collect_error(
                        "video", "poll_task",
                        prompt=curl_cmd,
                        error_type="VideoFailed",
                        error_message=error_msg,
                        response_body=resp.text,
                        extra={"video_id": video_id[:16], "status": status},
                    )
                    raise RuntimeError(error_msg)
            except (requests.exceptions.RequestException, asyncio.TimeoutError) as e:
                consecutive_failures += 1
                logger.warning(
                    f"[AgnesVideo] Poll error ({consecutive_failures}/{max_consecutive_failures}): {e}"
                )
                collect_error_from_exception(
                    "video", "poll_task",
                    exc=e, prompt=curl_cmd,
                    retry_count=consecutive_failures,
                    extra={"video_id": video_id[:16], "poll_count": poll_count},
                )
                if consecutive_failures >= max_consecutive_failures:
                    error_msg = (
                        f"[AgnesVideo] Polling failed after {max_consecutive_failures} "
                        f"consecutive errors for video {video_id[:16]}"
                    )
                    collect_error_from_exception(
                        "video", "poll_task",
                        exc=e, prompt=curl_cmd,
                        retry_count=max_consecutive_failures,
                        extra={"video_id": video_id[:16], "poll_count": poll_count},
                    )
                    raise RuntimeError(error_msg)

            await asyncio.sleep(interval)

    async def _submit_with_retry(self, payload: dict, mode_desc: str) -> str:
        frame_reductions_left = 2
        for attempt in range(self.max_retries):
            if self.shutdown_event and self.shutdown_event.is_set():
                raise RuntimeError("Video generation cancelled by user")
            try:
                logger.info(f"[AgnesVideo] Submitting {mode_desc} (attempt {attempt + 1}/{self.max_retries})...")
                await asyncio.to_thread(get_rate_limiter().acquire)
                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        requests.post,
                        f"{get_agnes_base_url()}/videos",
                        headers=self.headers,
                        json=payload,
                        timeout=(15, 60),
                    ),
                    timeout=90,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    video_id = result.get("video_id") or result.get("task_id") or result.get("id")
                    if video_id:
                        return video_id

                if resp.status_code == 429:
                    delay = self.retry_base_delay * (attempt + 1)
                    logger.warning(
                        f"[AgnesVideo] 429 rate limit on {mode_desc}, "
                        f"retry {attempt + 1}/{self.max_retries} in {delay:.0f}s..."
                    )
                    collect_error(
                        "video", "submit_video",
                        prompt=payload.get("prompt", ""),
                        error_type="RateLimit429",
                        error_message="HTTP 429: rate limited",
                        status_code=429,
                        response_body=resp.text,
                        retry_count=attempt + 1,
                        extra={"mode": mode_desc},
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 500:
                    delay = self.retry_base_delay * (attempt + 1)
                    logger.warning(
                        f"[AgnesVideo] {resp.status_code} server error on {mode_desc}, "
                        f"retry {attempt + 1}/{self.max_retries} in {delay:.0f}s..."
                    )
                    collect_error(
                        "video", "submit_video",
                        prompt=payload.get("prompt", ""),
                        error_type=f"HTTP{resp.status_code}",
                        error_message=f"HTTP {resp.status_code}: server error",
                        status_code=resp.status_code,
                        response_body=resp.text,
                        retry_count=attempt + 1,
                        extra={"mode": mode_desc},
                    )
                    await asyncio.sleep(delay)
                    continue

                error_text = resp.text[:500]
                if (resp.status_code == 400
                        and "num_frames" in error_text
                        and frame_reductions_left > 0):
                    old_nf = payload.get("num_frames", 0)
                    new_nf = max(int(old_nf * 0.7), 49)
                    logger.warning(
                        f"[AgnesVideo] 400 num_frames error ({old_nf} frames), "
                        f"reducing to {new_nf} and retrying "
                        f"({frame_reductions_left} reductions left)..."
                    )
                    collect_error(
                        "video", "submit_video",
                        prompt=payload.get("prompt", ""),
                        error_type="NumFramesExceeded",
                        error_message=f"HTTP 400: num_frames {old_nf} exceeded, reducing to {new_nf}",
                        status_code=400,
                        response_body=resp.text,
                        retry_count=attempt + 1,
                        extra={"mode": mode_desc, "old_nf": old_nf, "new_nf": new_nf},
                    )
                    payload["num_frames"] = new_nf
                    frame_reductions_left -= 1
                    continue

                logger.error(f"[AgnesVideo] HTTP {resp.status_code}: {error_text}")
                collect_error(
                    "video", "submit_video",
                    prompt=payload.get("prompt", ""),
                    error_type="HTTPError",
                    error_message=f"HTTP {resp.status_code}: {error_text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                    retry_count=attempt + 1,
                    extra={"mode": mode_desc},
                )
                raise RuntimeError(f"Agnes video submit failed (HTTP {resp.status_code}): {error_text}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                        asyncio.TimeoutError) as e:
                collect_error_from_exception(
                    "video", "submit_video",
                    exc=e, prompt=payload.get("prompt", ""),
                    retry_count=attempt + 1,
                    extra={"mode": mode_desc},
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (attempt + 1)
                    logger.warning(
                        f"[AgnesVideo] {type(e).__name__} on {mode_desc}, "
                        f"retry {attempt + 1}/{self.max_retries} in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        collect_error(
            "video", "submit_video",
            prompt=payload.get("prompt", ""),
            error_type="RetriesExhausted",
            error_message=f"{mode_desc}: max retries ({self.max_retries}) exceeded",
            retry_count=self.max_retries,
            extra={"mode": mode_desc},
        )
        raise RuntimeError(
            f"[AgnesVideo] {mode_desc}: max retries ({self.max_retries}) exceeded"
        )

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        duration: Optional[float] = None,
        width: int = 1152,
        height: int = 768,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        progress_callback=None,
        scene_label: Optional[str] = None,
        **kwargs,
    ) -> VideoOutput:
        video_id = await self.submit_video(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            duration=duration,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=negative_prompt,
            scene_label=scene_label,
            **kwargs,
        )
        return await self.wait_for_video(video_id, progress_callback)

    async def submit_video(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        duration: Optional[float] = None,
        width: int = 1152,
        height: int = 768,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        scene_label: Optional[str] = None,
        **kwargs,
    ) -> str:
        num_frames, frame_rate = self._get_frame_config(duration, width, height)

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }

        if seed is not None:
            payload["seed"] = seed
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        resolved_refs = []
        for p in reference_image_paths:
            resolved_refs.append(await self._resolve_image_ref(p))
        n_refs = len(resolved_refs)

        # Explicit mode routing: Exactly 1 image triggers ti2vid (image-to-video)
        if n_refs == 0:
            mode_desc = "text-to-video"
        elif n_refs == 1:
            payload["image"] = resolved_refs[0]
            payload["mode"] = "ti2vid"
            mode_desc = "image-to-video"
        else:
            payload["extra_body"] = {
                "image": resolved_refs,
                "mode": "keyframes",
            }
            mode_desc = f"keyframes ({n_refs} frames)"

        # IDENTITY TRACE: log scene number, reference image path(s), SHA256
        # of each local reference file, ref count, and the generation mode
        # about to be used -- required so identity drift between scenes can
        # be diagnosed from logs alone (path reused vs. actually different
        # bytes vs. silently zero references / wrong mode).
        ref_fingerprints = [
            {"path": p, "sha256": self._sha256_of_local_path(p)}
            for p in reference_image_paths
        ]
        logger.info(
            "[AgnesVideo][IdentityTrace] scene=%s mode=%s n_refs=%d refs=%s",
            scene_label or "unlabeled", mode_desc, n_refs, ref_fingerprints,
        )

        logger.info(f"[AgnesVideo] {mode_desc}: {prompt[:80]}...")

        video_id = await self._submit_with_retry(payload, mode_desc)
        logger.info(f"[AgnesVideo] Video submitted: {video_id[:20]}...")
        return video_id

    async def wait_for_video(self, video_id: str, progress_callback=None) -> VideoOutput:
        final = await self._poll_task(video_id, progress_callback=progress_callback)

        video_url = (
            final.get("remixed_from_video_id")
            or final.get("video_url")
            or final.get("url")
        )
        if not video_url:
            data = final.get("data", {})
            if isinstance(data, dict):
                video_url = data.get("video_url") or data.get("url")
            if not video_url:
                raise RuntimeError(f"Agnes video: no URL in completed task: {final}")

        logger.info(f"[AgnesVideo] Done: {video_url[:80]}...")
        return VideoOutput(fmt="url", ext="mp4", data=video_url)
