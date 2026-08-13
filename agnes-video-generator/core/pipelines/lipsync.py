"""core/pipelines/lipsync.py — audio-driven lip-sync via Wav2Lip, run as an
isolated subprocess (separate venv, no shared deps with the main project).

This module NEVER reports success just because a subprocess exited 0 or a
file exists at the expected path. It validates:
  1. Wav2Lip's process exited 0.
  2. The output file exists and is non-trivially sized.
  3. The output's duration is within tolerance of the source audio's duration
     (a badly-failed run sometimes emits a near-empty or truncated clip that
     still "exists").
  4. At least one face was detected somewhere in the source video (parsed
     from Wav2Lip's own stderr/stdout, which explicitly says "Face not
     detected!" and raises when detection fails for every frame).

If any check fails, `run_lipsync` raises LipSyncFailure instead of returning
a path. Callers MUST NOT fall back to silently using the original
(non-lip-synced) clip without clearly flagging that the scene has no lip-sync
-- see the retry/skip logic in manuscript_video.py's integration.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Adjust if you installed to a different location in setup_wav2lip.sh.
WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "/root/deepseekyt/wav2lip/Wav2Lip")
WAV2LIP_VENV_PYTHON = os.path.join(WAV2LIP_DIR, "venv", "bin", "python")
WAV2LIP_CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")

# Wav2Lip is slow on CPU. Generous timeout so a legitimately long scene
# doesn't get killed mid-render; tune down if your clips are always short.
_TIMEOUT_SECONDS = 600
_DURATION_TOLERANCE_SEC = 0.75


class LipSyncFailure(Exception):
    """Raised when lip-sync could not be verified as successful."""


@dataclass
class LipSyncResult:
    output_path: str
    duration_sec: float
    render_seconds: float


def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _preflight_check() -> None:
    if not os.path.exists(WAV2LIP_VENV_PYTHON):
        raise LipSyncFailure(
            f"Wav2Lip venv python not found at {WAV2LIP_VENV_PYTHON}. "
            "Run setup_wav2lip.sh first."
        )
    if not os.path.exists(WAV2LIP_CHECKPOINT):
        raise LipSyncFailure(
            f"Wav2Lip checkpoint not found at {WAV2LIP_CHECKPOINT}. "
            "Download wav2lip_gan.pth per setup_wav2lip.sh instructions."
        )


async def run_lipsync(
    face_video_path: str,
    audio_path: str,
    output_path: str,
    *,
    timeout_seconds: int = _TIMEOUT_SECONDS,
) -> LipSyncResult:
    """Run Wav2Lip on a single scene clip. Raises LipSyncFailure on any
    unverified outcome -- callers must handle this explicitly (retry once,
    or skip lip-sync for that scene and log it clearly; never silently claim
    success).
    """
    _preflight_check()

    if not os.path.exists(face_video_path):
        raise LipSyncFailure(f"Source face video missing: {face_video_path}")
    if not os.path.exists(audio_path):
        raise LipSyncFailure(f"Source audio missing: {audio_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        WAV2LIP_VENV_PYTHON, "inference.py",
        "--checkpoint_path", WAV2LIP_CHECKPOINT,
        "--face", face_video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        # Chin padding — Wav2Lip's default crop often clips the chin on
        # close-up vertical vlog framing; extra bottom pad helps.
        "--pads", "0", "20", "0", "0",
        "--resize_factor", "1",
    ]

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=WAV2LIP_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise LipSyncFailure(
            f"Wav2Lip timed out after {timeout_seconds}s on {face_video_path} "
            "(CPU-only inference can be slow -- consider shorter scenes or "
            "raising timeout_seconds)."
        )

    render_seconds = time.monotonic() - start
    log_text = (stdout_bytes or b"").decode(errors="replace")

    if proc.returncode != 0:
        raise LipSyncFailure(
            f"Wav2Lip exited {proc.returncode} on {face_video_path}:\n"
            f"{log_text[-2000:]}"
        )

    # Wav2Lip prints this explicit string when no face is found in ANY frame.
    if "Face not detected" in log_text and not os.path.exists(output_path):
        raise LipSyncFailure(
            f"Wav2Lip could not detect a face in {face_video_path}:\n"
            f"{log_text[-1000:]}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        raise LipSyncFailure(
            f"Wav2Lip reported success but output is missing/tiny: {output_path}\n"
            f"Last log output:\n{log_text[-1000:]}"
        )

    out_duration = _ffprobe_duration(output_path)
    src_duration = _ffprobe_duration(audio_path)
    if out_duration is None:
        raise LipSyncFailure(f"Could not read duration of output {output_path}")
    if src_duration is not None and abs(out_duration - src_duration) > _DURATION_TOLERANCE_SEC:
        raise LipSyncFailure(
            f"Output duration ({out_duration:.2f}s) doesn't match source audio "
            f"duration ({src_duration:.2f}s) -- likely a truncated/failed render, "
            f"not a genuine lip-synced clip."
        )

    logger.info(
        "[LipSync] OK: %s -> %s (%.1fs render, %.2fs output)",
        face_video_path, output_path, render_seconds, out_duration,
    )
    return LipSyncResult(output_path=output_path, duration_sec=out_duration, render_seconds=render_seconds)


async def run_lipsync_with_retry(
    face_video_path: str, audio_path: str, output_path: str, *, retries: int = 1,
) -> Optional[LipSyncResult]:
    """Try once, retry `retries` more times on failure. Returns None (never
    a fake success) if all attempts fail -- caller decides what to do with a
    scene that has no verified lip-sync (recommended: keep it as a non-
    talking B-roll shot for that scene rather than presenting an unsynced
    mouth as if it were synced).
    """
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await run_lipsync(face_video_path, audio_path, output_path)
        except LipSyncFailure as e:
            last_error = e
            logger.warning("[LipSync] Attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
    logger.error("[LipSync] All attempts failed for %s: %s", face_video_path, last_error)
    return None
