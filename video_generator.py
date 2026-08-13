import os
import time
import socket
import subprocess
import logging
import threading
import random
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Agnes will be cloned inside the project folder
AGNES_PATH = os.getenv('AGNES_PATH', os.path.join(os.path.dirname(__file__), 'agnes-video-generator'))
AGNES_HOST = os.getenv('AGNES_HOST', '127.0.0.1')
AGNES_PORT = int(os.getenv('AGNES_PORT', '8765'))
AGNES_BASE_URL = f'http://{AGNES_HOST}:{AGNES_PORT}'


def _load_agnes_api_keys() -> list:
    """
    Collects all configured Agnes API keys in priority order:
      1. AGNES_API_KEY_1, AGNES_API_KEY_2, AGNES_API_KEY_3, ...
      2. AGNES_API_KEYS (comma-separated list)
      3. AGNES_API_KEY (single key or comma-separated list)

    Returns a list of unique, non-empty stripped key strings.
    """
    keys = []

    # Check indexed keys AGNES_API_KEY_1, AGNES_API_KEY_2, etc.
    i = 1
    while True:
        k = os.getenv(f'AGNES_API_KEY_{i}', '').strip()
        if not k:
            if not any(os.getenv(f'AGNES_API_KEY_{j}') for j in range(i + 1, i + 10)):
                break
        else:
            if k not in keys:
                keys.append(k)
        i += 1

    # Check comma-separated AGNES_API_KEYS
    raw_keys = os.getenv('AGNES_API_KEYS', '').strip()
    if raw_keys:
        for k in raw_keys.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # Check single or comma-separated AGNES_API_KEY fallback
    raw_single = os.getenv('AGNES_API_KEY', '').strip()
    if raw_single:
        for k in raw_single.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    return keys


AGNES_API_KEYS = _load_agnes_api_keys()


def _mask_key(key: str) -> str:
    """Masks sensitive API key for safe logging."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return key[:2] + "***"
    return f"{key[:4]}***{key[-4:]}"


class ThreadSafeKeyPool:
    """Round-robin thread-safe key selector over configured Agnes API keys."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0

    def get_next_key(self) -> str:
        keys = _load_agnes_api_keys()
        if not keys:
            raise RuntimeError(
                "No Agnes API key configured. Set AGNES_API_KEY_1, AGNES_API_KEY_2, "
                "AGNES_API_KEYS, or AGNES_API_KEY in .env. Get free keys from "
                "https://platform.agnes-ai.com"
            )
        with self._lock:
            key = keys[self._index % len(keys)]
            self._index = (self._index + 1) % len(keys)
        return key


KEY_POOL = ThreadSafeKeyPool()

AGNES_VENV_PYTHON = Path(AGNES_PATH) / '.venv' / 'bin' / 'python'
# --- FIX: the real FastAPI app (uvicorn.run(), the actual __main__ entrypoint)
# lives in models/server.py, run as a package module (models/__init__.py exists).
# The top-level server.py is just a client helper (poll_video_status, etc.) with
# no app bootstrap at all -- running it directly does nothing and exits 0 instantly,
# which is exactly the symptom we were chasing (clean exit, zero output, no port bound).
AGNES_SERVER_MODULE = 'models.server'
AGNES_SERVER_SCRIPT = Path(AGNES_PATH) / 'models' / 'server.py'  # used only for the existence check
AGNES_LOG_FILE = Path(AGNES_PATH) / 'server.log'
AGNES_PID_FILE = Path(AGNES_PATH) / 'server.pid'

# --- FIX: bumped from 90s -> 240s. Agnes can legitimately take longer than 90s
# to bind its port on a cold start (loading Kokoro TTS / other models into memory
# for the first time since boot). 90s was too aggressive and caused false-positive
# "did not start" failures even when Agnes was simply still loading.
STARTUP_TIMEOUT = int(os.getenv('AGNES_STARTUP_TIMEOUT', '240'))  # seconds to wait for Agnes server to come up

POLL_INTERVAL = 10  # seconds between task status polls
MAX_POLL_TIME = 3600 * 2  # 2 hour ceiling
UNKNOWN_STATUS_LIMIT = 30  # consecutive "can't find a status field" polls before giving up

# Rough progress (0-100) for the stages that happen OUTSIDE Agnes's own pipeline
_PROGRESS_STARTING_SERVER = 2
_PROGRESS_SUBMITTING = 5
_PROGRESS_GENERATING_START = 8  # Agnes's own 0-100% progress is rescaled into [8, 88]
_PROGRESS_GENERATING_END = 88
_PROGRESS_DOWNLOADING = 90
_PROGRESS_ENHANCING = 95
_PROGRESS_DONE = 100


def _port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        try:
            return s.connect_ex((AGNES_HOST, AGNES_PORT)) == 0
        except OSError:
            return False


def _http_request_with_retry(method, url, key_used=None, max_retries=5, base_delay=2.0, **kwargs):
    """
    Executes HTTP request with rate limit handling (429), temporary server error handling (503),
    and exponential backoff with jitter.

    Fails immediately on 401/403 (invalid auth) to avoid endless retries.
    Never logs raw API keys.
    """
    masked = _mask_key(key_used) if key_used else "default"
    last_exc = None

    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, **kwargs)

            # 401 / 403 Authentication Error -> raise immediately
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Agnes API authentication failed ({resp.status_code}) for key {masked}: {resp.text}"
                )

            # 429 Rate Limit
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = float(retry_after)
                else:
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Agnes API 429 Rate Limit (%s). Waiting %.1fs before retry (%d/%d)...",
                    masked, sleep_time, attempt + 1, max_retries
                )
                time.sleep(sleep_time)
                continue

            # 502 / 503 / 504 Temporary Server Errors
            if resp.status_code in (502, 503, 504):
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Agnes API %d Temporary Error (%s). Waiting %.1fs before retry (%d/%d)...",
                    resp.status_code, masked, sleep_time, attempt + 1, max_retries
                )
                time.sleep(sleep_time)
                continue

            return resp

        except (requests.RequestException, socket.error) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(
                "Agnes API network error (%s): %s. Retrying in %.1fs (%d/%d)...",
                masked, e, sleep_time, attempt + 1, max_retries
            )
            time.sleep(sleep_time)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Agnes API request failed after {max_retries} retries for key {masked}")


def ensure_agnes_server_running(api_key):
    """
    Makes sure the Agnes Video Generator FastAPI server is running in the
    background at AGNES_BASE_URL. Starts it HEADLESSLY (no browser window)
    if it isn't already up, and waits until it responds.
    """
    if _port_open():
        return

    if not AGNES_SERVER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Agnes models/server.py not found at {AGNES_SERVER_SCRIPT}. "
            "Run setup.sh first to clone and install it."
        )
    if not AGNES_VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"Agnes venv python not found at {AGNES_VENV_PYTHON}. "
            "Run setup.sh first to create it."
        )

    logger.info("Agnes server not running, starting it headlessly on port %s...", AGNES_PORT)

    env = os.environ.copy()
    env['AGNES_API_KEY'] = api_key
    # --- FIX: force unbuffered stdout/stderr. Without this, Python block-buffers
    # output when it's redirected to a file (not a TTY), so uvicorn's startup
    # banner, model-loading progress, and even crash tracebacks can sit in memory
    # and never reach server.log if the process is killed/timed out first.
    env['PYTHONUNBUFFERED'] = '1'
    # --- FIX: models/server.py reads HOST/PORT (not AGNES_HOST/AGNES_PORT) to
    # decide what to bind to -- pass them explicitly so it stays in sync with
    # whatever AGNES_HOST/AGNES_PORT are configured to in this project's .env.
    env['HOST'] = AGNES_HOST
    env['PORT'] = str(AGNES_PORT)
    # --- FIX: models/server.py does `from core.audio.voices import ...` etc,
    # expecting the project root (agnes-video-generator/) to be on sys.path so
    # `core` resolves as a top-level package. Running via `-m models.server`
    # doesn't reliably put that root on sys.path in every environment, which
    # was causing "ModuleNotFoundError: No module named 'core'". Setting
    # PYTHONPATH explicitly guarantees it, regardless of -m's cwd quirks.
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        str(AGNES_PATH) if not existing_pythonpath
        else f"{AGNES_PATH}{os.pathsep}{existing_pythonpath}"
    )
    env['PROMPT_LANGUAGE'] = os.getenv('PROMPT_LANGUAGE', 'en')
    if os.getenv('OPENROUTER_API_KEY'):
        env['OPENROUTER_API_KEY'] = os.environ['OPENROUTER_API_KEY']
        env['OPENROUTER_MODEL'] = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-sonnet-4.5')

    with open(AGNES_LOG_FILE, 'a') as log:
        proc = subprocess.Popen(
            # --- FIX: run as a module (`-m models.server`) from the project root,
            # not `python server.py` (the top-level server.py is just a client
            # helper with no uvicorn.run() / __main__ block -- it was exiting
            # cleanly and instantly with zero output, which is what we were chasing).
            [str(AGNES_VENV_PYTHON), '-u', '-m', AGNES_SERVER_MODULE],
            cwd=str(AGNES_PATH),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach process
        )
    AGNES_PID_FILE.write_text(str(proc.pid))

    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if _port_open():
            time.sleep(1)
            return
        # --- FIX: if the subprocess died on its own (crash) during startup,
        # fail fast instead of waiting out the full timeout for no reason.
        if proc.poll() is not None:
            raise RuntimeError(
                f"Agnes server process exited early (code {proc.returncode}) while starting up. "
                f"Check the log at {AGNES_LOG_FILE} for the actual error."
            )
        time.sleep(1)

    raise RuntimeError(
        f"Agnes server did not start within {STARTUP_TIMEOUT}s. "
        f"Check the log at {AGNES_LOG_FILE} for errors."
    )


def _push_api_key(api_key):
    """Pushes the given key to Agnes's live /api/config endpoint."""
    try:
        requests.post(
            f"{AGNES_BASE_URL}/api/config",
            json={"api_key": api_key},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning("Could not push API key via /api/config: %s", e)


def submit_manuscript_task(script, api_key):
    """
    Submits the script to Agnes's Manuscript Video pipeline using the specified api_key.
    Returns the Agnes task id.
    """
    payload = {
        "manuscript_text": script,
        "video_width": 720,
        "video_height": 1280,
        "audio_voice": "am_michael",
        "audio_lang": "en",
        "audio_rate": "+0%",
        "subtitle_enabled": True,
        "subtitle_style_mode": "fixed",
        "subtitle_font": "ArchivoBlack-Regular.ttf",
        "subtitle_fontsize": 46,
        "subtitle_position": "bottom",
        "subtitle_color": "white",
        "subtitle_stroke_color": "black",
        "subtitle_stroke_width": 3,
        "subtitle_bg_color": "none",
        "api_key": api_key,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }

    _push_api_key(api_key)

    resp = _http_request_with_retry(
        "POST",
        f"{AGNES_BASE_URL}/api/tasks/manuscript",
        key_used=api_key,
        data=payload,
        headers=headers,
        timeout=30,
    )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Agnes rejected the manuscript task ({resp.status_code}) with key {_mask_key(api_key)}: {resp.text}"
        )

    data = resp.json()
    task_id = data.get('task_id') or data.get('id') or (data.get('task') or {}).get('id')
    if not task_id:
        raise RuntimeError(f"Agnes response didn't include a task id: {data}")
    return task_id


def _extract_status(data):
    task = data.get('task', data)
    status = task.get('status') or task.get('state')
    progress = task.get('current_progress')
    if progress is None:
        progress = task.get('progress')
    return status, progress


_GENERATION_STAGE_LABELS = [
    (10, "Writing scene descriptions..."),
    (25, "Submitting scenes for video generation..."),
    (75, "Generating video clips (this is the slow part)..."),
    (85, "Recording narration..."),
    (92, "Generating captions..."),
    (100, "Stitching final video together..."),
]


def _describe_generation_stage(agnes_progress):
    if not isinstance(agnes_progress, (int, float)):
        return "Generating video (this can take a while)..."
    for threshold, label in _GENERATION_STAGE_LABELS:
        if agnes_progress <= threshold:
            return label
    return "Finishing up..."


def wait_for_task(task_id, task=None, api_key=None):
    """
    Polls GET /api/tasks/{id} until the Agnes task finishes or fails.
    Preserves the api_key associated with task_id for all status polling.
    Rescales Agnes's progress into the [_PROGRESS_GENERATING_START, _PROGRESS_GENERATING_END] slice.
    """
    deadline = time.time() + MAX_POLL_TIME
    last_message = None
    last_progress = None
    unknown_status_count = 0
    last_raw = None

    headers = {}
    params = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
        params["api_key"] = api_key

    _push_api_key(api_key)

    while time.time() < deadline:
        resp = _http_request_with_retry(
            "GET",
            f"{AGNES_BASE_URL}/api/tasks/{task_id}",
            key_used=api_key,
            headers=headers,
            params=params,
            timeout=30,
        )

        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"Agnes authentication failed polling task {task_id} with key {_mask_key(api_key)}: {resp.text}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Failed to poll Agnes task {task_id} ({resp.status_code}): {resp.text}")

        data = resp.json()
        last_raw = data
        status, agnes_progress = _extract_status(data)
        message = _describe_generation_stage(agnes_progress)

        overall_progress = None
        if isinstance(agnes_progress, (int, float)):
            span = _PROGRESS_GENERATING_END - _PROGRESS_GENERATING_START
            overall_progress = round(_PROGRESS_GENERATING_START + (agnes_progress / 100.0) * span)

        if task and (message != last_message or overall_progress != last_progress):
            meta = {'stage': message}
            if overall_progress is not None:
                meta['progress'] = overall_progress
            task.update_state(state='PROGRESS', meta=meta)
            last_message = message
            last_progress = overall_progress

        if status in ('completed', 'success', 'done', 'finished'):
            return data
        if status in ('failed', 'error'):
            raise RuntimeError(f"Agnes video generation failed for task {task_id}: {data}")

        if status is None:
            unknown_status_count += 1
            if unknown_status_count >= UNKNOWN_STATUS_LIMIT:
                raise RuntimeError(
                    "Could not find a recognizable status field in Agnes's task "
                    f"response after {UNKNOWN_STATUS_LIMIT} polls. Last raw response: {last_raw}. "
                    "The API schema may have changed -- check agnes-video-generator/server.log."
                )
        else:
            unknown_status_count = 0

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Agnes task {task_id} did not finish within {MAX_POLL_TIME}s")


def download_video(task_id, output_path, api_key=None):
    headers = {}
    params = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
        params["api_key"] = api_key

    resp = _http_request_with_retry(
        "GET",
        f"{AGNES_BASE_URL}/api/video/{task_id}",
        key_used=api_key,
        headers=headers,
        params=params,
        stream=True,
        timeout=60,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to download finished video for task {task_id} ({resp.status_code}): {resp.text}")

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return output_path


def postprocess_for_shorts(input_path, output_path):
    """
    Upscales Agnes's 720x1280 output to true 1080x1920 (YouTube Shorts standard),
    applies a light sharpen pass, and re-encodes with a clear bitrate.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', 'scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:5:5:0.4',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg post-processing failed:\n{result.stdout}")
    return output_path


def _generate_with_key(script, work_dir, api_key, key_label, task=None):
    """Runs the full pipeline against ONE specific Agnes API key."""
    if task:
        task.update_state(state='PROGRESS', meta={
            'stage': f'Starting Agnes AI server ({key_label})...',
            'progress': _PROGRESS_STARTING_SERVER,
        })
    ensure_agnes_server_running(api_key)
    _push_api_key(api_key)

    if task:
        task.update_state(state='PROGRESS', meta={
            'stage': f'Submitting script to Agnes AI ({key_label})...',
            'progress': _PROGRESS_SUBMITTING,
        })
    task_id = submit_manuscript_task(script, api_key=api_key)

    if task:
        task.update_state(state='PROGRESS', meta={
            'stage': 'Generating video (this can take a while)...',
            'progress': _PROGRESS_GENERATING_START,
        })
    wait_for_task(task_id, task=task, api_key=api_key)

    if task:
        task.update_state(state='PROGRESS', meta={
            'stage': 'Downloading finished video...',
            'progress': _PROGRESS_DOWNLOADING,
        })
    raw_path = Path(work_dir) / 'raw_video.mp4'
    download_video(task_id, raw_path, api_key=api_key)

    if task:
        task.update_state(state='PROGRESS', meta={
            'stage': 'Enhancing video quality for Shorts...',
            'progress': _PROGRESS_ENHANCING,
        })
    final_path = Path(work_dir) / 'final_video.mp4'
    postprocess_for_shorts(raw_path, final_path)

    if task:
        task.update_state(state='PROGRESS', meta={'stage': 'Done!', 'progress': _PROGRESS_DONE})

    return str(final_path)


def generate_video_from_script(script, work_dir, task=None):
    """
    High-level function that selects an Agnes API key via round-robin load distribution
    and attempts execution. If a key fails, failover tries remaining keys in order.
    """
    keys = _load_agnes_api_keys()
    if not keys:
        raise RuntimeError(
            "No Agnes API key configured. Set AGNES_API_KEY_1, AGNES_API_KEY_2, "
            "AGNES_API_KEYS, or AGNES_API_KEY in .env. Get free keys from "
            "https://platform.agnes-ai.com"
        )

    # Load balance across available keys using round-robin selection
    start_key = KEY_POOL.get_next_key()
    start_idx = keys.index(start_key) if start_key in keys else 0
    ordered_keys = keys[start_idx:] + keys[:start_idx]

    last_error = None
    for i, api_key in enumerate(ordered_keys):
        masked = _mask_key(api_key)
        key_label = f"key {i + 1}/{len(ordered_keys)} ({masked})"
        try:
            return _generate_with_key(script, work_dir, api_key, key_label, task)
        except Exception as e:
            logger.warning("Generation failed with %s: %s", key_label, e)
            last_error = e
            if i < len(ordered_keys) - 1:
                if task:
                    task.update_state(state='PROGRESS', meta={
                        'stage': f'{key_label} failed, retrying with next key...',
                        'progress': _PROGRESS_STARTING_SERVER,
                    })
                try:
                    if AGNES_PID_FILE.exists():
                        pid = int(AGNES_PID_FILE.read_text().strip())
                        os.kill(pid, 9)
                        AGNES_PID_FILE.unlink()
                except (OSError, ValueError):
                    pass
                time.sleep(3)
                continue
            raise RuntimeError(f"All {len(ordered_keys)} Agnes API key(s) failed. Last error: {last_error}")
