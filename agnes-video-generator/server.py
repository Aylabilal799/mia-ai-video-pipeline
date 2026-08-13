"""
core.api.agnes_video — Agnes Video API Hub 交互模块。

实现多 API Key 动态加载、轮询负载均衡 (Round-Robin)、自动故障转移 (Failover)
以及与 Agnes V2.0 规范对齐的 429/503 重试逻辑。
"""

import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

AGNES_API_HUB_URL = "https://apihub.agnes-ai.com/agnesapi"


def get_all_agnes_keys() -> List[str]:
    """
    收集所有已配置的 Agnes API Key，优先级顺序：
    1. AGNES_API_KEY_1, AGNES_API_KEY_2, AGNES_API_KEY_3, ...
    2. AGNES_API_KEYS (逗号分隔)
    3. AGNES_API_KEY (单个或逗号分隔)
    返回去重且非空的 Key 列表。
    """
    keys = []

    # 1. 检查索引环境变量 AGNES_API_KEY_1, AGNES_API_KEY_2 等
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

    # 2. 检查逗号分隔的 AGNES_API_KEYS
    raw_keys = os.getenv('AGNES_API_KEYS', '').strip()
    if raw_keys:
        for k in raw_keys.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # 3. 检查单 Key / 回退 AGNES_API_KEY
    raw_single = os.getenv('AGNES_API_KEY', '').strip()
    if raw_single:
        for k in raw_single.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    return keys


def _mask_key(key: str) -> str:
    """脱敏输出 API Key，切勿打印明文 Key。"""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return key[:2] + "***"
    return f"{key[:4]}***{key[-4:]}"


def _get_key_label(key: str, all_keys: Optional[List[str]] = None) -> str:
    if all_keys is None:
        all_keys = get_all_agnes_keys()
    masked = _mask_key(key)
    if key in all_keys:
        idx = all_keys.index(key) + 1
        return f"API key #{idx}/{len(all_keys)} ({masked})"
    return f"API key ({masked})"


class ServerKeyPool:
    """Agnes 服务端线程安全的 Round-Robin API Key 连接池。"""
    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0

    def get_next_key(self) -> str:
        keys = get_all_agnes_keys()
        if not keys:
            raise RuntimeError("No Agnes API key configured on Agnes server.")
        with self._lock:
            key = keys[self._index % len(keys)]
            self._index = (self._index + 1) % len(keys)
            return key


KEY_POOL = ServerKeyPool()


def submit_video_request(
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    max_retries_per_key: int = 3,
) -> Tuple[str, str]:
    """
    提交视频生成/关键帧请求到 Agnes API Hub。
    若未指定 api_key，则使用 KEY_POOL 进行 Round-Robin 选择；
    如遇到 429 (Rate Limit) 或 503 (Server Error)，自动 Failover 到下一张 Key。

    Returns:
        (video_id, bound_api_key)
    """
    keys = get_all_agnes_keys()
    if not keys:
        raise RuntimeError("No Agnes API keys configured.")

    if api_key:
        start_key = api_key
    else:
        start_key = KEY_POOL.get_next_key()

    start_idx = keys.index(start_key) if start_key in keys else 0
    ordered_keys = keys[start_idx:] + keys[:start_idx]

    last_error = None

    for candidate_key in ordered_keys:
        label = _get_key_label(candidate_key, keys)
        logger.info("[AgnesServer] Using %s for scene video submission", label)

        headers = {
            "Authorization": f"Bearer {candidate_key}",
            "X-API-Key": candidate_key,
            "Content-Type": "application/json",
        }

        req_payload = dict(payload)
        req_payload["api_key"] = candidate_key

        for attempt in range(max_retries_per_key):
            try:
                resp = requests.post(
                    AGNES_API_HUB_URL,
                    json=req_payload,
                    headers=headers,
                    timeout=45,
                )

                if resp.status_code in (200, 201, 202):
                    data = resp.json()
                    video_id = data.get("video_id") or data.get("task_id") or data.get("id")
                    if not video_id and isinstance(data.get("data"), dict):
                        video_id = data["data"].get("video_id") or data["data"].get("task_id")
                    if video_id:
                        logger.info("[AgnesServer] Successfully submitted scene video %s with %s", video_id, label)
                        return str(video_id), candidate_key
                    raise RuntimeError(f"Agnes response missing video_id: {data}")

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_sec = float(retry_after) if (retry_after and retry_after.isdigit()) else (2.0 ** attempt + random.uniform(0.5, 1.5))
                    logger.warning("[AgnesServer] 429 Rate Limit on %s (attempt %d/%d). Failing over to next key...", label, attempt + 1, max_retries_per_key)
                    time.sleep(sleep_sec)
                    break  # Failover to next key in pool

                if resp.status_code in (502, 503, 504):
                    sleep_sec = 2.0 ** attempt + random.uniform(0.5, 1.5)
                    logger.warning("[AgnesServer] %d Server Error on %s (attempt %d/%d). Failing over to next key...", resp.status_code, label, attempt + 1, max_retries_per_key)
                    time.sleep(sleep_sec)
                    break  # Failover to next key in pool

                if resp.status_code in (401, 403):
                    logger.warning("[AgnesServer] Authentication error (%d) on %s. Trying next key...", resp.status_code, label)
                    break  # Failover to next key

                resp.raise_for_status()

            except (requests.RequestException, Exception) as e:
                logger.warning("[AgnesServer] Submission error on %s: %s", label, e)
                last_error = e
                time.sleep(1.0)

    raise RuntimeError(f"All Agnes API keys failed for video submission. Last error: {last_error}")


def poll_video_status(
    video_id: str,
    api_key: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    查询单个场景视频的生成状态。
    必须使用创建该 video_id 时绑定的 api_key，切勿盲目切换 Key！
    """
    all_keys = get_all_agnes_keys()
    label = _get_key_label(api_key, all_keys)
    logger.debug("[AgnesServer] Polling video %s using %s", video_id, label)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }
    params = {
        "video_id": video_id,
        "api_key": api_key,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                AGNES_API_HUB_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_sec = float(retry_after) if (retry_after and retry_after.isdigit()) else (2.0 * (attempt + 1) + random.uniform(0.5, 1.0))
                logger.warning("[AgnesServer] 429 Rate Limit polling %s on %s. Waiting %.1fs...", video_id, label, sleep_sec)
                time.sleep(sleep_sec)
                continue

            if resp.status_code in (502, 503, 504):
                sleep_sec = 2.0 * (attempt + 1) + random.uniform(0.5, 1.0)
                logger.warning("[AgnesServer] %d Server Error polling %s on %s. Waiting %.1fs...", resp.status_code, video_id, label, sleep_sec)
                time.sleep(sleep_sec)
                continue

            if resp.status_code in (401, 403):
                raise RuntimeError(f"Agnes poll auth error ({resp.status_code}) on {label} for video {video_id}: {resp.text}")

            resp.raise_for_status()
            return resp.json()

        except (requests.RequestException, Exception) as e:
            last_error = e
            time.sleep(1.5)

    raise RuntimeError(f"Failed to poll video {video_id} on {label}: {last_error}")
