"""core.audio.tts — TTS 统一接口：KokoroTTSEngine + SilentTTSEngine

Edge-TTS 已被完全移除。项目现在只有一个 TTS 实现：Kokoro（本地、离线、
Apache-2.0 许可）。Mia 的固定音色是 af_heart（美式英语女声）。

设计要点（对应用户的硬性要求）：
  - 没有任何 Edge-TTS 代码、没有 en-US-AriaNeural、没有自动降级到别的引擎。
  - 如果 Kokoro 生成失败，直接抛出 RuntimeError，让上层任务清楚地失败并
    报告真实错误 —— 不再像旧代码那样静默降级为无声占位音频。
  - 字幕/卡拉OK 高亮所需的逐词时间戳优先使用 Kokoro 自带的 token 级时间戳
    （misaki G2P，在实际调用模型合成时会为每个 token 填充 start_ts/end_ts）。
    只有在 Kokoro 确实没有返回任何时间戳时，才回退到本地强制对齐
    （faster-whisper word-level alignment，跑在刚生成的真实音频上），
    而不是伪造均匀分布的假时间戳。
"""

import asyncio
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Cue types — deliberately duck-type-compatible with the old edge_tts
# SubMaker/Subtitle interface (`.cues`, each cue has `.start`, `.end`
# (timedelta) and `.content` (str)) so every downstream consumer
# (SubtitleGenerator, karaoke renderer) keeps working unmodified.
# ---------------------------------------------------------------------------

@dataclass
class _Cue:
    start: timedelta
    end: timedelta
    content: str


@dataclass
class _SubMakerLike:
    cues: List[_Cue] = field(default_factory=list)

    def get_srt(self) -> str:  # pragma: no cover - legacy-interface shim only
        raise AttributeError("get_srt is not supported by KokoroTTSEngine cues")


def _log_cue_diagnostics(sub_maker: "_SubMakerLike", text: str, source: str) -> None:
    cues = sub_maker.cues if sub_maker else []
    cue_count = len(cues)
    logger.info(f"[TTS] Word cues: {cue_count} (source={source})")

    if cue_count == 0:
        logger.warning("[TTS] No word-level cues available -- captions will "
                        "fall back to legacy (non-cue-aware) timing.")
        return

    expected_words = max(len(text.split()), 1)
    avg_words_per_cue = sum(len(c.content.split()) or 1 for c in cues) / cue_count
    if avg_words_per_cue > 2.5 or cue_count < expected_words * 0.4:
        logger.warning(
            f"[TTS] WARNING: cues look coarser than word-level "
            f"({cue_count} cues for ~{expected_words} words, "
            f"avg {avg_words_per_cue:.1f} words/cue)."
        )
    else:
        logger.info(
            f"[TTS] Cue granularity looks word-level "
            f"({cue_count} cues for ~{expected_words} words, "
            f"avg {avg_words_per_cue:.1f} words/cue)."
        )


class TTSEngine(ABC):
    """TTS 抽象基类。"""

    @abstractmethod
    async def generate(
        self, text: str, output_path: str, voice: str = "af_heart", rate: str = "+0%"
    ) -> Tuple[str, object]:
        """生成音频文件，返回 (audio_path, sub_maker_or_cues)。"""
        ...


class KokoroTTSEngine(TTSEngine):
    """本地 Kokoro TTS 引擎（唯一的语音合成实现）。

    generate() 返回 (audio_path, sub_maker)，sub_maker.cues 是逐词时间戳列表，
    结构与旧 edge_tts.SubMaker.cues 兼容，供 SubtitleGenerator / karaoke 直接使用。

    Kokoro 是同步 + CPU/GPU 均可运行的模型，这里用 run_in_executor 丢到线程池，
    避免阻塞事件循环。pipeline 实例按 lang_code 懒加载并缓存为类变量，
    这样同一进程内多次调用不用每次重新加载模型权重。
    """

    _pipelines: dict = {}

    # Kokoro voice -> misaki lang_code (only English is wired up for Mia;
    # 'a' = American English, which is what af_* voices need).
    _VOICE_LANG_CODE = {
        # American English
        **{v: "a" for v in (
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
        )},
        # British English
        **{v: "b" for v in (
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        )},
    }

    @classmethod
    def _lang_code_for_voice(cls, voice: str) -> str:
        return cls._VOICE_LANG_CODE.get(voice, "a")

    @classmethod
    def _get_pipeline(cls, lang_code: str):
        pipeline = cls._pipelines.get(lang_code)
        if pipeline is None:
            logger.info(f"[TTS] Loading Kokoro pipeline (lang_code={lang_code!r})...")
            from kokoro import KPipeline  # imported lazily so the whole app
            pipeline = KPipeline(lang_code=lang_code)  # doesn't hard-fail to import
            cls._pipelines[lang_code] = pipeline
            logger.info("[TTS] Kokoro pipeline loaded.")
        return pipeline

    async def generate(
        self, text: str, output_path: str, voice: str = "af_heart", rate: str = "+0%"
    ) -> Tuple[str, "_SubMakerLike"]:
        """生成 Kokoro TTS 音频 + 逐词时间戳。

        Raises:
            RuntimeError: Kokoro 生成失败时抛出（不降级到其它引擎/假时间戳，
            按用户要求"fail clearly and report the actual error"）。
        """
        logger.info("[TTS] Engine: Kokoro")
        logger.info(f"[TTS] Voice: {voice}")
        logger.info("[TTS] Generating audio...")

        try:
            audio_path, sub_maker = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, text, output_path, voice, rate,
            )
        except Exception as e:
            logger.error(f"[TTS] Kokoro generation failed: {e}")
            raise RuntimeError(f"Kokoro TTS generation failed: {e}") from e

        logger.info("[TTS] Audio generated successfully")
        return audio_path, sub_maker

    def _generate_sync(
        self, text: str, output_path: str, voice: str, rate: str
    ) -> Tuple[str, "_SubMakerLike"]:
        import numpy as np
        import soundfile as sf

        speed = _rate_string_to_speed(rate)
        lang_code = self._lang_code_for_voice(voice)
        pipeline = self._get_pipeline(lang_code)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        audio_chunks: List["np.ndarray"] = []
        word_cues: List[_Cue] = []
        t_offset = 0.0

        for result in pipeline(text, voice=voice, speed=speed):
            audio = result.audio
            audio_np = audio.detach().cpu().numpy() if hasattr(audio, "detach") else (
                audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            )
            if audio_np.size == 0:
                continue
            chunk_dur = len(audio_np) / KOKORO_SAMPLE_RATE

            for tok in (result.tokens or []):
                start_ts = getattr(tok, "start_ts", None)
                end_ts = getattr(tok, "end_ts", None)
                word_text = (getattr(tok, "text", "") or "").strip()
                if start_ts is None or end_ts is None or not word_text:
                    continue
                word_cues.append(_Cue(
                    start=timedelta(seconds=t_offset + float(start_ts)),
                    end=timedelta(seconds=t_offset + float(end_ts)),
                    content=word_text,
                ))

            audio_chunks.append(audio_np)
            t_offset += chunk_dur

        if not audio_chunks:
            raise RuntimeError("Kokoro produced no audio output for this text")

        full_audio = np.concatenate(audio_chunks)
        full_audio = np.clip(full_audio, -1.0, 1.0)

        tmp_wav = output_path + ".kokoro_tmp.wav"
        try:
            sf.write(tmp_wav, full_audio, KOKORO_SAMPLE_RATE, subtype="PCM_16")
            _verify_and_log_wav(tmp_wav)
            _encode_to_output(tmp_wav, output_path)
        finally:
            if os.path.exists(tmp_wav) and tmp_wav != output_path:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

        used_native = bool(word_cues)
        if not word_cues:
            logger.warning(
                "[TTS] Kokoro returned no built-in token timestamps for this "
                "text -- attempting local forced alignment instead of fake "
                "evenly-spaced timings."
            )
            word_cues = _forced_align_word_cues(output_path, text)

        sub_maker = _SubMakerLike(cues=word_cues)
        _log_cue_diagnostics(
            sub_maker, text, source="kokoro-native" if used_native else "forced-alignment",
        )
        return output_path, sub_maker


def _rate_string_to_speed(rate: str) -> float:
    """把旧的 edge_tts 风格 rate 字符串（如 "+10%", "-5%"）转成 Kokoro 的
    speed 倍率（1.0 = 正常速度）。"""
    if not rate:
        return 1.0
    r = rate.strip()
    try:
        if r.endswith("%"):
            pct = float(r.rstrip("%"))
            return max(0.5, min(2.0, 1.0 + pct / 100.0))
        return float(r)
    except ValueError:
        return 1.0


def _verify_and_log_wav(wav_path: str) -> dict:
    """Verifies that the generated Kokoro WAV file exists, is non-empty,
    and readable by soundfile, and logs its detailed properties.
    """
    import soundfile as sf

    if not os.path.exists(wav_path):
        raise RuntimeError(f"Generated WAV file does not exist: {wav_path}")

    file_size = os.path.getsize(wav_path)
    if file_size == 0:
        raise RuntimeError(f"Generated WAV file is empty (0 bytes): {wav_path}")

    try:
        info = sf.info(wav_path)
    except Exception as e:
        raise RuntimeError(f"Generated WAV file is invalid or unreadable by soundfile: {e}") from e

    if info.frames == 0 or info.duration == 0:
        raise RuntimeError(f"Generated WAV file has zero frames or duration: {wav_path}")

    logger.info(f"[TTS] Kokoro WAV: {wav_path}")
    logger.info(f"[TTS] sample_rate: {info.samplerate}")
    logger.info(f"[TTS] channels: {info.channels}")
    logger.info(f"[TTS] samples: {info.frames}")
    logger.info(f"[TTS] duration: {info.duration:.3f}s")
    logger.info(f"[TTS] format: {info.format_info} ({info.subtype_info})")
    logger.info(f"[TTS] file_size: {file_size} bytes")

    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "samples": info.frames,
        "duration": info.duration,
        "file_size": file_size,
    }


def _encode_to_output(wav_path: str, output_path: str) -> None:
    """Encodes Kokoro WAV output to target output_path format.

    If output_path ends in .wav, copies directly.
    Otherwise (e.g. .mp3), encodes via FFmpeg with explicit format (-f mp3),
    atomic file replacement, and complete stderr logging.
    """
    if output_path.lower().endswith(".wav"):
        shutil.copyfile(wav_path, output_path)
        return

    import subprocess

    tmp_out = output_path + ".tmp.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-f", "mp3",
        "-c:a", "libmp3lame",
        "-q:a", "2",
        "-ar", "24000",
        "-ac", "1",
        tmp_out,
    ]

    logger.info(f"[TTS] FFmpeg command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr_text = proc.stderr.decode(errors="replace") if proc.stderr else ""

        if proc.returncode != 0:
            logger.error(f"[TTS] FFmpeg failed with exit code {proc.returncode}")
            logger.error(f"[TTS] FFmpeg stderr:\n{stderr_text}")

            stderr_lines = stderr_text.strip().splitlines()
            last_lines = "\n".join(stderr_lines[-15:]) if stderr_lines else stderr_text

            raise RuntimeError(
                f"ffmpeg failed to encode Kokoro output (code {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"FFmpeg stderr tail:\n{last_lines}"
            )

        if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
            raise RuntimeError(
                f"ffmpeg reported code 0 but output file {tmp_out} is missing or empty."
            )

        os.replace(tmp_out, output_path)
        logger.info(f"[TTS] MP3 successfully encoded -> {output_path}")

    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass


def _forced_align_word_cues(audio_path: str, text: str) -> List[_Cue]:
    """本地强制对齐兜底路径：只有当 Kokoro 自己没有返回 token 时间戳时才会
    走到这里。使用 faster-whisper 的 word_timestamps 对刚生成的真实 Kokoro
    音频做转录级对齐，取真实时间戳，而不是猜测/均分。

    faster-whisper 未安装，或对齐本身失败时，直接抛出异常而不是编造时间轴
    —— 与用户的要求一致："Do NOT create fake evenly-spaced word timings."
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "Kokoro did not provide word timestamps and faster-whisper "
            "(needed for local forced alignment) is not installed. Install "
            "it with `pip install faster-whisper` to enable the fallback "
            "alignment path. Refusing to fabricate evenly-spaced timings."
        ) from e

    model = _get_whisper_align_model(WhisperModel)
    segments, _info = model.transcribe(audio_path, word_timestamps=True, language="en")

    cues: List[_Cue] = []
    for seg in segments:
        for w in (seg.words or []):
            word_text = (w.word or "").strip()
            if not word_text:
                continue
            cues.append(_Cue(
                start=timedelta(seconds=float(w.start)),
                end=timedelta(seconds=float(w.end)),
                content=word_text,
            ))

    if not cues:
        raise RuntimeError(
            "Local forced alignment (faster-whisper) produced no word "
            "timestamps for the generated Kokoro audio."
        )
    return cues


_whisper_align_model = None


def _get_whisper_align_model(whisper_model_cls):
    global _whisper_align_model
    if _whisper_align_model is None:
        logger.info("[TTS] Loading faster-whisper alignment model (base.en, CPU int8)...")
        _whisper_align_model = whisper_model_cls("base.en", device="cpu", compute_type="int8")
    return _whisper_align_model


class SilentTTSEngine(TTSEngine):
    """静音占位 TTS 引擎。

    生成指定时长的静音音频，返回空 cues。用于用户关闭旁白时仍需要字幕时间轴，
    或某个场景本来就没有配音文本的场景 —— 这是一个独立于「TTS 引擎选择」的
    功能，不是 Kokoro 失败时的自动降级路径。
    """

    async def generate(
        self,
        text: str,
        output_path: str,
        voice: str = "af_heart",
        rate: str = "+0%",
        duration_sec: Optional[float] = None,
    ) -> Tuple[str, dict]:
        if duration_sec is None:
            # 估算时长：英文按约 2.5 词/秒估算
            duration_sec = max(len(text.split()) / 2.5, 1.0) if text else 1.0

        logger.info(f"[TTS] Generating silent audio: {duration_sec:.1f}s → {output_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(duration_sec),
            "-c:a", "libmp3lame",
            "-q:a", "4",
            output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[:500] if stderr else ""
            raise RuntimeError(
                f"[TTS] ffmpeg silent generation failed (code {proc.returncode}): {err_msg}"
            )

        return output_path, None
