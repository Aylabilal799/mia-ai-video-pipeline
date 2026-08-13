"""core.pipelines.manuscript_video -- Manuscript Video Generation Pipeline
"""
import asyncio
import json
import logging
import math
import os
import re
from typing import Callable, List, Optional

from core.api.agnes_video import AgnesVideoAPI
from core.api.agnes_image import AgnesImageAPI
from core.compositor.concatenator import VideoConcatenator
from core.screenwriter import Screenwriter
from core.pipelines import MultiScenePipeline, PipelineShutdown
from core.audio.voices import detect_text_script
from core.audio.subtitle import SubtitleGenerator
from models.task import (
    ManuscriptVideoTask,
    ManuscriptParagraph,
    SceneTask,
    StepStatus,
    AudioConfig,
    SubtitleConfig,
)

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")

_CHARS_PER_SEC_BY_SCRIPT = {
    "zh": 4.0,
    "ja": 4.0,
    "ko": 3.5,
    "ru": 12.0,
    "latin": 15.0,  # ~150 wpm average English speech
    "unknown": 8.0,
}


def _chars_per_sec(text: str) -> float:
    script = detect_text_script(text) if text else "unknown"
    return _CHARS_PER_SEC_BY_SCRIPT.get(script, _CHARS_PER_SEC_BY_SCRIPT["unknown"])


# Greedy-merge segment duration thresholds adjusted to yield ~6-7 scenes for a 45-60s vlog
_MAX_SEGMENT_DURATION = 8.5
_MIN_SEGMENT_DURATION = 4.5

_SUBMIT_RETRY_INTERVAL_BASE_SECONDS = 15
_WAIT_RETRY_INTERVAL_BASE_SECONDS = 20

_PROGRESS_SCENE_PROMPTS_START = 0.05
_PROGRESS_SCENE_PROMPTS_SPAN = 0.10
_PROGRESS_SUBMIT_START = 0.15
_PROGRESS_SUBMIT_SPAN = 0.20
_PROGRESS_WAIT_START = 0.35
_PROGRESS_WAIT_SPAN = 0.25
_PROGRESS_AUDIO_START = 0.60
_PROGRESS_SUBTITLE_START = 0.75
_PROGRESS_CONCAT_START = 0.80


class ManuscriptVideoPipeline(MultiScenePipeline):

    def __init__(
        self,
        api_key: str,
        task_id: str,
        dir_name: str = None,
        chat_model: str = "agnes-2.0-flash",
        image_model: str = "agnes-image-2.1-flash",
        video_model: str = "agnes-video-v2.0",
        progress_callback: Optional[Callable] = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(api_key, task_id, dir_name, progress_callback, shutdown_event)
        self.video_api = AgnesVideoAPI(api_key=api_key, model=video_model)
        self.video_api.shutdown_event = shutdown_event
        self.screenwriter = Screenwriter(api_key=api_key, model=chat_model)
        self.image_generator = AgnesImageAPI(api_key=api_key, model=image_model)

    def _get_watermark_language_text(self) -> str:
        return self._state.manuscript_text

    async def _build_scenes(self) -> None:
        if not self._state.paragraphs:
            paragraphs = self._split_text(self._state.manuscript_text)
            self._state.paragraphs = paragraphs
            self.task_manager.update_state(paragraphs=paragraphs)
        else:
            logger.info("[Manuscript] Reusing %d existing paragraphs", len(self._state.paragraphs))

        await self._generate_scene_prompts(self._state.paragraphs)

        self._state.scenes = [
            SceneTask(
                index=p.index,
                scene_prompt=p.scene_prompt,
                narration_text=p.text,
                duration=max(int(math.ceil(len(p.text) / _chars_per_sec(p.text))), 3),
            )
            for p in self._state.paragraphs
        ]
        self.task_manager.update_state(scenes=[s.model_dump() for s in self._state.scenes])

    async def _build_reference_images(self) -> None:
        return

    def _get_identity_reference_paths(self) -> List[str]:
        ref = self._state.reference_image
        if ref and os.path.exists(ref):
            return [ref]
        return []

    async def _get_scene_reference_images(self, para: ManuscriptParagraph, para_dir: str) -> List[str]:
        anchor_refs = self._get_identity_reference_paths()
        if not anchor_refs:
            return []
        anchor = anchor_refs[0]

        keyframe_path = os.path.join(para_dir, "identity_keyframe.png")
        if os.path.exists(keyframe_path):
            return [anchor, keyframe_path]

        keep_identity = (
            "[PRESERVE -- keep exactly] Same person as the reference image: "
            "same face, same hair, same body type, same identity. Do NOT alter identity.\n\n"
            "[CHANGE -- mini-vlog scene action] " + (para.scene_prompt or "")
        )
        os.makedirs(para_dir, exist_ok=True)
        for attempt in range(3):
            self._check_shutdown()
            try:
                img_output = await self.image_generator.generate_single_image(
                    prompt=keep_identity,
                    reference_image_paths=[anchor],
                    size=f"{self._state.video_width}x{self._state.video_height}",
                )
                img_output.save(keyframe_path)
                return [anchor, keyframe_path]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep((attempt + 1) * 15)
                else:
                    logger.error(f"[MIA] i2i keyframe failed for scene {para.index}: {e}")

        return [anchor]

    def _split_text(self, text: str) -> List[ManuscriptParagraph]:
        text = self.fix_double_utf8(text)
        if text != self._state.manuscript_text:
            self._state.manuscript_text = text
            self.task_manager.update_state(manuscript_text=text)

        if self._state.paragraphs:
            return self._state.paragraphs

        raw_blocks = [b.strip() for b in text.split("\n") if b.strip()]
        candidate_sentences: List[str] = []
        for block in raw_blocks:
            parts = _SENTENCE_END_RE.split(block)
            for part in parts:
                part = part.strip()
                if part:
                    candidate_sentences.append(part)

        if not candidate_sentences:
            return []

        merged: List[str] = []
        current_text = ""
        current_duration = 0.0

        for sentence in candidate_sentences:
            sentence_duration = len(sentence) / _chars_per_sec(sentence)
            if not current_text:
                current_text = sentence
                current_duration = sentence_duration
                continue

            prospective_duration = current_duration + sentence_duration
            if prospective_duration <= _MAX_SEGMENT_DURATION:
                current_text += " " + sentence
                current_duration = prospective_duration
            else:
                merged.append(current_text)
                current_text = sentence
                current_duration = sentence_duration

        if current_text:
            merged.append(current_text)

        final_texts: List[str] = []
        for segment in merged:
            seg_duration = len(segment) / _chars_per_sec(segment)
            if seg_duration < _MIN_SEGMENT_DURATION and final_texts:
                final_texts[-1] += " " + segment
            else:
                final_texts.append(segment)

        paragraphs = [ManuscriptParagraph(index=idx, text=t.strip()) for idx, t in enumerate(final_texts)]
        logger.info("[Manuscript] Text split into %d scenes (target ~6-7)", len(paragraphs))
        return paragraphs

    async def _generate_scene_prompts(self, paragraphs: List[ManuscriptParagraph]) -> None:
        total = len(paragraphs)
        for i, para in enumerate(paragraphs):
            self._check_shutdown()

            if para.scene_prompt:
                continue

            await self._emit(
                "scene_prompts", "running",
                f"Generating vlog scene visual {i + 1}/{total}",
                _PROGRESS_SCENE_PROMPTS_START + _PROGRESS_SCENE_PROMPTS_SPAN * (i / max(total, 1)),
            )

            # Alternate between two shot types instead of banning talking-head
            # shots outright: roughly every other scene (and always at least
            # one scene) is a direct-to-camera speaking shot with her face and
            # mouth clearly visible, so the narration reads as HER telling the
            # story rather than an unrelated voiceover over pure B-roll. The
            # remaining scenes stay B-roll/action shots for visual variety.
            is_talking_scene = (i % 2 == 1) or (total == 1)

            if is_talking_scene:
                prompt_instructions = (
                    f"Narration line: \"{para.text}\"\n"
                    f"Scene {i+1} of {total} in a mini-vlog video.\n"
                    "Create a clear visual prompt featuring Mia in a realistic 9:16 vlog shot.\n"
                    "- Mia is talking DIRECTLY TO CAMERA, as if speaking this exact line to her "
                    "audience. Her full face must be clearly visible and unobstructed (no hand, "
                    "hair, or object covering her mouth), facing or nearly facing the lens, in a "
                    "medium close-up or chest-up framing.\n"
                    "- Natural, animated speaking expression and subtle head/hand gestures while "
                    "she talks -- not a static, frozen pose.\n"
                    "- Handheld selfie-vlog camera style, slight natural handheld motion."
                )
            else:
                prompt_instructions = (
                    f"Narration line: \"{para.text}\"\n"
                    f"Scene {i+1} of {total} in a mini-vlog video.\n"
                    "Create a clear visual prompt featuring Mia in a realistic 9:16 vlog shot.\n"
                    "- Show clear visual action (e.g. walking down street, entering cafe, holding coffee, looking around, unboxing object).\n"
                    "- Handheld vlog camera style, medium or three-quarter shot, smooth natural movement.\n"
                    "- One clear main action per scene."
                )

            prompt = await asyncio.to_thread(
                self.screenwriter.generate_scene_prompt_for_paragraph,
                prompt_instructions,
                self._state.character_style or "",
            )
            para.scene_prompt = prompt.strip()
            self.task_manager.update_state(paragraphs=paragraphs)

        self.save_prompts({
            "scene_prompts": [
                {"index": p.index, "text": p.text, "scene_prompt": p.scene_prompt}
                for p in paragraphs
            ],
        })

    async def _generate_videos(self) -> None:
        _SUBMIT_RETRIES = 3
        _WAIT_RETRIES = 3
        paragraphs = self._state.paragraphs
        total = len(paragraphs)
        pending: list[tuple[int, str, str]] = []

        for i, para in enumerate(paragraphs):
            self._check_shutdown()
            para_dir = os.path.join(self.working_dir, f"para_{para.index}")
            video_path = os.path.join(para_dir, "video.mp4")

            if os.path.exists(video_path):
                para.video_file = video_path
                continue

            if not para.scene_prompt:
                continue

            os.makedirs(para_dir, exist_ok=True)
            saved_video_id = self._load_task_json(para_dir)
            if saved_video_id:
                para.video_id = saved_video_id
                pending.append((para.index, saved_video_id, video_path))
                continue

            await self._emit(
                "video_gen", "running",
                f"Submitting scene video {i + 1}/{total}",
                _PROGRESS_SUBMIT_START + _PROGRESS_SUBMIT_SPAN * (i / max(total, 1)),
            )

            para_duration = max(int(math.ceil(len(para.text) / _chars_per_sec(para.text))), 3)
            scene_refs = await self._get_scene_reference_images(para, para_dir)

            for retry in range(_SUBMIT_RETRIES):
                try:
                    video_id = await self.video_api.submit_video(
                        prompt=para.scene_prompt,
                        reference_image_paths=scene_refs,
                        duration=para_duration,
                        width=self._state.video_width,
                        height=self._state.video_height,
                        seed=self._state.character_seed,
                    )
                    para.video_id = video_id
                    self._save_task_json(para_dir, {"video_id": video_id})
                    pending.append((para.index, video_id, video_path))
                    break
                except Exception as e:
                    if retry < _SUBMIT_RETRIES - 1:
                        await asyncio.sleep(_SUBMIT_RETRY_INTERVAL_BASE_SECONDS * (retry + 1))
                    else:
                        raise

        self.task_manager.update_state(paragraphs=paragraphs)

        for j, (para_idx, video_id, video_path) in enumerate(pending):
            self._check_shutdown()
            para = paragraphs[para_idx]

            await self._emit(
                "video_gen", "running",
                f"Rendering scene video {j + 1}/{len(pending)}",
                _PROGRESS_WAIT_START + _PROGRESS_WAIT_SPAN * (j / max(len(pending), 1)),
            )

            for retry in range(_WAIT_RETRIES):
                try:
                    video_output = await self.video_api.wait_for_video(video_id)
                    video_output.save(video_path)
                    break
                except Exception as e:
                    if retry < _WAIT_RETRIES - 1:
                        await asyncio.sleep(_WAIT_RETRY_INTERVAL_BASE_SECONDS * (retry + 1))
                    else:
                        raise

            para.video_file = video_path
            self.task_manager.update_state(paragraphs=paragraphs)
            await self._validate_and_regenerate_scene(para, para_dir, video_path)

    _SCENE_VALIDATION_MAX_RETRIES = 2

    async def _validate_and_regenerate_scene(
        self, para: ManuscriptParagraph, para_dir: str, video_path: str,
    ) -> None:
        for attempt in range(self._SCENE_VALIDATION_MAX_RETRIES + 1):
            ok, reason = await self._check_scene_video(video_path, para.scene_prompt)
            if ok:
                return

            if attempt >= self._SCENE_VALIDATION_MAX_RETRIES:
                logger.warning(f"[SCENE] Scene {para.index} keeping video despite validation fail: {reason}")
                return

            self._check_shutdown()
            task_json = os.path.join(para_dir, "task.json")
            for stale in (task_json, video_path):
                if os.path.exists(stale):
                    os.remove(stale)

            scene_refs = await self._get_scene_reference_images(para, para_dir)
            try:
                video_id = await self.video_api.submit_video(
                    prompt=para.scene_prompt,
                    reference_image_paths=scene_refs,
                    duration=max(int(math.ceil(len(para.text) / _chars_per_sec(para.text))), 3),
                    width=self._state.video_width,
                    height=self._state.video_height,
                    seed=self._state.character_seed,
                )
                self._save_task_json(para_dir, {"video_id": video_id})
                video_output = await self.video_api.wait_for_video(video_id)
                video_output.save(video_path)
                para.video_id = video_id
                para.video_file = video_path
                self.task_manager.update_state(paragraphs=self._state.paragraphs)
            except Exception as e:
                logger.error(f"[SCENE] Scene {para.index} retry {attempt+1} failed: {e}")

    async def _check_scene_video(self, video_path: str, scene_prompt: str) -> tuple:
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 10_000:
            return False, "missing or empty video file"

        duration = await self._probe_duration(video_path)
        if duration is not None and duration < 1.0:
            return False, f"suspiciously short video ({duration:.2f}s)"

        try:
            frame_path = video_path + ".check_frame.jpg"
            ok = await self._extract_mid_frame(video_path, frame_path)
            if not ok:
                return True, "frame extraction failed, skipping vision check"

            system_prompt = (
                "You are a QA reviewer for AI-generated video scenes featuring a fixed human character."
            )
            user_prompt = (
                "This is a frame from a video scene. The scene was supposed to show: \"" + (scene_prompt or "") + "\"\n\n"
                "Answer with exactly one word, YES or NO: does this image clearly show a woman as the main subject, "
                "actually performing/matching the described scene?"
            )
            answer = await asyncio.to_thread(
                self.screenwriter._chat_multimodal, system_prompt, user_prompt, [frame_path],
            )
            if os.path.exists(frame_path):
                os.remove(frame_path)
            normalized = (answer or "").strip().upper()
            if normalized.startswith("NO"):
                return False, "vision check: main subject not clearly visible / scene mismatch"
            return True, "ok"
        except Exception as e:
            logger.debug(f"[SCENE] vision validation skipped due to error: {e}")
            return True, "vision check unavailable, structural checks only"

    @staticmethod
    async def _extract_mid_frame(video_path: str, frame_path: str) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "select='eq(n\\,0)+gte(t\\,0.001)'",
                "-ss", "00:00:00.5",
                "-frames:v", "1",
                frame_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=20)
            return proc.returncode == 0 and os.path.exists(frame_path)
        except Exception:
            return False

    @staticmethod
    async def _probe_duration(video_path: str) -> Optional[float]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            return float(out.decode().strip())
        except Exception:
            return None

    async def _generate_audio(self) -> object:
        paragraphs = self._state.paragraphs
        audio_config = self._state.audio_config
        full_text = "\n\n".join(p.text for p in paragraphs if p.text)
        if not full_text:
            return None

        audio_path = os.path.join(self.working_dir, "full_narration.mp3")
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            self._state.combined_audio = audio_path
            return await self._recover_sub_maker(
                full_text, self._state.audio_config, self._state.subtitle_config,
            )

        await self._emit("audio", "running", f"Generating narration ({len(full_text)} chars)...", _PROGRESS_AUDIO_START)
        sub_maker = await self._generate_audio_with_fallback(
            output_path=audio_path,
            text=full_text,
            audio_config=audio_config,
            subtitle_config=self._state.subtitle_config,
            duration_sec=0.0,
            empty_placeholder="",
        )

        self._state.combined_audio = audio_path
        self.task_manager.update_state(combined_audio=audio_path)
        return sub_maker

    async def _generate_subtitles(self, sub_maker: object = None) -> None:
        paragraphs = self._state.paragraphs
        subtitle_config = self._state.subtitle_config
        segment_texts = [p.text for p in paragraphs if p.text]
        if not segment_texts:
            return

        segment_durations = [max(len(p.text) / _chars_per_sec(p.text), 2.0) for p in paragraphs]
        await self._emit("subtitle", "running", "Generating captions...", _PROGRESS_SUBTITLE_START)

        srt_path, styles_path = await self.generate_subtitles_common(
            segment_texts=segment_texts,
            segment_durations=segment_durations,
            subtitle_config=subtitle_config,
            sub_maker=sub_maker,
            audio_path=self._state.combined_audio or "",
            screenwriter=self.screenwriter,
            video_width=self._state.video_width,
            video_height=self._state.video_height,
        )

        if styles_path:
            self._state.subtitle_styles_path = styles_path
            self.task_manager.update_state(subtitle_styles_path=styles_path)

        self._state.combined_subtitle = srt_path
        self.task_manager.update_state(combined_subtitle=srt_path)

        word_cues = getattr(sub_maker, "cues", None) if sub_maker else None
        if word_cues:
            try:
                karaoke_data = SubtitleGenerator.generate_karaoke_word_data(word_cues)
                if karaoke_data:
                    words_path = os.path.join(self.working_dir, "karaoke_words.json")
                    with open(words_path, "w", encoding="utf-8") as f:
                        json.dump(karaoke_data, f, ensure_ascii=False)
                    self._state.combined_subtitle_words = words_path
                    self.task_manager.update_state(combined_subtitle_words=words_path)
            except Exception:
                logger.exception("[Manuscript] Karaoke word timing extraction failed")

        # Lip-sync the talking-shot scenes using the SAME word_cues timestamps
        # captions were just built from (requirement: one timing source for
        # both). Runs after scene videos + full narration audio both exist.
        if word_cues:
            await self._apply_lipsync_to_talking_scenes(word_cues)

    async def _apply_lipsync_to_talking_scenes(self, word_cues: list) -> None:
        """Audio-driven lip-sync (Wav2Lip, isolated venv subprocess) applied
        only to the scenes marked as direct-to-camera talking shots in
        _generate_scene_prompts (the odd-indexed scenes -- see
        `is_talking_scene` there). Every other scene stays untouched B-roll.

        Failure handling: if Wav2Lip can't be verified as successful for a
        given scene (see lipsync.LipSyncFailure), that scene's video is left
        as-is -- i.e. it keeps whatever native mouth motion the video model
        produced, WITHOUT being represented as lip-synced anywhere in logs or
        Discord output. We do not retry into a fake success.
        """
        try:
            from core.pipelines.lipsync import run_lipsync_with_retry
        except ImportError:
            logger.warning("[Manuscript] lipsync module not available -- skipping lip-sync stage")
            return

        paragraphs = self._state.paragraphs
        combined_audio = self._state.combined_audio
        if not combined_audio or not os.path.exists(combined_audio):
            logger.warning("[Manuscript] No combined_audio available -- skipping lip-sync stage")
            return

        # Map word_cues (flat, whole-script, in order) back to each paragraph
        # by cumulative word count, so each scene gets exactly the audio span
        # its own line corresponds to -- the same source of truth captions use.
        cue_idx = 0
        total = len(paragraphs)
        lipsync_count = 0
        for i, para in enumerate(paragraphs):
            is_talking_scene = (i % 2 == 1) or (total == 1)
            n_words = len(para.text.split())
            para_cues = word_cues[cue_idx: cue_idx + n_words]
            cue_idx += n_words

            if not is_talking_scene or not para_cues or not para.video_file:
                continue

            start_sec = para_cues[0].start.total_seconds() if hasattr(para_cues[0].start, "total_seconds") else float(para_cues[0].start)
            end_sec = para_cues[-1].end.total_seconds() if hasattr(para_cues[-1].end, "total_seconds") else float(para_cues[-1].end)
            if end_sec <= start_sec:
                logger.warning("[Manuscript] Scene %d: bad cue span (%.2f-%.2f), skipping lip-sync", i, start_sec, end_sec)
                continue

            scene_audio_path = os.path.join(self.working_dir, f"scene_{i}_audio.wav")
            synced_video_path = os.path.join(self.working_dir, f"scene_{i}_lipsynced.mp4")

            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", combined_audio,
                    "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
                    "-ar", "16000", "-ac", "1", scene_audio_path,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode != 0 or not os.path.exists(scene_audio_path):
                    logger.warning("[Manuscript] Scene %d: failed to slice audio span, skipping lip-sync", i)
                    continue
            except Exception:
                logger.exception("[Manuscript] Scene %d: audio slicing errored, skipping lip-sync", i)
                continue

            await self._emit("lipsync", "running", f"Lip-syncing talking scene {i+1}...", None)
            result = await run_lipsync_with_retry(
                face_video_path=os.path.join(self.working_dir, para.video_file) if not os.path.isabs(para.video_file) else para.video_file,
                audio_path=scene_audio_path,
                output_path=synced_video_path,
                retries=1,
            )

            if result is None:
                logger.warning(
                    "[Manuscript] Scene %d: lip-sync could not be verified as successful "
                    "after retry -- leaving original clip in place (NOT reporting lip-sync "
                    "for this scene).", i,
                )
                continue

            para.video_file = synced_video_path
            lipsync_count += 1
            logger.info("[Manuscript] Scene %d: lip-sync verified OK (%.2fs)", i, result.duration_sec)

        logger.info(
            "[Manuscript] Lip-sync stage complete: %d/%d scenes successfully lip-synced.",
            lipsync_count, total,
        )
        # Surface this honestly in task state so the Discord bot can report
        # accurate lip-sync coverage instead of implying it always worked.
        self.task_manager.update_state(lipsync_scenes_synced=lipsync_count, lipsync_scenes_total=total)

    async def _composite_final(self) -> str:
        paragraphs = self._state.paragraphs
        subtitle_config = self._state.subtitle_config
        output_path = os.path.join(self.working_dir, "final_video.mp4")

        if os.path.exists(output_path):
            return output_path

        video_paths = [p.video_file for p in paragraphs if p.video_file and os.path.exists(p.video_file)]
        if not video_paths:
            raise RuntimeError("[Manuscript] No valid video scenes to concatenate")

        has_audio = self._state.audio_config.enabled and bool(self._state.combined_audio)
        has_subtitle = subtitle_config.enabled and bool(self._state.combined_subtitle)
        styles_path = self._state.subtitle_styles_path or ""

        await self._emit("concatenate", "running", "Compositing video + audio + captions...", _PROGRESS_CONCAT_START)

        if has_audio or has_subtitle:
            await asyncio.to_thread(
                VideoConcatenator.concat_videos_with_audio_overlay,
                video_paths=video_paths,
                audio_path=self._state.combined_audio or "",
                srt_path=self._state.combined_subtitle if has_subtitle else None,
                output_path=output_path,
                subtitle_style=subtitle_config.style if has_subtitle else None,
                subtitle_styles_path=styles_path if os.path.exists(styles_path) else None,
                karaoke_words_path=(
                    self._state.combined_subtitle_words
                    if has_subtitle and self._state.combined_subtitle_words
                    and os.path.exists(self._state.combined_subtitle_words)
                    else None
                ),
            )
        else:
            await asyncio.to_thread(VideoConcatenator.concat_videos, video_paths, output_path)

        return output_path
