"""core.pipelines.manuscript_video -- Manuscript Video Generation Pipeline
"""
import asyncio
import json
import logging
import math
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

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


def _lipsync_enabled() -> bool:
    """Lip-sync (Wav2Lip) is OFF by default. Current pipeline direction is
    voiceover narration + AI-generated lifestyle visuals + karaoke captions,
    not forced mouth-sync -- this was a deliberate product decision (Wav2Lip
    was failing/OOMing on CPU and its payoff for Shorts retention is
    unproven). Set AGNES_ENABLE_LIPSYNC=1 in the environment to re-enable
    later if analytics show it's actually needed.
    """
    return os.environ.get("AGNES_ENABLE_LIPSYNC", "0").strip() == "1"


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
# NOTE: these chars-per-second thresholds are ONLY used to decide where to
# split the manuscript into paragraphs (text segmentation) and as a last-
# resort fallback if TTS timing is unavailable. Once real TTS word cues
# exist (the normal case, see FIX #2 below), actual scene *duration* is
# always taken from those cues, never from this estimate.
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

# FIX #1 -- identity lock. Layered on top of whatever the screenwriter wrote
# for a scene, and only when a reference image is actually being sent with
# the request (never rely on prompt text alone -- the real reference image
# is still what's supplied to every scene generation call).
#
# NOTE: this locks FACIAL / CHARACTER identity only (face, facial
# structure, eyes/nose/lips, hair identity & hairstyle, apparent age,
# overall character identity). Clothing/outfit is intentionally NOT locked
# here -- wardrobe is allowed to vary scene-to-scene so Mia can be dressed
# appropriately for whatever each scene is depicting. A clothing change
# between scenes is expected behavior, not an identity failure.
_IDENTITY_LOCK_PREFIX = (
    "[IDENTITY -- do not change] The woman in this scene is the exact same "
    "woman from the provided reference image: same face, same facial "
    "structure, same eyes/nose/lips, same hair identity and hairstyle, same "
    "apparent age, same overall character identity. Do not alter her "
    "identity in any way. Her clothing/outfit MAY change naturally between "
    "scenes to suit the scene -- wardrobe variation is expected and is not "
    "an identity change.\n\n"
)

# FIX #2 -- assembly-time tolerance. Scene clips within this many seconds of
# their target narration span are left untouched (re-encoding a
# near-perfect match just burns time and quality for no benefit).
_DURATION_CONFORM_TOLERANCE_SECONDS = 0.35

# FIX #3 -- reference-image leak guard. If a generated scene's extracted
# frame is an average-hash match to the canonical anchor/reference image
# within this Hamming distance, it is treated as the reference image having
# leaked into scene output (not real generated content) and the scene is
# failed rather than accepted. This is deliberately tight: two different
# photos of the same woman in different poses/backgrounds will differ far
# more than this once she's actually doing something in a scene.
_REFERENCE_LEAK_HAMMING_THRESHOLD = 3.0

# FIX #4 -- reference-image lead-in strip. Agnes's image-to-video mode
# anchors the first frame(s) of a freshly generated clip to the exact
# conditioning/reference image before real motion starts. That lead-in must
# never reach the final composited video as scene content. These control
# how far into a clip (and at what sampling resolution) we scan for that
# leading run of reference-matching frames before giving up and treating a
# still-matching window as a genuine generation failure instead.
#
# FIX #4b -- lead-in scan MUST be treated as authoritative, not best-effort.
# The original implementation broke out of the scan loop identically
# whether a sampled frame (a) did not match the reference (real motion
# started -- genuinely done) or (b) could not even be extracted (ffmpeg
# failure / clip not fully flushed yet / bad seek). Both cases fell through
# to the same "return, nothing more to do" path, which meant a failed
# extraction at t=0 -- the single hardest, most failure-prone frame to grab,
# since it's requested immediately after the clip is written to disk --
# silently looked identical to "no lead-in here, clip is clean". That is
# how the anchor frame was reaching the final video: stripping silently did
# nothing and nobody was told. `_strip_reference_leadin` now returns a bool
# and an extraction failure is NEVER treated as "clip is clean" -- see its
# docstring and both call sites in `_generate_videos` /
# `_validate_and_regenerate_scene`.
_REFERENCE_LEADIN_MAX_SCAN_SECONDS = 1.5
_REFERENCE_LEADIN_SAMPLE_STEP_SECONDS = 0.15


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

        # FIX #2 state: {paragraph_index: (scene_start_sec, scene_end_sec)}
        # derived from real TTS word cues in _build_scenes, and the cached
        # sub_maker so _generate_audio doesn't have to regenerate narration
        # that was already produced up-front.
        self._scene_time_spans: Dict[int, Tuple[float, float]] = {}
        self._narration_sub_maker = None

        # FIX #3 state: paragraph indices whose scene video could not be
        # validated (missing/empty/too-short/reference-image-leak/vision
        # mismatch) after exhausting retries. These scenes are dropped
        # (para.video_file set back to None) rather than silently kept or
        # backfilled with the reference image. _composite_final refuses to
        # ship a video while this set is non-empty.
        self._failed_scene_indices: set = set()

    def _get_watermark_language_text(self) -> str:
        return self._state.manuscript_text

    # ------------------------------------------------------------------
    # FIX #2: TTS-first master timeline
    # ------------------------------------------------------------------

    @staticmethod
    def _cue_seconds(v) -> float:
        return v.total_seconds() if hasattr(v, "total_seconds") else float(v)

    def _derive_scene_spans_from_cues(
        self, paragraphs: List[ManuscriptParagraph], word_cues: list,
    ) -> Dict[int, Tuple[float, float]]:
        """Maps the flat, whole-script word_cues (in paragraph order) back
        to each paragraph by cumulative word count -- the same technique
        already used for lip-sync -- then sets each paragraph's scene span
        to [own first-word start, next paragraph's first-word start), with
        the final paragraph's span running to the last cue's end.

        Using the NEXT paragraph's start (rather than this paragraph's own
        last-word end) as the boundary means scenes tile the narration with
        zero gaps, including any natural pause between sentences -- so the
        concatenated video timeline matches the narration timeline exactly,
        which is the actual requirement (not just "each scene's own words
        are covered").
        """
        cue_idx = 0
        per_para: List[Tuple[int, Optional[float], Optional[float]]] = []
        for para in paragraphs:
            n_words = len(para.text.split()) if para.text else 0
            para_cues = word_cues[cue_idx: cue_idx + n_words]
            cue_idx += n_words
            if para_cues:
                start = self._cue_seconds(para_cues[0].start)
                end = self._cue_seconds(para_cues[-1].end)
            else:
                start = end = None
            per_para.append((para.index, start, end))

        last_end = self._cue_seconds(word_cues[-1].end) if word_cues else None

        spans: Dict[int, Tuple[float, float]] = {}
        for i, (idx, start, end) in enumerate(per_para):
            if start is None:
                continue
            next_start = None
            if i + 1 < len(per_para):
                next_start = per_para[i + 1][1]
            span_end = next_start if next_start is not None else (last_end if last_end is not None else end)
            span_end = max(span_end, start + 0.5)
            spans[idx] = (start, span_end)
        return spans

    async def _ensure_scene_spans_from_tts(self) -> Dict[int, Tuple[float, float]]:
        """FIX #2 core step. Generates (or, on resume, recovers) the full
        narration audio and its real word/token timestamps BEFORE any scene
        video is submitted, then derives every paragraph's exact
        scene_start/scene_end from those timestamps -- not from
        len(text) / chars_per_second.

        Idempotent by design: if `full_narration.mp3` already exists (a
        prior run, or a resumed task), this recovers cues from it instead
        of regenerating, using the same `_recover_sub_maker` path
        `_generate_audio` already relied on -- so `_generate_audio` later
        in the pipeline just reuses what's cached here instead of doing a
        second TTS pass.

        Returns {} (never raises) if audio is disabled or there's no text
        to narrate -- callers fall back to the old chars-per-second
        estimate only in that case.
        """
        paragraphs = self._state.paragraphs
        if not paragraphs or not self._state.audio_config.enabled:
            return {}

        full_text = "\n\n".join(p.text for p in paragraphs if p.text)
        if not full_text:
            return {}

        audio_path = os.path.join(self.working_dir, "full_narration.mp3")
        try:
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                sub_maker = await self._recover_sub_maker(
                    full_text, self._state.audio_config, self._state.subtitle_config,
                )
            else:
                await self._emit(
                    "audio", "running",
                    f"Generating narration ({len(full_text)} chars)...",
                    _PROGRESS_AUDIO_START,
                )
                sub_maker = await self._generate_audio_with_fallback(
                    output_path=audio_path,
                    text=full_text,
                    audio_config=self._state.audio_config,
                    subtitle_config=self._state.subtitle_config,
                    duration_sec=0.0,
                    empty_placeholder="",
                )
        except Exception:
            logger.exception(
                "[Manuscript] Up-front TTS generation failed -- scene timing "
                "will fall back to chars-per-second estimate for this run"
            )
            return {}

        self._state.combined_audio = audio_path
        self.task_manager.update_state(combined_audio=audio_path)
        self._narration_sub_maker = sub_maker

        word_cues = getattr(sub_maker, "cues", None) if sub_maker else None
        if not word_cues:
            logger.warning(
                "[Manuscript] TTS produced no word cues -- scene timing will "
                "fall back to chars-per-second estimate for this run"
            )
            return {}

        return self._derive_scene_spans_from_cues(paragraphs, word_cues)

    def _scene_duration_seconds(self, para: ManuscriptParagraph, spans: Dict[int, Tuple[float, float]]) -> int:
        span = spans.get(para.index)
        if span:
            return max(int(round(span[1] - span[0])), 2)
        # Fallback ONLY when real TTS timing wasn't available this run.
        return max(int(math.ceil(len(para.text) / _chars_per_sec(para.text))), 3)

    # ------------------------------------------------------------------

    async def _build_scenes(self) -> None:
        if not self._state.paragraphs:
            paragraphs = self._split_text(self._state.manuscript_text)
            self._state.paragraphs = paragraphs
            self.task_manager.update_state(paragraphs=paragraphs)
        else:
            logger.info("[Manuscript] Reusing %d existing paragraphs", len(self._state.paragraphs))

        await self._generate_scene_prompts(self._state.paragraphs)

        # FIX #2: real narration timing, computed BEFORE scene videos are
        # submitted, is the single source of truth for scene duration.
        self._scene_time_spans = await self._ensure_scene_spans_from_tts()

        self._state.scenes = [
            SceneTask(
                index=p.index,
                scene_prompt=p.scene_prompt,
                narration_text=p.text,
                duration=self._scene_duration_seconds(p, self._scene_time_spans),
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
        """FIX #1: every scene uses ONLY the single canonical MIA anchor
        image as its identity reference (image-to-video, one reference),
        instead of ALSO generating a fresh per-scene "identity keyframe"
        image.

        The old path sent Agnes [anchor, freshly-generated keyframe], which
        put Agnes into keyframes mode. Because each keyframe was itself
        generated independently, scene by scene, THAT keyframe was free to
        drift the face/hair/identity -- which is what produced different-
        looking women between scenes, even though the anchor itself never
        changed. Passing the anchor alone keeps one, unmodified, reused
        identity source across every scene (image-to-video / "ti2vid" mode
        in AgnesVideoAPI.submit_video when exactly one reference is given).

        IMPORTANT: this path (and `_get_identity_reference_paths`) is the
        ONLY place the canonical reference image is ever referenced in this
        pipeline. It is returned here purely to be handed to Agnes as
        `reference_image_paths` (an identity INPUT to video generation in
        agnes_video.py). It must never be assigned to `para.video_file`,
        appended to the concat list in `_composite_final`, or used as a
        stand-in/fallback frame anywhere. See `_frame_matches_reference`
        for the runtime guard that enforces this even if a future change
        accidentally reintroduces a fallback.
        """
        return self._get_identity_reference_paths()

    def _identity_locked_prompt(self, scene_prompt: str, has_reference_image: bool) -> str:
        """Do not rely on prompt text alone for identity -- the real
        reference image is still what's supplied via
        `_get_scene_reference_images`. This just makes the accompanying
        text explicit and forbids identity drift, per Agnes's own
        recommendation for i2v/keyframe prompts.

        This locks FACE / CHARACTER identity only. It deliberately does not
        constrain clothing/outfit -- wardrobe is allowed to vary between
        scenes (see `_IDENTITY_LOCK_PREFIX`).
        """
        if not has_reference_image:
            return scene_prompt
        return _IDENTITY_LOCK_PREFIX + (scene_prompt or "")

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

            # Talking-to-camera shots are only generated when lip-sync is
            # actually enabled. With lip-sync off (the current default),
            # generating a close-up "talking directly to camera" shot with
            # no real mouth-sync just draws attention to the mismatch, so
            # every scene uses natural lifestyle/B-roll framing instead
            # (spec section 12: avoid fake talking shots without lipsync).
            is_talking_scene = _lipsync_enabled() and ((i % 2 == 1) or (total == 1))

            if is_talking_scene:
                prompt_instructions = (
                    f"Narration line: \"{para.text}\"\n"
                    f"Scene {i+1} of {total} in a mini-vlog video.\n"
                    "Create a clear visual prompt featuring Mia in a realistic 9:16 vlog shot.\n"
                    "- Mia is talking DIRECTLY TO CAMERA, as if speaking this exact line to her "
                    "audience. Her full face must be clearly visible and unobstructed (no hand, "
                    "hair, or object covering her mouth), facing or nearly facing the lens, in a "
                    "medium close-up or chest-up framing.\n"
                    "- Read the narration line above and pick a speaking expression and demeanor "
                    "that actually match what is being narrated -- not a default reaction reused "
                    "for every line. For example: surprising or bad news -> genuine surprise or "
                    "concern, not smiling; rushing or running late -> visibly hurried, energetic "
                    "delivery; thinking, deciding, or waiting -> neutral or thoughtful; discovering "
                    "or noticing something -> natural curiosity; enjoying something small like a "
                    "coffee -> relaxed, subtle happiness, not a big grin; something funny or "
                    "embarrassing -> a natural small laugh or smile, not exaggerated; a calm or "
                    "peaceful moment -> a genuine, settled expression. Do NOT default to a constant "
                    "happy smile regardless of content, and avoid exaggerated, theatrical acting.\n"
                    "- Natural, animated speaking expression and subtle head/hand gestures while "
                    "she talks -- not a static, frozen pose, and not an exaggerated grin held for "
                    "the whole shot.\n"
                    "- Handheld selfie-vlog camera style, slight natural handheld motion.\n"
                    "- She is the exact same woman shown in the provided reference image -- do not "
                    "change her face, facial structure, eyes/nose/lips, hair identity/hairstyle, "
                    "apparent age, or overall character identity. Her clothing/outfit MAY vary "
                    "naturally between scenes -- wardrobe changes are allowed and expected.\n"
                    "\n"
                    "HARD REQUIREMENT: the prompt you write MUST make it obvious, just from reading "
                    "it, which specific moment of the narration line above it depicts. A generic "
                    "prompt that would look correct on almost any other line of this script is a "
                    "FAILURE and must not be produced."
                )
            else:
                prompt_instructions = (
                    f"Narration line: \"{para.text}\"\n"
                    f"Scene {i+1} of {total} in a mini-vlog video.\n"
                    "Create a clear visual prompt featuring Mia in a realistic 9:16 vlog shot.\n"
                    "\n"
                    "First, work out from the narration line above:\n"
                    "  1. ACTION -- the concrete physical thing she is doing right now that this "
                    "line describes (e.g. approaching a queue, receiving a drink, sitting by a "
                    "window, unboxing something, walking somewhere specific) -- not a generic "
                    "attractive pose. Extract this from what the line actually says, whatever the "
                    "topic of this particular script is.\n"
                    "  2. EXPRESSION -- the facial expression and demeanor a real person would "
                    "naturally have while this is happening, matched to what the line is actually "
                    "narrating -- never apply the same reaction to every line. For example: "
                    "surprising or bad news -> genuine surprise or concern, not a smile; rushing or "
                    "running late -> visibly hurried, urgent energy; thinking, deciding, or waiting "
                    "-> a neutral or thoughtful expression; discovering or noticing something -> "
                    "natural curiosity; enjoying something small like a coffee or a nice moment -> "
                    "relaxed, subtle happiness, not a big grin; something funny or embarrassing -> a "
                    "natural small laugh or smile, not exaggerated; a calm or peaceful ending -> a "
                    "genuine, settled expression. Do NOT default to a smile -- only use a smile if "
                    "the line's content actually calls for one, and keep it subtle rather than a "
                    "constant camera-facing grin. Avoid exaggerated, theatrical acting.\n"
                    "  3. EYE DIRECTION -- where she is naturally looking given the action (e.g. at "
                    "the object/place/person the line describes, into the distance, down at what "
                    "she's holding). She should be looking at what she's doing, NOT at the camera, "
                    "unless the action itself would naturally involve a brief glance toward it.\n"
                    "  4. BODY LANGUAGE -- posture/gesture consistent with the action and mood (e.g. "
                    "leaning slightly forward, relaxed shoulders, hands around a cup, casual stride, "
                    "or visibly quicker/urgent movement if she's rushing) rather than a static, "
                    "frozen posed stance.\n"
                    "Then write the visual prompt so it clearly conveys all four of those, specific "
                    "to THIS line -- never reuse the same expression/action/pose you'd use for a "
                    "different line just because it's also Mia in a vlog.\n"
                    "\n"
                    "- Handheld vlog camera style, medium or three-quarter shot, smooth natural "
                    "movement, one clear main action per scene.\n"
                    "- If other people appear in the background, keep it to a SMALL number (roughly "
                    "2-5, fewer for a quiet/indoor setting), each moving at a natural, realistic "
                    "walking pace. Do not describe a large crowd or dozens of pedestrians. Avoid "
                    "background people freezing mid-motion, looping the same movement, teleporting "
                    "position, sliding instead of stepping, or walking through the main character. "
                    "Background presence should stay secondary to Mia -- she remains the visual "
                    "focus of the shot.\n"
                    "- She is the exact same woman shown in the provided reference image -- do not "
                    "change her face, facial structure, eyes/nose/lips, hair identity/hairstyle, "
                    "apparent age, or overall character identity. Her clothing/outfit MAY vary "
                    "naturally between scenes to suit the action -- wardrobe changes are allowed "
                    "and expected, not an identity failure.\n"
                    "\n"
                    "HARD REQUIREMENT: the prompt you write MUST make it obvious, just from reading "
                    "it, which specific moment of the narration line above it depicts. A generic "
                    "prompt that would look correct on almost any other line of this script (e.g. "
                    "'woman walking through the city', 'woman smiling on a city street') is a "
                    "FAILURE and must not be produced -- rewrite it around the concrete "
                    "ACTION/EXPRESSION/EYE DIRECTION you worked out in steps 1-4 instead."
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
        scene_by_index = {s.index: s for s in self._state.scenes}
        total = len(paragraphs)
        pending: list[tuple[int, str, str]] = []

        for i, para in enumerate(paragraphs):
            self._check_shutdown()
            para_dir = os.path.join(self.working_dir, f"para_{para.index}")
            video_path = os.path.join(para_dir, "video.mp4")

            if os.path.exists(video_path):
                # A cached clip from a prior run must clear the exact same
                # bar a freshly generated one does -- existence on disk is
                # not acceptance. Strip any reference-image lead-in, then
                # validate. If either fails, discard the stale cache (clip +
                # task.json) and fall through to normal submit/wait so the
                # scene gets regenerated below, instead of ever setting
                # para.video_file on an unverified cached clip.
                cache_ok = await self._strip_reference_leadin(para, video_path)
                if cache_ok:
                    cache_ok, _reason = await self._check_scene_video(video_path, para.scene_prompt)
                if cache_ok:
                    para.video_file = video_path
                    continue
                logger.warning(
                    "[Manuscript] Scene %d: cached clip failed lead-in/validation "
                    "check on resume, discarding cache and regenerating",
                    para.index,
                )
                task_json = os.path.join(para_dir, "task.json")
                for stale in (task_json, video_path):
                    if os.path.exists(stale):
                        os.remove(stale)

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

            # FIX #2: duration comes from the scene plan, which was set in
            # _build_scenes from real TTS word-cue spans (falls back to the
            # chars-per-second estimate only if TTS timing wasn't available).
            scene = scene_by_index.get(para.index)
            para_duration = scene.duration if scene else self._scene_duration_seconds(para, self._scene_time_spans)

            # FIX #1: single canonical anchor reference only. This path
            # supplies Agnes's `reference_image_paths` identity INPUT --
            # it never becomes scene output (see `_get_scene_reference_images`
            # docstring and `_frame_matches_reference`).
            scene_refs = await self._get_scene_reference_images(para, para_dir)
            scene_prompt = self._identity_locked_prompt(para.scene_prompt, bool(scene_refs))

            submit_failed = False
            for retry in range(_SUBMIT_RETRIES):
                try:
                    video_id = await self.video_api.submit_video(
                        prompt=scene_prompt,
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
                        # FIX #3: a scene that can't even be submitted is a
                        # failed scene, not a reason to kill the whole task
                        # and never a reason to substitute the reference
                        # image. Record it and move on to the next scene.
                        logger.error(
                            "[Manuscript] Scene %d: submit_video failed after %d retries, "
                            "dropping scene: %s", para.index, _SUBMIT_RETRIES, e,
                        )
                        self._failed_scene_indices.add(para.index)
                        submit_failed = True
            if submit_failed:
                continue

        self.task_manager.update_state(paragraphs=paragraphs)

        for j, (para_idx, video_id, video_path) in enumerate(pending):
            self._check_shutdown()
            para = paragraphs[para_idx]

            await self._emit(
                "video_gen", "running",
                f"Rendering scene video {j + 1}/{len(pending)}",
                _PROGRESS_WAIT_START + _PROGRESS_WAIT_SPAN * (j / max(len(pending), 1)),
            )

            wait_failed = False
            for retry in range(_WAIT_RETRIES):
                try:
                    video_output = await self.video_api.wait_for_video(video_id)
                    video_output.save(video_path)
                    # DIAG (temporary, no behavior change): fingerprint the
                    # raw clip BEFORE strip runs, so we can tell on the next
                    # run whether the leak is present in the source clip at
                    # all, and whether the pre-strip scan's own frame grabs
                    # are themselves suspect (corrupt/truncated) at t=0.
                    await self._diag_log_leadin_state(para, video_path, "PRE-STRIP")
                    # FIX #4: strip any reference-image lead-in frames before
                    # this clip is used for anything else. A False return
                    # means we could NOT confirm the clip is clean (either a
                    # lead-in was found but couldn't be trimmed, or a scan
                    # frame couldn't even be extracted) -- treat that the
                    # same as any other generation failure for this attempt
                    # rather than letting an unconfirmed clip pass through.
                    if not await self._strip_reference_leadin(para, video_path):
                        raise RuntimeError(
                            f"scene {para.index}: could not confirm clip is free of "
                            f"reference-image lead-in"
                        )
                    # DIAG (temporary, no behavior change): fingerprint the
                    # SAME video_path AGAIN immediately after strip claims
                    # success, so we can see directly whether the file on
                    # disk at this exact path is actually clean at t=0.
                    await self._diag_log_leadin_state(para, video_path, "POST-STRIP")
                    break
                except Exception as e:
                    if retry < _WAIT_RETRIES - 1:
                        await asyncio.sleep(_WAIT_RETRY_INTERVAL_BASE_SECONDS * (retry + 1))
                    else:
                        # FIX #3: same as above -- fail this scene cleanly,
                        # never fall back to inserting the reference image.
                        logger.error(
                            "[Manuscript] Scene %d: wait_for_video failed after %d retries, "
                            "dropping scene: %s", para.index, _WAIT_RETRIES, e,
                        )
                        self._failed_scene_indices.add(para.index)
                        wait_failed = True
            if wait_failed:
                continue

            para.video_file = video_path
            self.task_manager.update_state(paragraphs=paragraphs)
            await self._validate_and_regenerate_scene(para, para_dir, video_path)

    _SCENE_VALIDATION_MAX_RETRIES = 2

    async def _validate_and_regenerate_scene(
        self, para: ManuscriptParagraph, para_dir: str, video_path: str,
    ) -> None:
        scene_by_index = {s.index: s for s in self._state.scenes}
        for attempt in range(self._SCENE_VALIDATION_MAX_RETRIES + 1):
            ok, reason = await self._check_scene_video(video_path, para.scene_prompt)
            if ok:
                self._failed_scene_indices.discard(para.index)
                return

            if attempt >= self._SCENE_VALIDATION_MAX_RETRIES:
                # FIX #3: a scene that never validates is FAILED, not kept
                # "despite" the failure. Drop the clip and clear
                # para.video_file so nothing downstream (concat, thumbnails,
                # etc.) can pick it up. _composite_final refuses to ship
                # while any scene is in this state.
                logger.error(
                    "[Manuscript] Scene %d FAILED validation after %d retries, dropping "
                    "clip: %s", para.index, self._SCENE_VALIDATION_MAX_RETRIES, reason,
                )
                self._failed_scene_indices.add(para.index)
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError:
                        logger.warning("[Manuscript] Scene %d: could not remove failed clip %s", para.index, video_path)
                # NOTE: must stay a valid str, not None -- ManuscriptVideoTask
                # requires video_file: str. Setting None here previously made
                # the persisted task state fail Pydantic validation on the
                # next load, which made TaskManager unable to load the task
                # and made /api/tasks/{id} 404 mid-run (surfaced upstream as
                # a misleading "Agnes task not found" error). An empty string
                # is still falsy, so _composite_final's `if p.video_file and
                # os.path.exists(p.video_file)` filter still correctly drops
                # this scene from the concat list.
                para.video_file = ""
                self.task_manager.update_state(paragraphs=self._state.paragraphs)
                return

            self._check_shutdown()
            task_json = os.path.join(para_dir, "task.json")
            for stale in (task_json, video_path):
                if os.path.exists(stale):
                    os.remove(stale)

            # FIX #1: still just the single canonical anchor on regeneration.
            scene_refs = await self._get_scene_reference_images(para, para_dir)
            scene_prompt = self._identity_locked_prompt(para.scene_prompt, bool(scene_refs))
            scene = scene_by_index.get(para.index)
            para_duration = scene.duration if scene else self._scene_duration_seconds(para, self._scene_time_spans)
            try:
                video_id = await self.video_api.submit_video(
                    prompt=scene_prompt,
                    reference_image_paths=scene_refs,
                    duration=para_duration,
                    width=self._state.video_width,
                    height=self._state.video_height,
                    seed=self._state.character_seed,
                )
                self._save_task_json(para_dir, {"video_id": video_id})
                video_output = await self.video_api.wait_for_video(video_id)
                video_output.save(video_path)
                await self._diag_log_leadin_state(para, video_path, "PRE-STRIP-RETRY")
                # FIX #4: strip any reference-image lead-in frames before
                # this regenerated clip is re-validated. As above, treat
                # "could not confirm clean" as a failed regeneration attempt
                # rather than silently accepting an unconfirmed clip.
                if not await self._strip_reference_leadin(para, video_path):
                    logger.warning(
                        "[Manuscript] Scene %d retry %d: could not confirm clip is free "
                        "of reference-image lead-in, treating as failed regeneration",
                        para.index, attempt + 1,
                    )
                    continue
                await self._diag_log_leadin_state(para, video_path, "POST-STRIP-RETRY")
                para.video_id = video_id
                para.video_file = video_path
                self.task_manager.update_state(paragraphs=self._state.paragraphs)
            except Exception as e:
                logger.error(f"[SCENE] Scene {para.index} retry {attempt+1} failed: {e}")

    async def _extract_frame_at(self, video_path: str, output_path: str, timestamp: float) -> bool:
        """Extracts a single frame from video_path at `timestamp` seconds.

        Returns True only when ffmpeg exits cleanly (returncode == 0), the
        output JPG exists, and it's larger than 1000 bytes. Any invalid or
        truncated extracted frame is deleted and this returns False.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
                "-frames:v", "1", "-q:v", "2", output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15)

            valid = (
                proc.returncode == 0
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 1000
            )
            if not valid:
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                return False
            return True
        except Exception as e:
            logger.debug(f"[SCENE] frame extraction at {timestamp}s failed: {e}")
            return False

    async def _check_scene_video(self, video_path: str, scene_prompt: str) -> tuple:
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 10_000:
            return False, "missing or empty video file"

        duration = await self._probe_duration(video_path)
        if duration is not None and duration < 1.0:
            return False, f"suspiciously short video ({duration:.2f}s)"

        try:
            frame_paths = []
            if duration:
                timestamps = [max(0.1, duration * f) for f in (0.25, 0.5, 0.75)]
                for i, ts in enumerate(timestamps):
                    fp = f"{video_path}.check_frame_{i}.jpg"
                    if await self._extract_frame_at(video_path, fp, ts):
                        frame_paths.append(fp)

            # FIX #4b safety net: the 25/50/75% samples above never cover the
            # very start of the clip, which is exactly where a reference
            # lead-in lives. _strip_reference_leadin should already have
            # removed any lead-in before this function ever runs, but this
            # gives the reference-leak check below a chance to catch it here
            # too if that ever silently doesn't happen.
            start_frame_path = f"{video_path}.check_frame_start.jpg"
            if await self._extract_frame_at(video_path, start_frame_path, 0.1):
                frame_paths.append(start_frame_path)

            if not frame_paths:
                fp = video_path + ".check_frame.jpg"
                if await self._extract_mid_frame(video_path, fp):
                    frame_paths.append(fp)

            if not frame_paths:
                return True, "frame extraction failed, skipping vision check"

            ref_paths = self._get_identity_reference_paths()
            if ref_paths:
                for fp in frame_paths:
                    if self._frame_matches_reference(fp, ref_paths[0]):
                        for f in frame_paths:
                            if os.path.exists(f):
                                os.remove(f)
                        return False, "generated frame matches the reference/anchor image -- not real scene content"

            system_prompt = (
                "You are a QA reviewer for AI-generated video scenes featuring a fixed human character."
            )
            user_prompt = (
                "This is one frame sampled from a video scene. The scene overall was "
                "supposed to show: \"" + (scene_prompt or "") + "\"\n\n"
                "This single frame will only ever show ONE moment of that action, not "
                "all of it -- that is expected and fine.\n\n"
                "Answer with exactly one word, YES or NO: is a woman clearly present "
                "and recognizable as the main subject of this frame, in a setting/pose "
                "broadly consistent with the scene (not an empty shot, not someone "
                "else, not an unrelated scene)?"
            )

            passed = False
            for fp in frame_paths:
                answer = await asyncio.to_thread(
                    self.screenwriter._chat_multimodal, system_prompt, user_prompt, [fp],
                )
                normalized = (answer or "").strip().upper()
                if not normalized.startswith("NO"):
                    passed = True
                    break

            for fp in frame_paths:
                if os.path.exists(fp):
                    os.remove(fp)

            if not passed:
                return False, "vision check: main subject not clearly visible / scene mismatch"
            return True, "ok"
        except Exception as e:
            logger.debug(f"[SCENE] vision validation skipped due to error: {e}")
            return True, "vision check unavailable, structural checks only"

    @staticmethod
    def _frame_matches_reference(frame_path: str, ref_image_path: str, threshold: float = _REFERENCE_LEAK_HAMMING_THRESHOLD) -> bool:
        """Cheap average-hash comparison between a generated scene's
        extracted frame and the canonical anchor/reference image.

        Returns True only when the frame is essentially a static copy of
        the reference photo (e.g. some fallback path leaked the anchor
        image into scene output instead of real generated video) --
        NOT true for two different photos of the same woman, which will
        differ well past `threshold` once she's mid-action, in a different
        pose, framing, or background, as any real generated scene will be.

        Fails OPEN (returns False / "not a match") if Pillow isn't
        available or comparison errors -- this is a safety guard, not a
        required dependency, and must never itself block the pipeline.
        """
        try:
            from PIL import Image
        except ImportError:
            return False
        try:
            def ahash(path):
                img = Image.open(path).convert("L").resize((16, 16))
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                return [1 if p > avg else 0 for p in pixels]

            h1 = ahash(frame_path)
            h2 = ahash(ref_image_path)
            hamming = sum(a != b for a, b in zip(h1, h2))
            return hamming <= threshold
        except Exception:
            logger.debug("[Manuscript] reference-leak comparison failed, treating as no match", exc_info=True)
            return False

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

    async def _diag_log_leadin_state(self, para: ManuscriptParagraph, video_path: str, tag: str) -> None:
        """DIAGNOSTIC ONLY -- no effect on pipeline behavior or output.

        Logs, for `video_path` at this exact moment: real path, mtime,
        size, and whether a freshly-extracted frame at t=0.00 / t=0.15 /
        t=0.30 (a) could even be extracted at all and (b) hashes as a
        match against the canonical reference image. This exists purely
        to confirm or rule out, on the next real run, whether frame
        extraction immediately after a clip is written to disk is
        returning corrupt/truncated frames that read as "not a match"
        (and therefore short-circuit `_strip_reference_leadin`'s scan)
        even though the file is not actually clean yet.

        Never raises -- any failure here is swallowed and logged, since
        this must not be able to affect the real pipeline run.
        """
        try:
            ref_paths = self._get_identity_reference_paths()
            if not ref_paths or not os.path.exists(video_path):
                return
            ref_image_path = ref_paths[0]
            real_path = os.path.realpath(video_path)
            try:
                stat = os.stat(video_path)
                size_bytes, mtime = stat.st_size, stat.st_mtime
            except OSError:
                size_bytes, mtime = -1, -1.0

            samples = []
            for t in (0.0, 0.15, 0.30):
                probe_path = f"{video_path}.diag_{tag}_{t:.2f}.jpg"
                extracted = await self._extract_frame_at(video_path, probe_path, t)
                if not extracted:
                    samples.append((t, "EXTRACT_FAILED", None))
                    continue
                probe_size = os.path.getsize(probe_path) if os.path.exists(probe_path) else -1
                is_match = self._frame_matches_reference(probe_path, ref_image_path)
                hamming = None
                try:
                    from PIL import Image
                    def _ahash(p):
                        img = Image.open(p).convert("L").resize((16, 16))
                        px = list(img.getdata())
                        avg = sum(px) / len(px)
                        return [1 if v > avg else 0 for v in px]
                    h1, h2 = _ahash(probe_path), _ahash(ref_image_path)
                    hamming = sum(a != b for a, b in zip(h1, h2))
                except Exception:
                    pass
                samples.append((t, f"jpg_bytes={probe_size} match={is_match} hamming={hamming}", None))
                if os.path.exists(probe_path):
                    os.remove(probe_path)

            logger.warning(
                "[DIAG][%s] Scene %d: path=%s realpath=%s size=%d mtime=%.3f | %s",
                tag, para.index, video_path, real_path, size_bytes, mtime,
                " | ".join(f"t={t:.2f}s -> {info}" for t, info, _ in samples),
            )
        except Exception:
            logger.exception("[DIAG][%s] Scene %d: diagnostic logging itself failed", tag, para.index)

    async def _strip_reference_leadin(self, para: ManuscriptParagraph, video_path: str) -> bool:
        """FIX #4. Agnes's image-to-video mode anchors a freshly generated
        clip's first frame(s) to the exact conditioning/reference image
        before real motion starts -- confirmed by hash-scanning an actual
        output video, where the leading ~0.2-0.4s of scene clips matched
        the canonical anchor image at near-zero Hamming distance. That
        lead-in is a valid identity-locking artifact of generation, but it
        must never reach the final composited video as scene content.

        This scans the first _REFERENCE_LEADIN_MAX_SCAN_SECONDS of the
        SAVED clip file (in place, immediately after Agnes writes it) and
        trims off any leading run of frames that match the reference image,
        before the clip is validated, duration-conformed, or concatenated.

        If the entire scan window still matches the reference image, this
        intentionally does nothing -- that's a real generation failure (no
        motion was ever produced), not a trimmable lead-in, and is left to
        the existing reference-match guard in `_check_scene_video` to fail
        the scene properly rather than being silently deleted here.

        Returns:
            True  -- the clip is CONFIRMED clean: either no lead-in was
                     found (scan sampled real, non-reference frames), the
                     entire scan window still matched (handed off to
                     `_check_scene_video` to fail properly), or a detected
                     lead-in was successfully trimmed.
            False -- the clip could NOT be confirmed clean. Either a
                     detected lead-in existed but ffmpeg failed to trim it,
                     OR -- critically -- a scan frame could not even be
                     extracted (e.g. the clip wasn't fully flushed yet, a
                     bad seek, a transient ffmpeg error). A failed
                     extraction is NEVER treated as "no lead-in here": that
                     conflation is what previously let reference frames
                     reach the final video silently. Callers must fail this
                     scene (retry or drop it) rather than proceed, per the
                     requirement to never insert the anchor image into
                     scene output.
        """
        ref_paths = self._get_identity_reference_paths()
        if not ref_paths or not os.path.exists(video_path):
            return True
        ref_image_path = ref_paths[0]

        cutoff = 0.0
        matched_any = False
        extraction_failed = False
        t = 0.0
        while t <= _REFERENCE_LEADIN_MAX_SCAN_SECONDS:
            frame_path = f"{video_path}.leadin_{t:.2f}.jpg"
            if not await self._extract_frame_at(video_path, frame_path, t):
                # NOT the same as "real motion started". We genuinely don't
                # know whether this frame would have matched the reference
                # or not, so we cannot certify the clip as clean.
                extraction_failed = True
                break
            is_match = self._frame_matches_reference(frame_path, ref_image_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)
            if not is_match:
                break
            matched_any = True
            cutoff = t + _REFERENCE_LEADIN_SAMPLE_STEP_SECONDS
            t += _REFERENCE_LEADIN_SAMPLE_STEP_SECONDS
        else:
            # Every sampled frame in the scan window matched the reference --
            # leave the clip untouched; _check_scene_video will fail it.
            return True

        if extraction_failed:
            logger.warning(
                "[Manuscript] Scene %d: could not extract frame(s) while scanning for "
                "reference lead-in at %.2fs -- cannot confirm clip is free of the anchor "
                "image, failing this attempt rather than risk shipping it",
                para.index, t,
            )
            return False

        if not matched_any or cutoff <= 0.0:
            return True

        base, ext = os.path.splitext(video_path)
        trimmed_path = f"{base}_noleadin{ext or '.mp4'}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", f"{cutoff:.3f}", "-i", video_path,
                trimmed_path,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0 and os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 10_000:
                os.replace(trimmed_path, video_path)
                logger.info(
                    "[Manuscript] Scene %d: stripped %.2fs of reference-image lead-in from clip",
                    para.index, cutoff,
                )
                return True
            logger.warning(
                "[Manuscript] Scene %d: could not strip reference lead-in (ffmpeg failed), "
                "cannot confirm clip is clean",
                para.index,
            )
            return False
        except Exception:
            logger.exception("[Manuscript] Scene %d: error stripping reference lead-in", para.index)
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
        # FIX #2: narration audio (and its word cues) was already generated
        # up-front in _build_scenes -> _ensure_scene_spans_from_tts, since
        # real TTS timing must exist BEFORE scene videos and captions are
        # built. This just returns the cached result instead of generating
        # a second time. The defensive branches below only fire if that
        # up-front step was skipped or failed for some reason.
        if self._narration_sub_maker is not None and self._state.combined_audio:
            return self._narration_sub_maker

        paragraphs = self._state.paragraphs
        audio_config = self._state.audio_config
        full_text = "\n\n".join(p.text for p in paragraphs if p.text)
        if not full_text:
            return None

        audio_path = os.path.join(self.working_dir, "full_narration.mp3")
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            self._state.combined_audio = audio_path
            sub_maker = await self._recover_sub_maker(
                full_text, self._state.audio_config, self._state.subtitle_config,
            )
            self._narration_sub_maker = sub_maker
            return sub_maker

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
        self._narration_sub_maker = sub_maker
        return sub_maker

    async def _generate_subtitles(self, sub_maker: object = None) -> None:
        paragraphs = self._state.paragraphs
        subtitle_config = self._state.subtitle_config
        segment_texts = [p.text for p in paragraphs if p.text]
        if not segment_texts:
            return

        # FIX #2: caption segment durations now come from the SAME cue-
        # derived spans scene videos were timed against, not a separate
        # chars-per-second estimate -- one timing source, as required.
        # (The word-level karaoke highlighting below already used
        # sub_maker.cues directly and is unaffected either way.)
        segment_durations = [
            (self._scene_time_spans[p.index][1] - self._scene_time_spans[p.index][0])
            if self._scene_time_spans.get(p.index)
            else max(len(p.text) / _chars_per_sec(p.text), 2.0)
            for p in paragraphs if p.text
        ]
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
        if word_cues and _lipsync_enabled():
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
            is_talking_scene = _lipsync_enabled() and ((i % 2 == 1) or (total == 1))
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

    async def _conform_scene_to_span(self, para: ManuscriptParagraph, target_duration: float) -> None:
        """FIX #2 (assembly step). The video model only produces clips
        close to a requested duration, not frame-exact, and per-scene
        rounding error accumulates across a whole manuscript. Before
        concatenation, trim or freeze-extend each scene clip so its length
        matches the exact narration span for that paragraph:
          - actual > target -> trim to target_duration.
          - actual < target -> freeze (clone) the last frame to fill the gap,
            so the NEXT scene never starts early relative to the narration.
        Clips already within tolerance are left untouched. On any ffmpeg
        failure this logs a warning and keeps the original clip rather than
        breaking the run.
        """
        video_path = para.video_file
        if not video_path or not os.path.exists(video_path) or target_duration <= 0:
            return

        actual = await self._probe_duration(video_path)
        if actual is None:
            return
        diff = actual - target_duration
        if abs(diff) <= _DURATION_CONFORM_TOLERANCE_SECONDS:
            return

        base, ext = os.path.splitext(video_path)
        conformed_path = f"{base}_conformed{ext or '.mp4'}"

        try:
            if diff > 0:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", video_path, "-t", f"{target_duration:.3f}",
                    "-c", "copy", conformed_path,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode != 0 or not os.path.exists(conformed_path):
                    # Stream copy can miss the target if it doesn't land on a
                    # keyframe -- fall back to a re-encoded trim.
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-y", "-i", video_path, "-t", f"{target_duration:.3f}",
                        conformed_path,
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=60)
            else:
                pad_seconds = target_duration - actual
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}",
                    "-t", f"{target_duration:.3f}",
                    conformed_path,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.communicate(), timeout=60)

            if os.path.exists(conformed_path) and os.path.getsize(conformed_path) > 10_000:
                para.video_file = conformed_path
            else:
                logger.warning(
                    "[Manuscript] Scene %d: duration conform produced no usable output "
                    "(actual=%.2fs target=%.2fs), keeping original clip",
                    para.index, actual, target_duration,
                )
        except Exception:
            logger.exception(
                "[Manuscript] Scene %d: duration conform errored "
                "(actual=%.2fs target=%.2fs), keeping original clip",
                para.index, actual, target_duration,
            )

    async def _composite_final(self) -> str:
        paragraphs = self._state.paragraphs
        subtitle_config = self._state.subtitle_config
        output_path = os.path.join(self.working_dir, "final_video.mp4")

        if os.path.exists(output_path):
            return output_path

        # FIX #2: conform every scene clip to its exact narration span
        # before concatenation, so the assembled video's timeline matches
        # the narration timeline rather than drifting from accumulated
        # per-scene generation error.
        for para in paragraphs:
            self._check_shutdown()
            span = self._scene_time_spans.get(para.index)
            if span:
                await self._conform_scene_to_span(para, span[1] - span[0])

        # DIAG (temporary, no behavior change): fingerprint the EXACT file
        # each paragraph is about to hand to the concatenator, at the exact
        # path used, immediately before concatenation. Answers directly:
        # "is the cleaned file the one actually passed to concat?"
        for para in paragraphs:
            if para.video_file and os.path.exists(para.video_file):
                await self._diag_log_leadin_state(para, para.video_file, "PRE-CONCAT")

        video_paths = [p.video_file for p in paragraphs if p.video_file and os.path.exists(p.video_file)]
        if not video_paths:
            raise RuntimeError("[Manuscript] No valid video scenes to concatenate")

        # FIX #3: refuse to silently ship an incomplete video. If any scene
        # was dropped by submit/wait/validation failure (never backfilled
        # with the reference image -- see _generate_videos and
        # _validate_and_regenerate_scene), stop here with a clear error that
        # names exactly which scenes are missing, instead of compositing
        # fewer clips than the script called for without telling anyone.
        if self._failed_scene_indices:
            raise RuntimeError(
                f"[Manuscript] {len(self._failed_scene_indices)} scene(s) failed and were "
                f"dropped (paragraph indices: {sorted(self._failed_scene_indices)}). Refusing "
                f"to composite an incomplete video -- fix or re-run those scenes."
            )

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
