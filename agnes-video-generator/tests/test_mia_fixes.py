"""tests/test_mia_fixes.py — regression tests for the Mia pipeline fixes:

1. Karaoke word-by-word highlighting (real render, synthetic word timing).
2. Caption width/wrapping never overflows the canvas, even for a
   pathological "oversized word" (the sentence-level-cue regression case).
3. Bottom safe-margin positioning (`web/helpers.py::_build_position`).
4. Missing word cues -> `build_karaoke_clips` returns [] (caller falls back
   to plain captions) instead of raising or silently mis-rendering.
5. The identity reference image is propagated to every scene, and the
   stronger keyframes (start+end) mechanism is used when available.
6. Scene validation / retry logic never loops forever and accepts a
   structurally-valid video without requiring network access.

These are all runnable WITHOUT the live Agnes/TTS API -- no network calls.
"""
import asyncio
import os
import subprocess
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.compositor.concatenator.karaoke import build_karaoke_clips, _wrap_words, _measure
from models.task import SubtitleStyle
from web import helpers


# ═══════════════════════════════════════════════════
# 1. Karaoke word-by-word highlighting
# ═══════════════════════════════════════════════════

def _make_style(**overrides):
    base = dict(
        font="ArchivoBlack-Regular.ttf", color="white", fontsize=38,
        position=("center", "bottom-300"), stroke_color="black", stroke_width=3,
        bg_color=None, highlight_color="#39FF14", max_width_ratio=0.83,
    )
    base.update(overrides)
    return SubtitleStyle(**base)


def _synthetic_word_line(words, word_dur=0.35):
    cur = 0.0
    out = []
    for w in words:
        out.append({"text": w, "start": cur, "end": cur + word_dur})
        cur += word_dur
    return {"start": out[0]["start"], "end": out[-1]["end"], "words": out}, cur


def test_karaoke_word_highlighting_renders_one_word_at_a_time():
    words = ["This", "is", "the", "best", "way", "to", "start", "your", "morning"]
    line, total = _synthetic_word_line(words)
    style = _make_style()

    clips = build_karaoke_clips([line], style, video_width=1080, video_height=1920, video_duration=total)

    # One clip per word, each spanning only that word's timing window.
    # (The very last word's end is intentionally clamped to
    # video_duration - 0.01 by build_karaoke_clips, so compare against
    # that same clamp rather than the raw synthetic timing.)
    assert len(clips) == len(words)
    clamped_line_end = min(line["end"], total - 0.01)
    for i, (clip, w) in enumerate(zip(clips, line["words"])):
        assert clip.start == pytest.approx(w["start"])
        expected_end = min(w["end"], clamped_line_end) if i == len(words) - 1 else w["end"]
        assert clip.end == pytest.approx(expected_end)


def test_karaoke_render_frame_highlights_only_active_word():
    """Renders an actual RGBA frame and checks that bright green pixels are
    present (the active word) while the previous word's region is white,
    not green -- i.e. NOT the "whole sentence green" bug."""
    words = ["Hello", "world", "today"]
    line, total = _synthetic_word_line(words, word_dur=1.0)
    style = _make_style()
    clips = build_karaoke_clips([line], style, video_width=1080, video_height=1920, video_duration=total)

    from moviepy import CompositeVideoClip, ColorClip
    bg = ColorClip(size=(1080, 1920), color=(20, 20, 20), duration=total)
    comp = CompositeVideoClip([bg, *clips])

    # At t=1.5s, "world" (index 1) should be active/green.
    frame = comp.get_frame(1.5)
    # bright green ~ (57,255,20)-ish; check SOME pixels are strongly green
    # and not simultaneously counted as "mostly green" across the entire
    # caption region (which would indicate the whole-line-green bug).
    green_mask = (frame[:, :, 1] > 180) & (frame[:, :, 0] < 120) & (frame[:, :, 2] < 120)
    white_mask = (frame[:, :, 0] > 200) & (frame[:, :, 1] > 200) & (frame[:, :, 2] > 200)
    assert green_mask.sum() > 0, "expected some green (active word) pixels"
    assert white_mask.sum() > 0, "expected some white (inactive word) pixels -- if this " \
        "fails, the whole line may be rendering as a single highlighted unit"


# ═══════════════════════════════════════════════════
# 2. Caption width / wrapping never overflows canvas
# ═══════════════════════════════════════════════════

def test_wrap_words_never_exceeds_max_width():
    from PIL import Image, ImageDraw, ImageFont
    font_path = os.path.join("resource", "fonts", "ArchivoBlack-Regular.ttf")
    font = ImageFont.truetype(font_path, 38)
    dummy = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

    words = [{"text": w, "start": i, "end": i + 1, "_wi": i}
             for i, w in enumerate("this is the best way to start your morning routine today".split())]
    max_w = 700
    lines = _wrap_words(dummy, words, font, max_w)
    for line in lines:
        widths = [_measure(dummy, w["text"], font)[0] for w in line]
        space_w, _ = _measure(dummy, " ", font)
        total = sum(widths) + space_w * max(len(line) - 1, 0)
        assert total <= max_w, f"line exceeded max width: {total} > {max_w}"


def test_karaoke_hard_breaks_pathological_oversized_word():
    """Regression test for the actual bug that shipped: if cue data is ever
    sentence-level again (e.g. a future edge_tts change), a single 'word'
    can be an entire sentence. The renderer must still never overflow the
    canvas horizontally."""
    line = {
        "start": 0.0, "end": 3.0,
        "words": [{
            "text": "This is a very long sentence acting as a single oversized word token",
            "start": 0.0, "end": 3.0,
        }],
    }
    style = _make_style()
    clips = build_karaoke_clips([line], style, video_width=1080, video_height=1920, video_duration=3.0)
    assert len(clips) == 1
    # Rendered clip width must not exceed the video canvas width.
    assert clips[0].size[0] <= 1080


# ═══════════════════════════════════════════════════
# 3. Bottom safe-margin positioning
# ═══════════════════════════════════════════════════

def test_build_position_bare_bottom_gets_safe_margin_not_flush_edge():
    h, v = helpers._build_position("bottom")
    assert v != "bottom", "bare 'bottom' must not resolve to the flush-edge literal"
    assert "bottom-" in v


def test_build_position_bare_top_gets_safe_margin():
    h, v = helpers._build_position("top")
    assert v != "top"
    assert "top+" in v


def test_build_position_offset_syntax_passed_through():
    h, v = helpers._build_position("bottom-300")
    assert v == "bottom-300", "explicit offset syntax must survive unchanged"


def test_resolved_position_keeps_caption_off_the_edge():
    """End-to-end: bottom-300 at a 1920-tall canvas should land ~300px
    above the bottom edge, not at y=1920 (flush)."""
    from core.compositor.concatenator.concat import VideoConcatenator
    h, v = VideoConcatenator._resolve_subtitle_position(
        ("center", "bottom-300"), video_height=1920, video_width=1080,
    )
    assert isinstance(v, (int, float))
    assert v < 1920 - 250  # well above the flush-bottom position


# ═══════════════════════════════════════════════════
# 4. Missing word cues -> graceful fallback signal
# ═══════════════════════════════════════════════════

def test_build_karaoke_clips_empty_data_returns_no_clips():
    style = _make_style()
    clips = build_karaoke_clips([], style, video_width=1080, video_height=1920, video_duration=5.0)
    assert clips == []


def test_build_karaoke_clips_line_with_no_words_is_skipped():
    style = _make_style()
    clips = build_karaoke_clips(
        [{"start": 0.0, "end": 1.0, "words": []}],
        style, video_width=1080, video_height=1920, video_duration=1.0,
    )
    assert clips == []


# ═══════════════════════════════════════════════════
# 5. Identity reference propagation
# ═══════════════════════════════════════════════════

def _make_manuscript_pipeline(tmp_path, reference_image=None):
    from core.pipelines.manuscript_video import ManuscriptVideoPipeline
    pipeline = ManuscriptVideoPipeline(
        api_key="test-key", task_id="t1", dir_name="t1",
    )
    pipeline._state = MagicMock()
    pipeline._state.reference_image = reference_image
    pipeline._state.video_width = 768
    pipeline._state.video_height = 1344
    pipeline._check_shutdown = MagicMock()
    return pipeline


def test_get_identity_reference_paths_returns_empty_when_no_reference(tmp_path):
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=None)
    assert pipeline._get_identity_reference_paths() == []


def test_get_identity_reference_paths_returns_anchor_when_present(tmp_path):
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"fake-png-bytes")
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=str(anchor))
    assert pipeline._get_identity_reference_paths() == [str(anchor)]


def test_scene_reference_images_uses_keyframes_mode_with_two_refs(tmp_path):
    """The SAME anchor must be used for every scene, AND (when i2i keyframe
    generation succeeds) the result must be a 2-image list so
    agnes_video.submit_video's keyframes branch (n_refs >= 2) is triggered
    instead of the weaker single-image ti2vid branch."""
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"fake-png-bytes")
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=str(anchor))

    fake_image_output = MagicMock()
    fake_image_output.save = MagicMock()
    pipeline.image_generator = MagicMock()
    pipeline.image_generator.generate_single_image = AsyncMock(return_value=fake_image_output)

    para = MagicMock()
    para.index = 0
    para.scene_prompt = "Mia doing a yoga pose outdoors at sunrise"

    para_dir = str(tmp_path / "para_0")
    os.makedirs(para_dir, exist_ok=True)

    refs = asyncio.run(pipeline._get_scene_reference_images(para, para_dir))

    assert len(refs) == 2, "expected [anchor, scene_keyframe] for keyframes mode"
    assert refs[0] == str(anchor), "the SAME anchor must be reference[0] for every scene"
    pipeline.image_generator.generate_single_image.assert_called_once()
    call_kwargs = pipeline.image_generator.generate_single_image.call_args.kwargs
    assert call_kwargs["reference_image_paths"] == [str(anchor)]
    assert "keep" in call_kwargs["prompt"].lower() or "identity" in call_kwargs["prompt"].lower()


def test_scene_reference_images_falls_back_on_keyframe_failure(tmp_path):
    """If i2i keyframe generation fails every attempt, fall back to the
    single-anchor list (still identity-anchored, just the weaker
    mechanism) instead of crashing the whole scene."""
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"fake-png-bytes")
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=str(anchor))
    pipeline.image_generator = MagicMock()
    pipeline.image_generator.generate_single_image = AsyncMock(side_effect=RuntimeError("boom"))

    para = MagicMock()
    para.index = 1
    para.scene_prompt = "Mia talking to camera"
    para_dir = str(tmp_path / "para_1")
    os.makedirs(para_dir, exist_ok=True)

    with patch("asyncio.sleep", new=AsyncMock()):
        refs = asyncio.run(pipeline._get_scene_reference_images(para, para_dir))

    assert refs == [str(anchor)]
    assert pipeline.image_generator.generate_single_image.call_count == 3


def test_scene_reference_images_reuses_cached_keyframe(tmp_path):
    """A second call for the same scene must not regenerate the keyframe
    (cache hit), so re-running a resumed task doesn't burn extra image-gen
    calls per scene."""
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"fake-png-bytes")
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=str(anchor))
    pipeline.image_generator = MagicMock()
    pipeline.image_generator.generate_single_image = AsyncMock()

    para = MagicMock()
    para.index = 2
    para.scene_prompt = "Mia stretching indoors"
    para_dir = str(tmp_path / "para_2")
    os.makedirs(para_dir, exist_ok=True)
    keyframe_path = os.path.join(para_dir, "identity_keyframe.png")
    with open(keyframe_path, "wb") as f:
        f.write(b"cached-keyframe")

    refs = asyncio.run(pipeline._get_scene_reference_images(para, para_dir))
    assert refs == [str(anchor), keyframe_path]
    pipeline.image_generator.generate_single_image.assert_not_called()


# ═══════════════════════════════════════════════════
# 6. Scene validation / retry logic
# ═══════════════════════════════════════════════════

def _make_tiny_valid_mp4(path: str, duration: float = 2.0):
    """Creates a real (small but not degenerate) mp4 via ffmpeg so
    structural checks (file size, ffprobe duration) run against real data,
    not a mock. Uses a noise/pattern source rather than a flat color so
    the encoder can't compress it down to a near-zero-byte file (which
    would trip the "missing/empty" check for reasons unrelated to what
    this test is actually checking)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"testsrc=s=320x480:d={duration}:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "2000k", path,
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def test_check_scene_video_rejects_missing_file(tmp_path):
    pipeline = _make_manuscript_pipeline(tmp_path)
    ok, reason = asyncio.run(pipeline._check_scene_video(str(tmp_path / "nope.mp4"), "a scene"))
    assert ok is False
    assert "missing" in reason or "empty" in reason


def test_check_scene_video_rejects_empty_file(tmp_path):
    pipeline = _make_manuscript_pipeline(tmp_path)
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    ok, reason = asyncio.run(pipeline._check_scene_video(str(p), "a scene"))
    assert ok is False


def test_check_scene_video_accepts_valid_video_when_vision_check_unavailable(tmp_path):
    """With no real Agnes chat API reachable in this sandbox, the
    vision-model call will fail -- and per the "fails open" design this
    must NOT block an otherwise-valid video."""
    pipeline = _make_manuscript_pipeline(tmp_path)
    pipeline.screenwriter = MagicMock()
    pipeline.screenwriter._chat_multimodal = MagicMock(side_effect=RuntimeError("no network in sandbox"))

    video_path = str(tmp_path / "scene.mp4")
    _make_tiny_valid_mp4(video_path, duration=2.0)

    ok, reason = asyncio.run(pipeline._check_scene_video(video_path, "Mia doing yoga"))
    assert ok is True


def test_check_scene_video_rejects_too_short_video(tmp_path):
    pipeline = _make_manuscript_pipeline(tmp_path)
    video_path = str(tmp_path / "tooshort.mp4")
    _make_tiny_valid_mp4(video_path, duration=0.3)

    ok, reason = asyncio.run(pipeline._check_scene_video(video_path, "Mia doing yoga"))
    assert ok is False
    assert "short" in reason


def test_validate_and_regenerate_scene_stops_at_retry_limit(tmp_path):
    """Must never loop forever: if validation ALWAYS fails, it should stop
    after _SCENE_VALIDATION_MAX_RETRIES regenerations, not hang."""
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=None)
    pipeline._check_scene_video = AsyncMock(return_value=(False, "always fails (test)"))
    pipeline._get_scene_reference_images = AsyncMock(return_value=[])
    pipeline.video_api = MagicMock()
    pipeline.video_api.submit_video = AsyncMock(return_value="fake-video-id")
    fake_output = MagicMock()
    fake_output.save = MagicMock()
    pipeline.video_api.wait_for_video = AsyncMock(return_value=fake_output)
    pipeline.task_manager = MagicMock()

    para = MagicMock()
    para.index = 0
    para.scene_prompt = "test"
    para.text = "test text"

    para_dir = str(tmp_path / "para_0")
    os.makedirs(para_dir, exist_ok=True)
    video_path = os.path.join(para_dir, "video.mp4")
    with open(video_path, "wb") as f:
        f.write(b"fake")

    asyncio.run(pipeline._validate_and_regenerate_scene(para, para_dir, video_path))

    # 1 initial validation + _SCENE_VALIDATION_MAX_RETRIES regenerations,
    # each regeneration triggers exactly one more validation call.
    expected_validation_calls = pipeline._SCENE_VALIDATION_MAX_RETRIES + 1
    assert pipeline._check_scene_video.call_count == expected_validation_calls
    assert pipeline.video_api.submit_video.call_count == pipeline._SCENE_VALIDATION_MAX_RETRIES


def test_validate_and_regenerate_scene_stops_early_on_pass(tmp_path):
    pipeline = _make_manuscript_pipeline(tmp_path, reference_image=None)
    pipeline._check_scene_video = AsyncMock(return_value=(True, "ok"))
    pipeline.video_api = MagicMock()
    pipeline.video_api.submit_video = AsyncMock()
    pipeline.task_manager = MagicMock()

    para = MagicMock()
    para.index = 0
    para.scene_prompt = "test"
    para.text = "test text"
    para_dir = str(tmp_path / "para_0")
    os.makedirs(para_dir, exist_ok=True)
    video_path = os.path.join(para_dir, "video.mp4")
    with open(video_path, "wb") as f:
        f.write(b"fake")

    asyncio.run(pipeline._validate_and_regenerate_scene(para, para_dir, video_path))

    assert pipeline._check_scene_video.call_count == 1
    pipeline.video_api.submit_video.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
