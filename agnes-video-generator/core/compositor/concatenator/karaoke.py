"""core.compositor.concatenator.karaoke — word-by-word highlighted (karaoke
style) caption rendering.

Given per-word timing data (see SubtitleGenerator.generate_karaoke_word_data),
renders each caption line as a sequence of image frames: the full line stays
in a fixed position/layout, and only the currently-spoken word's fill color
changes to the highlight color for its own [start, end) window. This avoids
any position jitter between frames (the whole line is laid out identically
every frame; only one word's color changes), which is what makes it read as
"the words popping into color" rather than the whole caption twitching.
"""
import logging
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default "active word" highlight color used by CapCut/TikTok-style
# auto-captions, used only as a fallback when SubtitleStyle.highlight_color
# isn't provided by the caller (see build_karaoke_clips's highlight_color
# param, threaded through from SubtitleStyle in audio_overlay.py).
_HIGHLIGHT_COLOR = "#FFD60A"
_LINE_SPACING = 1.35
_SIDE_MARGIN = 40
_MIN_WORD_DISPLAY_SEC = 0.12


def _measure(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont"):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _hard_break_oversized_word(draw, word: dict, font, max_width: int) -> List[dict]:
    """Defense in depth: if a single "word" is on its own wider than
    max_width (this should not normally happen for real word-level cues,
    but WOULD happen if cue data is ever sentence-level again -- e.g. a
    future edge_tts regression -- since an entire sentence would arrive as
    one atomic "word" that _wrap_words can't break between), hard-split it
    on whitespace/character boundaries so it still can never render past
    the canvas edge. Preserves the same start/end timing on every piece so
    highlighting timing is unaffected.

    Returns a list of word dicts (same shape as the input), always at
    least one element.
    """
    text = word["text"]
    w, _ = _measure(draw, text, font)
    if w <= max_width or not text:
        return [word]

    pieces: List[dict] = []
    # Prefer splitting on spaces (in case a "word" is actually multiple
    # words glued together, e.g. sentence-level cue data).
    tokens = text.split(" ") if " " in text else list(text)
    current = ""
    for tok in tokens:
        candidate = (current + " " + tok) if (current and " " in text) else (current + tok)
        cw, _ = _measure(draw, candidate, font)
        if current and cw > max_width:
            pieces.append({**word, "text": current})
            current = tok
        else:
            current = candidate
    if current:
        pieces.append({**word, "text": current})
    return pieces or [word]


def _wrap_words(draw, words: list, font, max_width: int) -> List[list]:
    """Greedy word-wrap. Returns a list of lines, each a list of the same
    word dicts passed in (not copies unless hard-broken -- see
    _hard_break_oversized_word), so identity/index comparisons still work
    against the original `words` list in the common case."""
    # Guarantee no single "word" is wider than the available canvas before
    # doing line-wrapping, so a pathological oversized token can never
    # extend past the canvas edge (see _hard_break_oversized_word).
    safe_words: list = []
    for w in words:
        safe_words.extend(_hard_break_oversized_word(draw, w, font, max_width))

    lines: List[list] = []
    current: list = []
    current_width = 0
    space_w, _ = _measure(draw, " ", font)
    for w in safe_words:
        word_w, _ = _measure(draw, w["text"], font)
        add_w = word_w if not current else space_w + word_w
        if current and current_width + add_w > max_width:
            lines.append(current)
            current = [w]
            current_width = word_w
        else:
            current.append(w)
            current_width += add_w
    if current:
        lines.append(current)
    return lines


def _render_frame(
    words: list, active_idx: int, font, base_color: str, highlight_color: str,
    stroke_color: str, stroke_width: int, canvas_width: int, max_text_width: int,
) -> np.ndarray:
    """Renders one RGBA frame of the full line, with the word at
    `active_idx` drawn in `highlight_color` and all others in `base_color`.

    Matches by each word dict's stable "_wi" index tag (assigned once by
    the caller before any hard-breaking) rather than a running position
    counter, so highlighting still lands on the correct piece even if
    _wrap_words had to hard-split an oversized "word" into multiple
    rendered pieces (see _hard_break_oversized_word) -- a plain counter
    would drift out of sync with `active_idx` in that case.
    """
    dummy_img = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)

    lines = _wrap_words(dummy_draw, words, font, max_text_width)
    space_w, _ = _measure(dummy_draw, " ", font)
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * _LINE_SPACING)
    canvas_height = line_h * len(lines) + stroke_width * 4

    img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = stroke_width * 2
    for line in lines:
        widths = [_measure(dummy_draw, w["text"], font)[0] for w in line]
        total_w = sum(widths) + space_w * max(len(line) - 1, 0)
        x = (canvas_width - total_w) / 2
        for w, ww in zip(line, widths):
            color = highlight_color if w.get("_wi") == active_idx else base_color
            draw.text(
                (x, y), w["text"], font=font, fill=color,
                stroke_width=stroke_width, stroke_fill=stroke_color,
            )
            x += ww + space_w
        y += line_h

    return np.array(img)


def build_karaoke_clips(
    karaoke_data: list,
    subtitle_style,
    video_width: int,
    video_height: int,
    video_duration: float = 0.0,
) -> list:
    """Builds moviepy ImageClips implementing word-by-word highlighted
    captions from structured per-word timing data (see
    SubtitleGenerator.generate_karaoke_word_data). Returns [] if there's
    nothing renderable (caller should fall back to plain SRT captions)."""
    if not karaoke_data:
        return []

    from moviepy import ImageClip
    from core.config import resolve_font_path
    from core.compositor.concatenator.concat import VideoConcatenator

    font_path = resolve_font_path(subtitle_style.font)
    fontsize = subtitle_style.fontsize or 46
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except Exception:
        logger.warning("[Karaoke] Could not load font %s (fontsize=%s), using PIL default",
                        font_path, fontsize)
        font = ImageFont.load_default()

    base_color = subtitle_style.color or "white"
    stroke_color = subtitle_style.stroke_color or "black"
    stroke_width = subtitle_style.stroke_width or 3
    highlight_color = getattr(subtitle_style, "highlight_color", None) or _HIGHLIGHT_COLOR

    # Max caption width: prefer the configurable max_width_ratio (fraction
    # of video_width) so this is tunable from SubtitleStyle instead of a
    # hardcoded margin constant; fall back to the old fixed-margin behavior
    # for any caller still constructing a bare SubtitleStyle without it.
    max_width_ratio = getattr(subtitle_style, "max_width_ratio", None)
    if max_width_ratio:
        max_text_width = max(int(video_width * max_width_ratio), 100)
    else:
        max_text_width = max(video_width - _SIDE_MARGIN * 2, 100)
    logger.info(f"[CAPTIONS] Font size: {fontsize}")
    logger.info(f"[CAPTIONS] Max width: {max_text_width}px (video_width={video_width})")

    h_part, v_part = VideoConcatenator._resolve_subtitle_position(
        subtitle_style.position, video_height=video_height, video_width=video_width,
    )
    logger.info(f"[CAPTIONS] Position: {subtitle_style.position} -> resolved ({h_part}, {v_part})")

    clips = []
    for line in karaoke_data:
        words = line.get("words") or []
        if not words:
            continue
        # Stable index tag so hard-broken pieces (see
        # _hard_break_oversized_word) can still be matched back to the
        # correct "active" word during rendering.
        for _i, _w in enumerate(words):
            _w["_wi"] = _i
        line_start = line["start"]
        line_end = line["end"]
        if video_duration > 0:
            line_end = min(line_end, video_duration - 0.01)
            if line_end <= line_start:
                continue

        for i, w in enumerate(words):
            disp_start = max(w["start"], line_start)
            disp_end = words[i + 1]["start"] if i + 1 < len(words) else line_end
            disp_end = min(disp_end, line_end)
            if disp_end <= disp_start:
                disp_end = disp_start + _MIN_WORD_DISPLAY_SEC
            if video_duration > 0:
                disp_end = min(disp_end, video_duration - 0.01)
                if disp_end <= disp_start:
                    continue

            frame = _render_frame(
                words, i, font, base_color, highlight_color,
                stroke_color, stroke_width, video_width, max_text_width,
            )
            clip = ImageClip(frame)
            clip = (
                clip.with_start(disp_start)
                .with_end(disp_end)
                .with_duration(disp_end - disp_start)
            )
            v = v_part
            if isinstance(v, (int, float)):
                v = max(20, min(v, video_height - frame.shape[0] - 20))
            h = h_part if isinstance(h_part, (int, float)) else "center"
            clip = clip.with_position((h, v))
            clips.append(clip)

    logger.info("[Karaoke] Built %d word-highlight clips from %d lines", len(clips), len(karaoke_data))
    return clips
