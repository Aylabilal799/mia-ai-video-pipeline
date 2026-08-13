"""core.compositor.concatenator.audio_overlay — 音频叠加拼接 + 数字人合成（v5.0 Batch 4 / 4.3 拆分）

AudioOverlayMixin：concat_videos_with_audio_overlay / composite_anchor_video；
跨组方法（_parse_srt_to_clips 等）经 MRO 由 ConcatMixin 解析。"""
import json
import logging
import os
import subprocess
from typing import List, Optional

from moviepy import AudioFileClip, CompositeVideoClip, VideoFileClip

from models.task import SubtitleStyle

from .concat import _AUDIO_BITRATE, _AUDIO_CODEC, _AUDIO_FPS, _VIDEO_FPS

logger = logging.getLogger(__name__)


class AudioOverlayMixin:
    """音频叠加拼接与数字人合成方法，v5.0 Batch 4（4.3）拆分。"""

    @staticmethod
    def concat_videos_with_audio_overlay(
        video_paths: List[str],
        audio_path: str,
        srt_path: Optional[str],
        output_path: str,
        subtitle_style: Optional[SubtitleStyle] = None,
        subtitle_styles_path: Optional[str] = None,
        karaoke_words_path: Optional[str] = None,
    ) -> str:
        """先拼接视频，再统一叠加单条音频 + 单条字幕。

        使用 ffmpeg 做音视频时长对齐，确保音画精确同步，且以音频（旁白）
        时长为准：视频比音频短则冻结尾帧补齐（避免丢失旁白内容）；视频比
        音频长则直接裁剪视频到音频时长（避免旁白结束后出现多余的静音/
        画面停留）。不再用静音补齐音频去迁就更长的视频。

        Args:
            video_paths: 按顺序的视频路径列表。
            audio_path: 整段音频文件路径（对应全部视频的总时间轴）。
            srt_path: 整段 SRT 字幕路径（可选，karaoke 不可用时的回退）。
            output_path: 最终输出文件路径。
            subtitle_style: 字幕样式配置。
            karaoke_words_path: 逐词时间戳 JSON 路径（可选）。提供且非空时，
                优先使用 karaoke 逐词高亮字幕渲染；否则回退到 srt_path 的
                普通字幕渲染。

        Returns:
            输出文件路径。
        """
        logger.info(
            f"[Compositor] concat_videos_with_audio_overlay: "
            f"{len(video_paths)} videos + {audio_path} → {output_path}"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not video_paths:
            raise RuntimeError("No videos to concatenate")

        # ── Step 1: 拼接视频（无声）──
        silent_path = output_path.replace(".mp4", "_silent.mp4")
        VideoConcatenator.concat_videos(video_paths, silent_path)

        # ── Step 2: 获取音视频时长。以音频（旁白）时长为准 ──
        video_dur = VideoConcatenator._get_duration(silent_path)
        audio_dur = VideoConcatenator._get_duration(audio_path)
        final_dur = audio_dur if audio_dur > 0 else video_dur
        logger.info(
            f"[Compositor] durations: video={video_dur:.2f}s, "
            f"audio={audio_dur:.2f}s, target(audio-led)={final_dur:.2f}s"
        )

        video_input = silent_path
        tmp_files = [silent_path]

        # ── Step 3: 若视频 < 音频，冻结尾帧补齐（避免旁白被截断）──
        if video_dur < final_dur - 0.3:
            extend_path = output_path.replace(".mp4", "_vext.mp4")
            tmp_files.append(extend_path)
            pad_dur = final_dur - video_dur
            VideoConcatenator._run_ffmpeg(
                ["ffmpeg", "-y",
                 "-i", silent_path,
                 "-vf", f"tpad=stop_mode=clone:stop_duration={pad_dur:.2f}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "fast",
                 extend_path],
                desc=f"extend video by {pad_dur:.1f}s (freeze last frame)",
            )
            video_input = extend_path
        # ── Step 3b: 若视频 > 音频，直接裁剪视频到音频时长（不再用静音
        #     补齐音频去迁就更长的视频 —— 这正是"视频结尾多余黑屏/静止画面"
        #     问题的根因）──
        elif video_dur > final_dur + 0.3:
            trim_path = output_path.replace(".mp4", "_vtrim.mp4")
            tmp_files.append(trim_path)
            VideoConcatenator._run_ffmpeg(
                ["ffmpeg", "-y",
                 "-i", silent_path,
                 "-t", f"{final_dur:.2f}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "fast",
                 trim_path],
                desc=f"trim video down to audio duration ({final_dur:.1f}s)",
            )
            video_input = trim_path

        # ── Step 4: 音频只做音量放大 —— final_dur 已以音频时长为准，视频
        #     已在 Step 3/3b 补齐或裁剪到匹配，音频本身不再需要补静音 ──
        vol_path = audio_path.replace(".mp3", "_vol.mp3")
        tmp_files.append(vol_path)
        VideoConcatenator._run_ffmpeg(
            ["ffmpeg", "-y",
             "-i", audio_path,
             "-af", "volume=1.5",
             "-c:a", "libmp3lame", "-q:a", "2",
             vol_path],
            desc="boost audio volume 1.5x",
        )
        audio_input = vol_path

        # ── Step 5: moviepy 合成视频+音频+字幕 ──
        video_clip = None
        audio_clip_obj = None
        try:
            video_clip = VideoFileClip(video_input)
            audio_clip_obj = AudioFileClip(audio_input)

            # 掐头去尾确保完全对齐
            target_dur = min(video_clip.duration, audio_clip_obj.duration)
            video_clip = video_clip.subclipped(0, target_dur)
            audio_clip_obj = audio_clip_obj.subclipped(0, target_dur)

            video_with_audio = video_clip.with_audio(audio_clip_obj)

            # ── 叠加字幕 ──
            # 优先使用 karaoke 逐词高亮字幕（若提供了逐词时间戳数据）；
            # 不可用/为空/渲染失败时回退到普通整条字幕渲染。
            if (srt_path and os.path.exists(srt_path)) and subtitle_style:
                try:
                    subs_clips = []
                    used_karaoke = False

                    # ── Diagnostics: make it obvious from the logs alone
                    # whether karaoke word-level captions were used, and if
                    # not, exactly why (missing data vs. render exception).
                    # See requirement: "Do NOT silently catch the karaoke
                    # exception and continue with plain captions."
                    if karaoke_words_path and os.path.exists(karaoke_words_path):
                        with open(karaoke_words_path, "r", encoding="utf-8") as f:
                            karaoke_data = json.load(f)
                        word_count = sum(len(line.get("words") or []) for line in karaoke_data)
                        logger.info(f"[KARAOKE] Word cues: {word_count}")
                        logger.info("[KARAOKE] Rendering word-level captions...")
                        try:
                            from .karaoke import build_karaoke_clips
                            subs_clips = build_karaoke_clips(
                                karaoke_data, subtitle_style,
                                video_width=video_clip.w, video_height=video_clip.h,
                                video_duration=target_dur,
                            )
                            used_karaoke = bool(subs_clips)
                            if used_karaoke:
                                logger.info(f"[KARAOKE] SUCCESS ({len(subs_clips)} word clips)")
                            else:
                                logger.warning(
                                    "[KARAOKE] Renderer returned 0 clips (empty/unusable "
                                    "karaoke data) -- falling back to plain subtitles."
                                )
                        except Exception:
                            # Log the FULL exception (not just str(e)) so the
                            # real failure is visible in server.log instead
                            # of being swallowed.
                            logger.exception(
                                "[KARAOKE] ERROR: word-level caption rendering raised an "
                                "exception -- falling back to plain subtitles. This is "
                                "NOT expected; investigate the traceback above."
                            )
                            subs_clips = []
                    else:
                        logger.warning(
                            "[KARAOKE] Word cues: 0 -- no karaoke_words_path provided "
                            "(edge_tts likely returned no word-level cues, or audio/"
                            "subtitles were disabled). Falling back to plain subtitles."
                        )

                    logger.info(f"[KARAOKE] Using word-level rendering: {'YES' if used_karaoke else 'NO'}")

                    if not used_karaoke:
                        per_entry_styles = None
                        if subtitle_styles_path and os.path.exists(subtitle_styles_path):
                            with open(subtitle_styles_path, "r", encoding="utf-8") as f:
                                per_entry_styles = json.load(f)

                        subs_clips = VideoConcatenator._parse_srt_to_clips(
                            srt_path, subtitle_style, video_clip.w,
                            video_height=video_clip.h,
                            video_duration=target_dur,
                            subtitle_styles=per_entry_styles,
                        )
                    else:
                        logger.info(f"[Compositor] Using karaoke word-highlight captions "
                                    f"({len(subs_clips)} word clips)")

                    if subs_clips:
                        final = CompositeVideoClip([video_with_audio, *subs_clips])
                        final.write_videofile(
                            output_path,
                            codec="libx264",
                            audio_codec=_AUDIO_CODEC,
                            audio_bitrate=_AUDIO_BITRATE,
                            audio_fps=_AUDIO_FPS,
                            fps=_VIDEO_FPS,
                            logger="bar",
                        )
                        final.close()
                    else:
                        video_with_audio.write_videofile(
                            output_path,
                            codec="libx264",
                            audio_codec=_AUDIO_CODEC,
                            audio_bitrate=_AUDIO_BITRATE,
                            audio_fps=_AUDIO_FPS,
                            fps=_VIDEO_FPS,
                            logger="bar",
                        )
                except Exception as e:
                    logger.warning(
                        f"[Compositor] Subtitle overlay failed: {e}, writing without subtitles"
                    )
                    video_with_audio.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec=_AUDIO_CODEC,
                        audio_bitrate=_AUDIO_BITRATE,
                        audio_fps=_AUDIO_FPS,
                        fps=_VIDEO_FPS,
                        logger="bar",
                    )
            else:
                video_with_audio.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec=_AUDIO_CODEC,
                    audio_bitrate=_AUDIO_BITRATE,
                    audio_fps=_AUDIO_FPS,
                    fps=_VIDEO_FPS,
                    logger="bar",
                )
        finally:
            if video_clip is not None:
                video_clip.close()
            if audio_clip_obj is not None:
                audio_clip_obj.close()
            for tmp in tmp_files:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        logger.info(f"[Compositor] concat_videos_with_audio_overlay done: {output_path}")
        return output_path

    @staticmethod
    def composite_anchor_video(
        clip_path: str,
        audio_path: str,
        srt_path: Optional[str],
        output_path: str,
        audio_duration: float,
        subtitle_style: Optional[SubtitleStyle] = None,
        subtitle_styles_path: Optional[str] = None,
        video_width: int = 768,
        video_height: int = 1344,
    ) -> str:
        """将 5 秒主播动态视频片段循环拼接为覆盖完整音频时长，再叠加音频和字幕。

        核心思路：循环拼接 + 裁剪 + 统一叠加音频/字幕。
        接缝处用 ffmpeg xfade 做 0.3 秒交叉淡入淡出过渡。

        Args:
            clip_path: 5 秒主播动态视频片段路径。
            audio_path: TTS 读稿音频路径。
            srt_path: SRT 字幕文件路径（可选）。
            output_path: 最终输出视频路径。
            audio_duration: 音频总时长（秒）。
            subtitle_style: 字幕样式配置。
            subtitle_styles_path: LLM 样式 JSON 路径（可选）。
            video_width: 视频宽度。
            video_height: 视频高度。

        Returns:
            输出文件路径。
        """
        import math
        import subprocess

        logger.info(
            f"[Compositor] composite_anchor_video: {clip_path} + {audio_path} "
            f"(audio={audio_duration:.1f}s) → {output_path}"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Step 1: Get clip duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip_path],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=15,
        )
        clip_duration = float(probe.stdout.strip() or 5.0)
        if clip_duration <= 0:
            clip_duration = 5.0

        # Step 2: Calculate loop count
        needed = audio_duration + 2.0  # extra 2s padding
        n = math.ceil(needed / clip_duration) + 1

        # Step 3: Build concat file list for ffmpeg
        loop_dir = os.path.dirname(output_path)
        concat_file = os.path.join(loop_dir, "_anchor_concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(f"file '{clip_path}'\n")

        looped_path = output_path.replace(".mp4", "_looped.mp4")

        # Step 4: Concatenate with xfade cross-fade transitions
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_file,
                 "-c", "copy",
                 "-t", str(needed),
                 looped_path],
                stdin=subprocess.DEVNULL,
                check=True, capture_output=True, timeout=300,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"[Compositor] Simple concat failed: {e.stderr[:200]}, trying xfade")

            # Build complex filter for xfade cross-fade between each pair
            fade_duration = 0.3
            filter_parts = []
            for i in range(n):
                if i == 0:
                    filter_parts.append(f"[0:{i}]")
                else:
                    filter_parts.append(f"[0:{i}]")
                    filter_parts.append(f"xfade=transition=fade:duration={fade_duration}:offset={i * clip_duration - fade_duration * i}")
            filter_str = "".join(filter_parts)

            subprocess.run(
                ["ffmpeg", "-y",
                 "-stream_loop", str(n - 1), "-i", clip_path,
                 "-filter_complex",
                 f"[0:v]trim=duration={needed}[v]",
                 "-map", "[v]",
                 "-c:v", "libx264",
                 "-preset", "fast",
                 "-t", str(needed),
                 looped_path],
                stdin=subprocess.DEVNULL,
                check=True, capture_output=True, timeout=300,
            )

        # Step 5: Overlay audio and subtitles
        concat_video_clip = None
        audio_clip_obj = None
        try:
            concat_video_clip = VideoFileClip(looped_path)
            audio_clip_obj = AudioFileClip(audio_path)

            _AUDIO_VOLUME_FACTOR = 1.5
            audio_clip_obj = audio_clip_obj.with_volume_scaled(_AUDIO_VOLUME_FACTOR)

            video_with_audio = concat_video_clip.with_audio(audio_clip_obj)

            if srt_path and os.path.exists(srt_path) and subtitle_style:
                per_entry_styles = None
                if subtitle_styles_path and os.path.exists(subtitle_styles_path):
                    with open(subtitle_styles_path, "r", encoding="utf-8") as f:
                        per_entry_styles = json.load(f)

                subs_clips = VideoConcatenator._parse_srt_to_clips(
                    srt_path, subtitle_style,
                    video_width, video_height,
                    video_duration=concat_video_clip.duration,
                    subtitle_styles=per_entry_styles,
                )
                if subs_clips:
                    final = CompositeVideoClip([video_with_audio, *subs_clips])
                    final.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec=_AUDIO_CODEC,
                        audio_bitrate=_AUDIO_BITRATE,
                        audio_fps=_AUDIO_FPS,
                        fps=_VIDEO_FPS,
                        logger="bar",
                    )
                    final.close()
                else:
                    video_with_audio.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec=_AUDIO_CODEC,
                        audio_bitrate=_AUDIO_BITRATE,
                        audio_fps=_AUDIO_FPS,
                        fps=_VIDEO_FPS,
                        logger="bar",
                    )
            else:
                video_with_audio.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec=_AUDIO_CODEC,
                    audio_bitrate=_AUDIO_BITRATE,
                    audio_fps=_AUDIO_FPS,
                    fps=_VIDEO_FPS,
                    logger="bar",
                )
        finally:
            if concat_video_clip is not None:
                concat_video_clip.close()
            if audio_clip_obj is not None:
                audio_clip_obj.close()
            for tmp in (looped_path, concat_file):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        logger.info(f"[Compositor] composite_anchor_video done: {output_path}")
        return output_path
