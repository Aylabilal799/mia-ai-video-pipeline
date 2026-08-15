import shutil

PATH = "/root/deepseekyt/agnes-video-generator/core/pipelines/manuscript_video.py"

OLD = '''    async def _check_scene_video(self, video_path: str, scene_prompt: str) -> tuple:
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

            # FIX #3: hard guard -- if the generated scene's frame is
            # basically a static copy of the canonical reference/anchor
            # image, this is NOT valid scene content (the reference image
            # is an identity INPUT only, see `_get_scene_reference_images`)
            # and must be failed rather than accepted.
            ref_paths = self._get_identity_reference_paths()
            if ref_paths and self._frame_matches_reference(frame_path, ref_paths[0]):
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                return False, "generated frame matches the reference/anchor image -- not real scene content"

            system_prompt = (
                "You are a QA reviewer for AI-generated video scenes featuring a fixed human character."
            )
            user_prompt = (
                "This is a frame from a video scene. The scene was supposed to show: \\"" + (scene_prompt or "") + "\\"\\n\\n"
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
            return True, "vision check unavailable, structural checks only"'''

NEW = '''    async def _extract_frame_at(self, video_path: str, output_path: str, timestamp: float) -> bool:
        """Extracts a single frame from video_path at `timestamp` seconds."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", f"{timestamp:.2f}", "-i", video_path,
                "-vframes", "1", "-q:v", "2", output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
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
                "supposed to show: \\"" + (scene_prompt or "") + "\\"\\n\\n"
                "This single frame will only ever show ONE moment of that action, not "
                "all of it -- that is expected and fine.\\n\\n"
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
            return True, "vision check unavailable, structural checks only"'''

with open(PATH, "r", encoding="utf-8") as f:
    content = f.read()

if OLD not in content:
    print("NO MATCH FOUND -- file differs from what I expected. Nothing was changed.")
    print("This means the file on disk doesn't exactly match the block I have on record.")
    raise SystemExit(1)

count = content.count(OLD)
if count > 1:
    print(f"Found {count} matches, expected exactly 1 -- refusing to touch it, nothing changed.")
    raise SystemExit(1)

shutil.copy(PATH, PATH + ".bak")
content = content.replace(OLD, NEW)
with open(PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched successfully. Backup saved at:", PATH + ".bak")
