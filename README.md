# YouTube Shorts Bot – Free API Edition

Generates Shorts from Discord using Agnes AI's Manuscript Video pipeline (scene
generation + TTS narration + subtitles, all done by Agnes itself).

## What was fixed

The original `video_generator.py` shelled out to a `start.sh --mode manuscript
--text-file ... --output-dir ...` command that doesn't exist. Agnes Video
Generator is actually a **local web app** (FastAPI + browser UI on port 8765).
Its real `start.sh` starts that server and then opens a browser tab for you to
fill in a form by hand -- which is exactly what you were seeing on your Debian
box.

This version instead:
- Starts Agnes's `server.py` directly in the background (detached, no browser).
- Talks to its real REST API (`POST /api/tasks/manuscript`, polls
  `GET /api/tasks/{id}`, downloads `GET /api/video/{id}`).
- Lets Agnes do scene splitting, AI video generation, TTS narration, and
  subtitle burn-in itself, so the bot no longer needs its own `edge-tts` /
  `ffmpeg` captioning step.
- Fixes a bug in `discord_bot.py` where a failed task would crash the bot
  instead of reporting the error, because Celery stores the raised exception
  object (not a dict) as `result.info` on failure.

End-to-end flow is now: `!video <script>` in Discord → bot queues it → Agnes
generates the video in the background → finished MP4 gets posted back to the
channel. No manual re-pasting into a browser.

## Setup

1. Create the folder and paste all files.
2. Run: `./setup.sh`
3. Edit `.env` with your tokens (`DISCORD_TOKEN`, `AGNES_API_KEY`).
4. Run `./start_worker.sh` in one terminal.
5. Run `./start_bot.sh` in another terminal.
6. In Discord: `!video Your script here...`

The first `!video` request will take a little longer than later ones, since
it also has to boot the Agnes server in the background. Progress messages
will show what stage it's at.

## Troubleshooting

- **Nothing happens / errors quickly**: check `agnes-video-generator/server.log`
  -- that's the Agnes server's own log.
- **"AGNES_API_KEY is not set"**: get a free key from
  https://platform.agnes-ai.com and put it in `.env`.
- **Port 8765 already in use**: something else is bound to it, or a previous
  Agnes server is still running. `./stop_agnes_server.sh` then try again, or
  `lsof -ti:8765 | xargs kill`.
- **Video generation seems stuck / status never resolves**: the exact JSON
  field names Agnes uses for task status could differ by version; the code
  will raise a clear error after ~5 minutes of unrecognized responses and
  dump the raw response so you can see the actual field name to adjust.

## Requirements

- Debian/Ubuntu
- Internet connection
- Redis (installed by setup.sh)
- Python 3.10+ (for Agnes's own venv) and ffmpeg (installed by setup.sh)
