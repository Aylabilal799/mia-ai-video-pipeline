"""discord_bot.py — Discord bot interface for Mia Mini-Vlog and standard video generation.
"""
import os
import asyncio
import logging

from dotenv import load_dotenv
import discord
from discord.ext import commands

from tasks import generate_mia_video_task, generate_video_task
from celery.result import AsyncResult
import mia_config as mia_cfg
from schedule_utils import parse_schedule_datetime, ScheduleParseError, SCHEDULE_TIMEZONE

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

user_tasks = {}


@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    await bot.change_presence(activity=discord.Game(name="!mia <topic> | !video <script>"))


def render_progress_bar(percent, width=18):
    percent = max(0, min(100, int(percent)))
    filled = round(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"`[{bar}] {percent}%`"


@bot.command(name='video')
async def video_command(ctx, *, script: str):
    """Generate a YouTube Short from the provided script."""
    if len(script) < 10:
        await ctx.send("❌ Script too short. Please provide at least 10 characters.")
        return

    status_msg = await ctx.send("🎬 Video generation started! This may take a few minutes. I'll notify you when it's ready.")
    task = generate_video_task.delay(script, ctx.author.id)
    user_tasks[ctx.author.id] = task.id
    await check_task_and_respond(ctx, task.id, status_msg)


async def check_task_and_respond(ctx, task_id, status_msg):
    """Poll Celery task for standard !video command."""
    last_stage = None
    last_progress = None
    while True:
        result = AsyncResult(task_id)
        state = result.state

        if state == 'PENDING':
            await asyncio.sleep(5)
            continue
        elif state == 'PROGRESS':
            info = result.info if isinstance(result.info, dict) else {}
            stage = info.get('stage')
            progress = info.get('progress')
            if (stage, progress) != (last_stage, last_progress):
                last_stage, last_progress = stage, progress
                lines = [f"🎬 {stage}" if stage else "🎬 Working..."]
                if isinstance(progress, (int, float)):
                    lines.append(render_progress_bar(progress))
                try:
                    await status_msg.edit(content="\n".join(lines))
                except discord.HTTPException:
                    pass
            await asyncio.sleep(5)
            continue
        elif state == 'SUCCESS':
            video_url = result.result
            if video_url:
                await ctx.send(f"✅ Your video is ready! Download it here (link valid for 24h):\n{video_url}")
            else:
                await ctx.send("❌ Video generation finished but no download link was returned.")
            break
        elif state == 'FAILURE':
            info = result.info
            error = info.get('error', str(info)) if isinstance(info, dict) else str(info)
            await ctx.send(f"❌ Video generation failed: {error}")
            break
        else:
            await ctx.send(f"❓ Unknown state: {state}")
            break


def _parse_mia_args(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None, ""
    parts = raw.split(maxsplit=1)
    first = parts[0].lower()
    if first in mia_cfg.MIA_CONTENT_CATEGORIES:
        topic = parts[1] if len(parts) > 1 else ""
        return first, topic
    return None, raw


@bot.command(name='mia')
async def mia_command(ctx, *, args: str = ""):
    """Generate a Mia mini-vlog video."""
    category, topic = _parse_mia_args(args)
    status_msg = await ctx.send(
        f"💃 Mia mini-vlog started (category: {category or 'auto'})! "
        "Writing script, generating vlog scenes with karaoke captions, "
        "and extracting video thumbnail..."
    )
    task = generate_mia_video_task.delay(topic, category, ctx.author.id, False)
    user_tasks[ctx.author.id] = task.id
    await check_mia_task_and_respond(ctx, task.id, status_msg)


@bot.command(name='miascript')
async def mia_script_command(ctx, *, script: str):
    """Generate a Mia mini-vlog video from a verbatim script."""
    if len(script) < 10:
        await ctx.send("❌ Script too short. Please provide at least 10 characters.")
        return

    # Prevent double-triggers (double-click, resent gateway event, impatient
    # re-run, etc): if this user already has a task in flight, don't start a
    # second one -- it just races the first for the same LLM/rate-limit
    # budget and produces confusing duplicate output.
    existing_task_id = user_tasks.get(ctx.author.id)
    if existing_task_id:
        existing_state = AsyncResult(existing_task_id).state
        if existing_state in ('PENDING', 'PROGRESS', 'STARTED'):
            await ctx.send(
                "⏳ You already have a Mia video generating. Please wait for it "
                "to finish before starting another."
            )
            return

    status_msg = await ctx.send("💃 Mia mini-vlog started from custom script! Generating video...")
    task = generate_mia_video_task.delay(script, None, ctx.author.id, True)
    user_tasks[ctx.author.id] = task.id
    await check_mia_task_and_respond(ctx, task.id, status_msg)


@bot.command(name='miaschedule')
async def mia_schedule_command(ctx, date: str, time: str, *, script: str):
    """Generate a Mia mini-vlog from a verbatim script and schedule its
    YouTube publish time.

    Usage: !miaschedule 2026-08-20 14:30 <script text>
    Date/time are interpreted in the SCHEDULE_TIMEZONE configured in .env.
    """
    if len(script) < 10:
        await ctx.send("❌ Script too short. Please provide at least 10 characters.")
        return

    try:
        publish_at_iso = parse_schedule_datetime(date, time)
    except ScheduleParseError as e:
        await ctx.send(f"❌ {e}")
        return

    existing_task_id = user_tasks.get(ctx.author.id)
    if existing_task_id:
        existing_state = AsyncResult(existing_task_id).state
        if existing_state in ('PENDING', 'PROGRESS', 'STARTED'):
            await ctx.send(
                "⏳ You already have a Mia video generating. Please wait for it "
                "to finish before starting another."
            )
            return

    status_msg = await ctx.send(
        f"💃 Mia mini-vlog started! It'll upload as **private** now, then "
        f"YouTube will auto-publish it at **{date} {time} ({SCHEDULE_TIMEZONE})**..."
    )
    task = generate_mia_video_task.delay(script, None, ctx.author.id, True, publish_at_iso)
    user_tasks[ctx.author.id] = task.id
    await check_mia_task_and_respond(ctx, task.id, status_msg)


async def check_mia_task_and_respond(ctx, task_id, status_msg):
    last_stage = None
    last_progress = None
    while True:
        result = AsyncResult(task_id)
        state = result.state

        if state == 'PENDING':
            await asyncio.sleep(5)
            continue
        elif state == 'PROGRESS':
            info = result.info if isinstance(result.info, dict) else {}
            stage = info.get('stage')
            progress = info.get('progress')
            if (stage, progress) != (last_stage, last_progress):
                last_stage, last_progress = stage, progress
                lines = [f"💃 {stage}" if stage else "💃 Working..."]
                if isinstance(progress, (int, float)):
                    lines.append(render_progress_bar(progress))
                try:
                    await status_msg.edit(content="\n".join(lines))
                except discord.HTTPException:
                    pass
            await asyncio.sleep(5)
            continue
        elif state == 'SUCCESS':
            data = result.result if isinstance(result.result, dict) else {}
            video_url = data.get('video_url')
            thumbnail_url = data.get('thumbnail_url')
            seo_url = data.get('seo_url')
            topic = data.get('topic', '')
            category = data.get('category', '')
            publish_at = data.get('publish_at')

            if video_url:
                msg = (
                    f"✅ **Mia's Mini-Vlog is ready!**\n"
                    f"📌 Category: `{category}` | Topic: *{topic}*\n"
                    f"🎥 **Video Link:** {video_url}"
                )
                if seo_url:
                    msg += f"\n📄 **YouTube SEO Package:** {seo_url}"
                if publish_at:
                    msg += f"\n⏰ **Scheduled YouTube publish:** {publish_at} (UTC)"
                if thumbnail_url:
                    embed = discord.Embed(
                        title=f"🎬 Mia Mini-Vlog: {topic[:50]}",
                        description=f"Category: {category}",
                        color=0x00FF66,
                    )
                    embed.set_image(url=thumbnail_url)
                    await ctx.send(content=msg, embed=embed)
                else:
                    await ctx.send(msg)
            else:
                await ctx.send("❌ Video generation finished but no download link was returned.")
            break
        elif state == 'FAILURE':
            info = result.info
            error = info.get('error', str(info)) if isinstance(info, dict) else str(info)
            await ctx.send(f"❌ Mia mini-vlog generation failed: {error}")
            break
        else:
            await ctx.send(f"❓ Unknown state: {state}")
            break


bot.run(TOKEN)
