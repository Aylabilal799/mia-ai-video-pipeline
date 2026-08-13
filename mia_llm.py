"""mia_llm.py — LLM helper for script, topic, and SEO package generation for Mia.
"""
import json
import logging
import requests

import mia_config as cfg

logger = logging.getLogger(__name__)


def _chat(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
    if not cfg.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured")

    resp = requests.post(
        cfg.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg.OPENROUTER_MODEL,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Script Generation for Mini-Vlogs
# ---------------------------------------------------------------------------

_SCRIPT_SYSTEM_PROMPT = f"""\
You are the scriptwriter for {cfg.MIA_NAME}, a popular AI lifestyle mini-vlogger on YouTube Shorts, TikTok, and Instagram Reels.

{cfg.MIA_NAME}'s personality: {cfg.MIA_PERSONALITY}

Write a first-person spoken script for a 45–60 second vertical mini-vlog short video.

Rules:
- Write ONLY the words {cfg.MIA_NAME} says out loud.
- NO stage directions, NO brackets, NO scene descriptions, NO headers.
- Total length MUST be between 100 and 140 words (spoken in 45–60 seconds).
- Open with a strong, scroll-stopping hook in the first 1–2 seconds.
- Structure the mini-story with visual progression:
  1. Hook
  2. Setup / Context
  3. Journey / Activity
  4. Interesting moment or takeaway
  5. Satisfying payoff
  6. Natural, casual ending / CTA (e.g., "save this spot for later", "come along next time")
  (Do NOT print these category labels in the output).
- Tone: Casual, authentic, relatable creator talking directly to camera/friends.
- NO fitness instructions, NO numbered tips, NO generic motivational speeches, NO formal narration.
"""


def generate_script(topic: str, category: str) -> str:
    """Generates Mia's mini-vlog script for a given topic + category."""
    cat_info = cfg.MIA_CONTENT_CATEGORIES.get(category, cfg.MIA_CONTENT_CATEGORIES["mini_vlog"])
    user_prompt = (
        f"Category: {cat_info['label']}\n"
        f"Vlog Topic: {topic or cat_info['topic_hint']}\n\n"
        "Write the script now. Output ONLY spoken words."
    )
    try:
        script = _chat(_SCRIPT_SYSTEM_PROMPT, user_prompt, max_tokens=500)
        return _strip_code_fence(script)
    except Exception as e:
        logger.warning("[Mia] LLM script generation failed (%s), using template fallback", e)
        return _template_script(topic, category)


def _template_script(topic: str, category: str) -> str:
    cat_info = cfg.MIA_CONTENT_CATEGORIES.get(category, cfg.MIA_CONTENT_CATEGORIES["mini_vlog"])
    topic = topic or cat_info["topic_hint"]
    return (
        f"I had a couple of free hours today, so I decided to take you along for {topic}! "
        "Honestly, I wasn't expecting much when I first set out, but the moment I arrived, "
        "the whole atmosphere completely grabbed me. I found the cutest corner, tried something "
        "new, and spent time just soaking in the environment. It's funny how a simple change "
        "of scenery can instantly turn your whole day around. Save this for your next free afternoon, "
        "and follow along for the next spot we explore together!"
    )


# ---------------------------------------------------------------------------
# Topic Generation
# ---------------------------------------------------------------------------

def generate_topic(category: str) -> str:
    cat_info = cfg.MIA_CONTENT_CATEGORIES.get(category, cfg.MIA_CONTENT_CATEGORIES["mini_vlog"])
    system_prompt = (
        f"You suggest single, specific, relatable short mini-vlog topics for {cfg.MIA_NAME}, "
        "a lifestyle creator. Examples: 'Finding the cutest hidden café', 'I had 2 hours to explore', "
        "'A rainy day in my life'. Reply with ONLY the topic title, one short line."
    )
    user_prompt = f"Category: {cat_info['label']}. Suggest one mini-vlog topic."
    try:
        topic = _chat(system_prompt, user_prompt, max_tokens=60)
        return _strip_code_fence(topic).strip('"')
    except Exception as e:
        logger.warning("[Mia] LLM topic generation failed (%s), using template fallback", e)
        return cat_info["topic_hint"]


# ---------------------------------------------------------------------------
# YouTube SEO Package
# ---------------------------------------------------------------------------

_SEO_SYSTEM_PROMPT = f"""\
You are a YouTube Shorts SEO specialist for lifestyle mini-vlogs. Given {cfg.MIA_NAME}'s vlog script,
generate a complete SEO package specific to this exact video.

Respond with ONLY a JSON object (no markdown fences) with these exact keys:
{{
  "video_title": "...",
  "description": "...",
  "hook": "...",
  "keywords": "...",
  "tags": "...",
  "hashtags": "...",
  "seo_keywords": "...",
  "target_search_terms": "...",
  "thumbnail_text": "...",
  "video_category": "...",
  "content_angle": "...",
  "short_description": "...",
  "long_description": "...",
  "call_to_action": "..."
}}
"""

_SEO_FIELD_ORDER = [
    ("video_title", "VIDEO TITLE"),
    ("description", "DESCRIPTION"),
    ("hook", "HOOK"),
    ("keywords", "KEYWORDS"),
    ("tags", "TAGS"),
    ("hashtags", "HASHTAGS"),
    ("seo_keywords", "SEO KEYWORDS"),
    ("target_search_terms", "TARGET SEARCH TERMS"),
    ("thumbnail_text", "THUMBNAIL TEXT"),
    ("video_category", "VIDEO CATEGORY"),
    ("content_angle", "CONTENT ANGLE"),
    ("short_description", "SHORT DESCRIPTION"),
    ("long_description", "LONG DESCRIPTION"),
    ("call_to_action", "CALL TO ACTION"),
]


def generate_seo_package(topic: str, category: str, script: str) -> dict:
    cat_info = cfg.MIA_CONTENT_CATEGORIES.get(category, cfg.MIA_CONTENT_CATEGORIES["mini_vlog"])
    user_prompt = f"Category: {cat_info['label']}\nTopic: {topic}\nScript:\n{script}\n"
    try:
        raw = _chat(_SEO_SYSTEM_PROMPT, user_prompt, max_tokens=1200)
        raw = _strip_code_fence(raw)
        data = json.loads(raw)
        return {key: str(data.get(key, "")).strip() for key, _ in _SEO_FIELD_ORDER}
    except Exception as e:
        logger.warning("[Mia] LLM SEO generation failed (%s), using fallback", e)
        return _template_seo_package(topic, category, script)


def _template_seo_package(topic: str, category: str, script: str) -> dict:
    cat_info = cfg.MIA_CONTENT_CATEGORIES.get(category, cfg.MIA_CONTENT_CATEGORIES["mini_vlog"])
    hook = script.strip().split("\n")[0][:80] if script.strip() else topic
    base_tags = ["mia", "minivlog", "dayinmylife", "vlog", "lifestyle"]
    return {
        "video_title": f"{topic.strip().capitalize()} | Mia Mini Vlog",
        "description": f"Come along with Mia for {topic.strip()}!",
        "hook": hook,
        "keywords": f"{topic}, mini vlog, day in my life, lifestyle, Mia",
        "tags": ", ".join(base_tags),
        "hashtags": " ".join(f"#{t}" for t in base_tags) + " #shorts",
        "seo_keywords": f"{topic}, mini vlog aesthetic, day in the life",
        "target_search_terms": f"{topic} vlog, aesthetic mini vlog",
        "thumbnail_text": topic.strip().upper()[:24],
        "video_category": cat_info["label"],
        "content_angle": cat_info["topic_hint"],
        "short_description": f"A mini vlog with Mia: {topic}.",
        "long_description": script.strip(),
        "call_to_action": "Follow for more daily mini vlogs!",
    }


def seo_package_to_txt(seo: dict) -> str:
    lines = []
    for key, label in _SEO_FIELD_ORDER:
        lines.append(label)
        lines.append(seo.get(key, "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"
