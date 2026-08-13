"""mia_config.py — configuration for Mia, recurring AI lifestyle mini-vlogger.
"""
import os
import random

# OpenRouter configuration for script & SEO generation
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_API_KEY_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

MIA_NAME = "Mia"
MIA_PERSONALITY = (
    "A candid, upbeat, and relatable lifestyle mini-vlogger. Mia loves discovering "
    "hidden café spots, exploring city streets, sharing spontaneous micro-adventures, "
    "and giving her audience an authentic look into her daily life."
)

# Identity anchor description (used for initial reference image generation)
MIA_CHARACTER_DESCRIPTION = (
    "A 24-year-old female lifestyle creator with warm almond-brown eyes, natural shoulder-length "
    "light-brown hair, friendly approachable facial features, medium build"
)

# Visual prompt layer added to all generated video scenes for mini-vlogs
MIA_CHARACTER_STYLE_FOR_SCENES = (
    "Realistic short-form creator vlog footage, 9:16 vertical video. The subject is physically "
    "present in the real-world scene at all times -- never shown inside a phone screen, on a "
    "device display, or as a video-within-a-video. "
    "Natural handheld camera movement, stabilized vlog motion, medium shot, three-quarter shot, "
    "or direct-to-camera selfie angle. Authentic daylight, natural realistic human movement, fluid motion. "
    "No static standing poses, no exaggerated slow motion, no dramatic cinematic lighting, no impossible camera spins."
)

# Fixed identity anchor path and generation parameters
MIA_REFERENCE_IMAGE = "assets/mia_anchor.png"
MIA_SEED = 42

MIA_GENERATION_WIDTH = 768
MIA_GENERATION_HEIGHT = 1344

# TTS Configuration (AvaNeural for upbeat creator voice)
MIA_VOICE = "af_heart"
MIA_VOICE_RATE = "+5%"
MIA_VOICE_LANG = "en-US"

# 38px Vibrant Green Word-Highlight Karaoke Captions
# CapCut/TikTok-style karaoke captions: white idle text, green active word.
# NOTE: fontsize is applied at MIA_GENERATION_WIDTH (768px), not the final
# output width, since Agnes burns captions in *before* the Shorts postprocess
# resize. 38px on a 768-wide frame reads noticeably smaller than 38px would
# on a 1080-wide frame -- bumped to 46 to match a CapCut-proportional caption
# at this generation width.
MIA_CAPTION_FONT = "Montserrat ExtraBold"
MIA_CAPTION_FONT_SIZE = 46
MIA_CAPTION_COLOR = "#FFFFFF"  # idle/unspoken words: white
MIA_CAPTION_POSITION = "bottom"
MIA_CAPTION_STROKE_COLOR = "#000000"
MIA_CAPTION_STROKE_WIDTH = 4
MIA_CAPTION_BG_COLOR = None
MIA_CAPTION_HIGHLIGHT_COLOR = "#00FF66"  # currently-spoken word: green
MIA_CAPTION_MAX_WIDTH_RATIO = 0.85
MIA_CAPTION_BOTTOM_MARGIN_PX = 180
MIA_CAPTION_BOTTOM_MARGIN_PX_GEN = 180

# Mini-vlog categories & content mix weights
MIA_CONTENT_CATEGORIES = {
    "mini_vlog": {
        "label": "Mini Vlog / Day in the Life",
        "weight": 0.35,
        "topic_hint": "A random day in my life in the city",
    },
    "travel_vlog": {
        "label": "Travel & Exploration",
        "weight": 0.25,
        "topic_hint": "I had two hours to explore a hidden neighborhood",
    },
    "food_cafe": {
        "label": "Café & Food Discoveries",
        "weight": 0.25,
        "topic_hint": "Come with me to find the best hidden café",
    },
    "daily_life": {
        "label": "Micro-Stories & Spontaneous Moments",
        "weight": 0.15,
        "topic_hint": "I went out for one coffee and this happened",
    },
}


def pick_category() -> str:
    categories = list(MIA_CONTENT_CATEGORIES.keys())
    weights = [MIA_CONTENT_CATEGORIES[c]["weight"] for c in categories]
    return random.choices(categories, weights=weights, k=1)[0]
