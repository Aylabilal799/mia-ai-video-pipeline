"""core.screenwriter.scenes — 分镜/段落场景 prompt/诗词场景方法（Batch 4 拆分）"""
import logging
import os
import re
import time
from typing import List, Optional

import requests

from core.api.agnes_chat import strip_code_fence

logger = logging.getLogger(__name__)

# On 429, try up to this many DIFFERENT keys from the pool before giving up
# on the provider entirely. Each attempt uses a fresh (not-on-cooldown) key
# if one is available -- no need to sleep between attempts in that case,
# since we're not hammering the same rate-limited key twice. Only sleeps if
# every key in the pool is currently on cooldown.
_MAX_KEY_ATTEMPTS = 5
_ALL_KEYS_EXHAUSTED_WAIT = 5.0  # seconds, before giving the pool one more pass


def _call_with_key_rotation(call_fn, provider_name: str, key_pool: "KeyPool", *args, **kwargs):
    """Call `call_fn(*args, api_key=<key>, **kwargs)`, rotating through
    key_pool on 429s. Raises the last error if every attempt is exhausted.
    """
    last_exc = None
    for attempt in range(_MAX_KEY_ATTEMPTS):
        key = key_pool.get()
        if key is None:
            # Every key on cooldown -- wait once for the shortest cooldown to
            # clear rather than busy-looping.
            logger.warning(
                f"[Screenwriter] {provider_name}: all keys in pool on cooldown, "
                f"waiting {_ALL_KEYS_EXHAUSTED_WAIT:.0f}s..."
            )
            time.sleep(_ALL_KEYS_EXHAUSTED_WAIT)
            key = key_pool.get()
            if key is None:
                raise last_exc or RuntimeError(f"{provider_name}: no keys available in pool")

        try:
            return call_fn(*args, api_key=key, **kwargs)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_exc = e
            if status != 429:
                raise
            key_pool.mark_rate_limited(key)
            # loop again -- pool.get() will hand back a different key if one
            # is available
    raise last_exc


class KeyPool:
    """Rotates across multiple API keys for one provider. When a key hits a
    429, it's put on cooldown (default 65s -- just past a typical per-minute
    quota window) and the pool moves to the next available key. If every key
    is currently on cooldown, `get()` returns None (caller should fall
    through to the next provider) rather than retrying a key that will
    almost certainly still be rate-limited.

    Configure via a comma-separated env var, e.g.:
        GEMINI_API_KEY=key1,key2,key3
    A single bare key (no commas) works exactly as before -- this is
    backwards compatible with existing single-key .env files.
    """

    def __init__(self, keys: List[str], cooldown_seconds: float = 65.0):
        self._keys = [k.strip() for k in keys if k.strip()]
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until = {k: 0.0 for k in self._keys}
        self._next_idx = 0

    def __bool__(self):
        return bool(self._keys)

    def __len__(self):
        return len(self._keys)

    def _mask(self, key: str) -> str:
        return key[:6] + "..." + key[-4:] if len(key) > 12 else "***"

    def get(self) -> Optional[str]:
        """Return the next available (not-on-cooldown) key, round-robin."""
        now = time.monotonic()
        for _ in range(len(self._keys)):
            key = self._keys[self._next_idx]
            self._next_idx = (self._next_idx + 1) % len(self._keys)
            if now >= self._cooldown_until[key]:
                return key
        return None

    def mark_rate_limited(self, key: str) -> None:
        self._cooldown_until[key] = time.monotonic() + self._cooldown_seconds
        available = sum(1 for k in self._keys if time.monotonic() >= self._cooldown_until[k])
        logger.warning(
            f"[Screenwriter] Key {self._mask(key)} rate-limited, cooling down "
            f"{self._cooldown_seconds:.0f}s ({available}/{len(self._keys)} keys "
            f"still available in pool)"
        )


def _load_key_pool(env_var: str) -> KeyPool:
    raw = os.environ.get(env_var, "")
    keys = [k for k in raw.split(",") if k.strip()]
    return KeyPool(keys)


# --- Optional OpenRouter override for scene-prompt writing ------------------
# If OPENROUTER_API_KEY is set, generate_scene_prompt_for_paragraph() uses
# OpenRouter (calling Claude, or whatever OPENROUTER_MODEL is set to) instead
# of Agnes's own built-in LLM for this one step. Everything else (video
# generation, TTS, captions, concatenation) still runs through Agnes as
# normal -- only the visual-description-writing step is swapped out, since
# that's the step whose quality/consistency actually matters most for how
# the final video looks.
#
# Supports multiple comma-separated keys for rotation on rate-limit:
#   OPENROUTER_API_KEY=key1,key2,key3
OPENROUTER_KEYS = _load_key_pool("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Log this once at import time (i.e. once per Agnes server start) so it's
# immediately obvious from server.log which provider is actually active for
# scene-prompt generation, without having to wait for the first paragraph.
if OPENROUTER_KEYS:
    logger.info(
        f"[Screenwriter] OpenRouter ENABLED for scene-prompt generation: "
        f"model={OPENROUTER_MODEL}, {len(OPENROUTER_KEYS)} key(s) in pool"
    )
else:
    logger.info(
        "[Screenwriter] OpenRouter NOT configured (OPENROUTER_API_KEY not set) "
        "-- using Agnes's built-in LLM for scene-prompt generation"
    )


# --- Optional Gemini primary provider (falls back to OpenRouter above) ------
# If GEMINI_API_KEY is set, generate_scene_prompt_for_paragraph() tries
# Gemini FIRST (via Google's OpenAI-compatible endpoint), and only falls
# back to OpenRouter (if configured) if the Gemini call fails for any
# reason (rate limit, error, empty response, etc). If neither is
# configured, Agnes's own built-in LLM is used, same as before.
#
# Supports multiple comma-separated keys for rotation on rate-limit:
#   GEMINI_API_KEY=key1,key2,key3
GEMINI_KEYS = _load_key_pool("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

if GEMINI_KEYS:
    logger.info(
        f"[Screenwriter] Gemini ENABLED as primary provider for scene-prompt generation: "
        f"model={GEMINI_MODEL}, {len(GEMINI_KEYS)} key(s) in pool"
    )


# --- Optional FreeLLMAPI last-resort fallback --------------------------------
# If FREELLMAPI_API_KEY is set, generate_scene_prompt_for_paragraph() tries
# FreeLLMAPI (a self-hosted OpenAI-compatible router that aggregates many
# providers' free tiers, including Gemini and OpenRouter) as a LAST resort,
# only after Gemini and OpenRouter have both already failed. This exists
# purely to survive the case where both of the above are simultaneously
# rate-limited or erroring -- FreeLLMAPI's own internal fallback chain gives
# the request many more providers to try before the whole paragraph fails.
#
# FREELLMAPI_BASE_URL defaults to a local instance on the same host
# (http://localhost:3001). Only the chat completions path is needed here.
# Model defaults to "auto" so FreeLLMAPI's router picks whatever's healthy.
FREELLMAPI_KEYS = _load_key_pool("FREELLMAPI_API_KEY")
FREELLMAPI_MODEL = os.environ.get("FREELLMAPI_MODEL", "auto")
FREELLMAPI_BASE_URL = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
FREELLMAPI_URL = f"{FREELLMAPI_BASE_URL.rstrip('/')}/chat/completions"

if FREELLMAPI_KEYS:
    logger.info(
        f"[Screenwriter] FreeLLMAPI ENABLED as last-resort fallback for scene-prompt "
        f"generation: model={FREELLMAPI_MODEL}, url={FREELLMAPI_URL}, "
        f"{len(FREELLMAPI_KEYS)} key(s) in pool"
    )


def _call_gemini(system_prompt: str, user_prompt: str, api_key: str) -> str:
    resp = requests.post(
        GEMINI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GEMINI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 2000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError(f"Gemini returned an empty response: {data}")
    return content


def _call_openrouter(system_prompt: str, user_prompt: str, api_key: str) -> str:
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            # Scene descriptions are ~80-150 words; capping this avoids
            # OpenRouter reserving the model's full max output (often
            # 64k+ tokens) against your account balance for every request,
            # which fails with a 402 on low-credit accounts even though the
            # actual usage is tiny.
            "max_tokens": 2000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_freellmapi(system_prompt: str, user_prompt: str, api_key: str) -> str:
    resp = requests.post(
        FREELLMAPI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": FREELLMAPI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 2000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError(f"FreeLLMAPI returned an empty response: {data}")
    return content


class ScreenwriterScenesMixin:
    """分镜设计/段落场景 prompt/诗词场景拆分方法（Batch 4 拆分）。"""

    def design_shots_for_scene(self, scene_text: str, style: str, max_shots: int = 5) -> list:
        system_prompt = self._prompt(
            zh_text="""\
你是一位专业的分镜师。为单个场景设计镜头。

[输出格式] 返回一个 JSON 对象：
{
  "shots": [
    {
      "visual_desc": "镜头的整体视觉描述",
      "variation_type": "large|medium|small",
      "ff_desc": "首帧——静态快照描述",
      "lf_desc": "末帧——静态快照描述",
      "motion_desc": "帧间运动。对话格式：<角色>说：\\"文本\\"",
      "audio_desc": "[音效] 描述"
    }
  ]
}

规则：
- 第一个镜头必须建立场景环境。
- 最后一个镜头应自然地结束场景。
- variation_type："large"（大幅场景变化）、"medium"（新元素出现）、"small"（微小运动）
- 首帧/末帧描述是静态图像——不使用运动词汇。
- 运动描述包含所有动作和对话。
- 包含丰富的视觉细节用于图像生成（光线、色彩、构图）。
- 使用与输入场景相同的语言输出。

构图用词指导：
- 戏剧性/紧张场景："具有强烈对角线的引人注目构图"、\
"紧凑取景增强幽闭感"、"低角度强调权力关系"、\
"推镜营造张力升级"。
- 情感节拍用电影化速写："更宽广的取景创造情感距离"、\
"浅景深将角色孤立在情感中"、\
"缓慢的轨道后拉揭示时刻的分量"。
- 动作/兴奋场景："充满动能的动态构图"、\
"快速摇镜暗示混乱"、"爆发式运动定格在画面中"。
- 浪漫/亲密内容："特写中的柔和温暖亲密感"、\
"沐浴在柔和金色光线中"、"中景取景中温柔相依"。
- 使用专业的电影摄影术语和视觉构图语言表达所有叙事内容。

使用与输入场景相同的语言输出。
""",
            en_text="""\
You are a professional storyboard artist. Design shots for a single scene.

[Output Format] Return a JSON object:
{
  "shots": [
    {
      "visual_desc": "Overall visual description of the shot",
      "variation_type": "large|medium|small",
      "ff_desc": "First frame — static snapshot description",
      "lf_desc": "Last frame — static snapshot description",
      "motion_desc": "Motion between frames. Include dialogue as: <Char> says: \\"text\\"",
      "audio_desc": "[Sound Effect] description"
    }
  ]
}

Rules:
- First shot must establish the scene environment.
- Last shot should end the scene naturally.
- variation_type: "large" (big scene change), "medium" (new element appears), "small" (minor movement)
- First/last frame descriptions are STATIC images — no motion words.
- Motion description includes all movement AND dialogue.
- Include rich visual details for image generation (lighting, colors, composition).
- Output in the SAME LANGUAGE as the input scene.

VOCABULARY GUIDANCE FOR COMPOSITION:
- For dramatic/tension-filled scenes: "striking composition with strong diagonals", \
"tight framing heightens claustrophobic feel", "low angle emphasizes power dynamic", \
"camera push creates mounting intensity".
- Use cinematic shorthand for emotional beats: "wider shot creates emotional \
distance", "shallow focus isolates the character emotionally", \
"slow dolly pull reveals the gravity of the moment".
- Frame action/excitement as: "dynamic composition with kinetic energy", \
"rapid whip-pan suggests chaos", "explosive movement captured in frame".
- Frame romantic/intimate content as: "soft, warm intimacy in close-up", \
"bathing in gentle golden light", "tender proximity framed in medium shot".
- Express ALL narrative content through professional cinematography terminology \
and visual composition language.

Output in the SAME LANGUAGE as the input scene.
""",
        )
        user_prompt = f"""\
<scene>
{scene_text}
</scene>

<style>{style}</style>
<max_shots>{max_shots}</max_shots>
"""
        logger.info(f"[Screenwriter] Designing shots for scene...")
        result = self._chat_json(system_prompt, user_prompt)
        shots = result.get("shots", [])
        logger.info(f"[Screenwriter] Designed {len(shots)} shots")
        return shots

    def generate_scene_prompt_for_paragraph(self, text: str, style: str = "") -> str:
        """为稿件段落生成视频场景 prompt（语言跟随输入段落）。

        基于段落语义生成适合 AI 视频生成的视觉描述，
        原文将直接作为旁白文本 + 字幕内容（D2 决策）。

        Args:
            text: 段落文本
            style: 风格描述（可选）

        Returns:
            视频 prompt 字符串（语言与输入一致）
        """
        system_prompt = self._prompt(
            zh_text="""\
你是一位专业的视频导演和视觉提示词工程师。给定一段\
将作为旁白朗读的文本，请生成一个详细的视觉描述\
用于 AI 视频生成。

规则：
- 使用与输入段落相同的语言编写详细的视觉描述，\
80-150 词。
- 聚焦于：环境、光线、色彩、镜头运动、氛围、情绪。
- 包含电影感细节：镜头类型、景深、调色、\
天气、时间。
- 不要在描述中包含任何文字叠加、标题或字幕。
- 不要描述旁白本身——描述观众看到的内容。
- 视觉应补充和增强文本的含义。
- 描述动作和动态，而非静态图像。
- 时间和光线必须严格匹配段落本身的内容，而非默认使用某种情绪化的光线。\
如果段落明确说明或清楚暗示了具体时间（早晨、中午、傍晚、夜晚）或场景\
（室内、室外、具体地点），视觉描述必须如实反映。只有当文本本身将场景\
设定在夜晚或黑暗环境中时，才能描绘夜晚/黑暗——不要仅仅因为情绪紧张或\
戏剧化就默认使用黑暗画面。即使是白天发生的紧张时刻，也应保持日光照明\
（可用强烈阴影、高对比度、刺眼强光来表现紧张感），而不要改写成夜晚。
- 如果段落完全没有提及具体时间（这在第一人称自白/叙事类文本中很常见），\
应默认使用明亮、清晰、中性的正午式日光（阳光高照，光线均匀开阔）或正常\
的室内光线。不要默认使用黄金时段光线、"傍晚"、黄昏或"光线渐暗"——这些\
同样是未经文本支持的情绪化选择，并不比直接使用黑暗画面更"安全"或更中性。\
不要仅仅因为故事内容严肃、私密或带有忏悔性质，就默认使用傍晚、黄昏或\
夜晚的光线——一段以自白形式讲述的个人经历，本身并不能证明事件发生在\
夜晚或渐暗的光线中。只有当文本给出了具体理由时（明确提到的时间、\
"天黑之后"、卧室睡前场景等），才使用傍晚/黄昏/夜晚光线。
- 不要让视觉基调随着故事情节愈发沉重或情绪化，而在连续段落之间逐渐\
变暗变冷。每个段落的光线只应根据该段落自身的文本来判断——绝不应该\
因为"故事情绪越来越沉重，所以光线也应随之变暗"而调整。如果连续两个\
段落都没有提及具体时间/场景，它们应该使用相同的明亮/中性默认光线，\
而不是从明亮逐渐过渡到昏暗。
- 不要在场景中途凭空编造光线或色调的变化（例如"光线突然变暗"、\
"暖色调转为冷蓝色"），除非段落文本明确描述了会导致这种变化的事件\
（灯被关掉、暴风雨来袭等）。情绪变得紧张或戏剧化本身不构成改变光线\
或色调的理由——应保持场景既有的光线和色调基调贯穿始终，转而通过镜头\
运动、取景和人物细节来传达紧张感。
- 从一个段落的视觉描述到下一个段落，环境和光线应有所变化（除非故事\
明确停留在同一连续地点/时间），避免连续场景看起来完全相同。

用词指导：
- 使用电影化语言传达情感基调："充满张力的氛围"、\
"戏剧性光线"、"亲密取景"、"富有诗意的镜头运动"。
- 对于确实设定在夜晚或黑暗环境中的紧张/激烈段落，依靠视觉隐喻和氛围\
描述："随着张力升级阴影加深"、"不安的镜头运动\
映射内心动荡"、"光与影的强烈对比"。对于设定在白天的紧张/激烈段落，\
改用日光对应的表达方式："正午强光投下锐利阴影"、"过曝的惨白光线\
增添不安感"、"热浪扭曲了地平线"。这两种情况都是描述现有光线/场景\
的呈现方式，而不是允许凭空添加文本中没有的光线变化。
- 对于情感共鸣、悲伤或与哀悼相关的时刻，不要仅仅因为情绪悲伤就默认\
使用冷蓝色调/去饱和调色作为惯用手法，除非文本本身描述了寒冷、蓝光\
或灰暗的场景。优先使用中性、自然的色调（或场景本身所需要的任何色调），\
通过节奏、取景和静止感来传达情绪。
- "轻柔的镜头推进捕捉到温柔"、"柔焦赋予梦幻质感"这类表达适用于\
设定在自然温暖/柔和光线中的情感共鸣时刻。
- 通过镜头所见来表达叙事冲击力——让视觉\
构图承载情感分量。

只输出视觉提示词文本，不要 JSON，不要解释。
""",
            en_text="""\
You are a professional video director and visual prompt engineer. Given a \
paragraph of text that will be narrated as voiceover, generate a \
detailed VISUAL DESCRIPTION for AI video generation.

Rules:
- Write a detailed VISUAL DESCRIPTION in the SAME LANGUAGE as the input paragraph, \
80-150 words.
- Focus on: environment, lighting, colors, camera movement, atmosphere, mood.
- Include cinematic details: lens type, depth of field, color grading, \
weather, time of day.
- Do NOT include any text overlays, titles, or subtitles in the description.
- Do NOT describe the narration itself — describe what the VIEWER SEES.
- The visual should complement and enhance the meaning of the text.
- Describe MOTION and ACTION, not a static image.
- TIME OF DAY / LIGHTING MUST MATCH THE PARAGRAPH'S OWN CONTENT, not a default
mood. If the paragraph states or clearly implies a specific time (morning, \
midday, sunset, night) or setting (indoors, outdoors, a specific location), \
the visual MUST reflect that literally. Only depict night/darkness when the \
text itself places the scene at night or in a dark location -- do not default \
to darkness just because the tone is tense or dramatic. A tense moment in \
broad daylight should still be lit like daylight (harsh shadows, high \
contrast, glare) rather than being rewritten as nighttime.
- IF THE PARAGRAPH DOES NOT MENTION A TIME OF DAY AT ALL (very common in \
first-person confession/storytelling narration), DEFAULT TO BRIGHT, CLEAR, \
NEUTRAL MIDDAY-STYLE DAYLIGHT (sun high, even and open lighting) OR NORMAL \
INDOOR ROOM LIGHTING. Do NOT default to golden hour, "late afternoon", dusk, \
evening, or "light fading" -- these are just as much an unearned mood choice \
as literal darkness, and are NOT a safe/neutral default. Do NOT default to \
evening, dusk, or night just because the story is a serious, personal, or \
confessional account -- a personal confession being narrated is not, by \
itself, evidence the events happened at night or in fading light. Only use \
evening/dusk/night lighting when the text gives an actual reason to (a \
stated time, "after dark", a bedroom at bedtime, etc).
- DO NOT LET THE VISUAL MOOD DRIFT DARKER OR DIMMER ACROSS SUCCESSIVE \
PARAGRAPHS as the story becomes more emotional or serious. Each paragraph's \
lighting is judged ONLY against what THAT paragraph's own text says -- never \
against "the story is getting heavier now, so the light should dim to match." \
If two consecutive paragraphs both lack a stated time/setting, they should \
both get the same bright/neutral default, not a progression from bright to \
dim.
- DO NOT INVENT a lighting or color-grade CHANGE partway through a scene \
(e.g. "the light suddenly darkens", "warm tones give way to cool blue") \
unless the paragraph's text explicitly describes something happening that \
would cause that change (a light being switched off, a storm rolling in, \
etc). A moment becoming emotionally tense or dramatic is NOT by itself a \
valid reason to shift the lighting or color grade -- keep the scene's \
established lighting and color palette consistent throughout, and convey \
the tension instead through camera movement, framing, and character detail.
- Vary the environment and lighting from one paragraph's visual to the next \
(unless the story explicitly stays in one continuous place/time) so \
consecutive scenes don't all end up looking the same.

VOCABULARY GUIDANCE:
- Use cinematic language to convey emotional tone: "charged atmosphere", \
"dramatic lighting", "intimate framing", "lyrical camera movement".
- For intense or tense segments that ARE set at night or in darkness, rely on \
visual metaphor and atmospheric description: "shadows deepen as tension \
mounts", "restless camera work mirrors inner turmoil", "stark contrast \
between light and shadow". For intense or tense segments set in daylight, \
reach for daylight equivalents instead: "harsh midday sun casts sharp \
shadows", "washed-out overexposed light adds unease", "heat haze distorts \
the horizon". In both cases, this is about HOW the existing light/setting is \
described, not a license to add a lighting change that isn't in the text.
- For emotionally resonant, sad, or grief-related moments, do NOT default to \
cool blue / desaturated color grading as a shorthand for sadness unless the \
text describes a cold, blue-lit, or grey setting. Prefer neutral, natural \
tones (or whatever the actual described setting calls for) and convey the \
emotion through pacing, framing, and stillness instead of a blue tint.
- "gentle camera push captures the tenderness", "soft focus lends a \
dreamlike quality" remain good tools for emotionally resonant moments set in \
naturally warm/soft light.
- Express the narrative impact through what the CAMERA sees — let visual \
composition carry the emotional weight.


Output ONLY the visual prompt text, no JSON, no explanation.
""",
        )
        style_block = f"\n<style>{style}</style>\n" if style else ""
        user_prompt = f"""\
<paragraph>
{text}
</paragraph>
{style_block}
{self._prompt(
    zh_text="请为此段落生成一个详细的视觉提示词。",
    en_text="Generate a detailed visual prompt for this paragraph."
)}
"""
        logger.info(f"[Screenwriter] Generating scene prompt for paragraph ({len(text)} chars)...")
        raw = None
        if GEMINI_KEYS:
            logger.info(f"[Screenwriter] Provider: Gemini (model={GEMINI_MODEL}, {len(GEMINI_KEYS)} key(s))")
            try:
                raw = _call_with_key_rotation(
                    _call_gemini, "Gemini", GEMINI_KEYS, system_prompt, user_prompt
                )
                logger.info(f"[Screenwriter] Gemini response received ({len(raw)} chars)")
            except Exception as e:
                logger.warning(
                    f"[Screenwriter] Gemini call FAILED (model={GEMINI_MODEL}): {e}. "
                    f"Falling back..."
                )
                raw = None

        if raw is None and OPENROUTER_KEYS:
            logger.info(
                f"[Screenwriter] Provider: OpenRouter (model={OPENROUTER_MODEL}, {len(OPENROUTER_KEYS)} key(s))"
            )
            try:
                raw = _call_with_key_rotation(
                    _call_openrouter, "OpenRouter", OPENROUTER_KEYS, system_prompt, user_prompt
                )
                logger.info(f"[Screenwriter] OpenRouter response received ({len(raw)} chars)")
            except Exception as e:
                logger.warning(
                    f"[Screenwriter] OpenRouter call FAILED (model={OPENROUTER_MODEL}): {e}. "
                    f"Falling back..."
                )
                raw = None

        if raw is None and FREELLMAPI_KEYS:
            logger.info(
                f"[Screenwriter] Provider: FreeLLMAPI (model={FREELLMAPI_MODEL}, "
                f"{len(FREELLMAPI_KEYS)} key(s))"
            )
            try:
                raw = _call_with_key_rotation(
                    _call_freellmapi, "FreeLLMAPI", FREELLMAPI_KEYS, system_prompt, user_prompt
                )
                logger.info(f"[Screenwriter] FreeLLMAPI response received ({len(raw)} chars)")
            except Exception as e:
                # FreeLLMAPI was the last configured provider in the chain --
                # a failure here must surface as a real error rather than
                # quietly degrading further.
                logger.error(
                    f"[Screenwriter] FreeLLMAPI call FAILED (model={FREELLMAPI_MODEL}): {e}"
                )
                raw = None

        if raw is None:
            if GEMINI_KEYS or OPENROUTER_KEYS or FREELLMAPI_KEYS:
                raise RuntimeError(
                    "All configured LLM providers (Gemini/OpenRouter/FreeLLMAPI) "
                    "failed to generate a scene prompt."
                )
            logger.info(
                "[Screenwriter] Provider: Agnes built-in LLM "
                "(no GEMINI_API_KEY / OPENROUTER_API_KEY / FREELLMAPI_API_KEY set)"
            )
            raw = self._chat(system_prompt, user_prompt)
        prompt = strip_code_fence(raw)
        logger.info(f"[Screenwriter] Scene prompt: {prompt[:100]}...")
        return prompt

    def generate_poetry_scenes(
        self,
        poem_text: str,
        scene_count: int = 0,
        scene_durations: Optional[List[int]] = None,
        total_duration: int = 30,
        style: str = "",
    ) -> List[dict]:
        """LLM 拆分整首诗词为若干场景（朗诵文案 + 视频 prompt），以行格式返回。

        与「用户直接粘贴的分镜描述」保持同一格式：``原诗句 | 画面描述``。
        每行一个场景，「|」左为对应原诗片段（narration），右为视频画面描述。
        这样内部 LLM 生成与外部 LLM（用户拿同样提示词生成后贴回）输出完全一致。

        Args:
            poem_text: 整首诗词原文。
            scene_count: 期望分镜数；0 表示自动（prompt 来源模式）。
            scene_durations: 各场景时长（秒）列表；非空时提示词写出每场景时长+合计，
                与可复制提示词、实际视频生成时长完全一致。
            total_duration: 目标总时长（秒），用于 prompt/auto 模式把握节奏。
            style: 视觉风格（与创意视频一致），注入每个场景画面描述。

        Returns:
            场景列表，每个元素为 {"narration": str, "scene_prompt": str}。
            朗诵文案严格保留原诗文字，不改写不翻译。
        """
        system_prompt, user_prompt = self._poetry_scene_prompts(
            poem_text, scene_count, scene_durations or [], total_duration, style)
        logger.info(
            f"[Screenwriter] Splitting poem into scenes (count={scene_count or 'auto'}, "
            f"durations={scene_durations or 'auto'})..."
        )
        raw = self._chat(system_prompt, user_prompt)
        scenes = self._parse_poetry_scene_lines(raw)
        logger.info(f"[Screenwriter] Poetry scenes: {len(scenes)}")
        return scenes

    def _poetry_scene_prompts(
        self,
        poem_text: str,
        scene_count: int,
        scene_durations: List[int],
        total_duration: int,
        style: str,
    ) -> tuple:
        """返回（system_prompt, user_prompt）。

        内部 LLM 与外部 LLM（用户拿同样提示词生成后贴回）共用此提示词，
        UI 端点 ``build_poetry_scene_prompt`` 也复用同一份，保证格式一致。
        时长表达：有每场景时长时写出「各场景时长：d1,d2…秒（合计T秒）」，
        否则（auto/prompt 模式）写「目标总时长：T秒」。
        """
        count_hint = (
            f"{scene_count} 个" if scene_count and scene_count > 0
            else "由你依据诗意自行决定（通常 2-6 个）"
        )
        if scene_durations:
            dur_list = "、".join(f"{d}秒" for d in scene_durations)
            total = sum(scene_durations)
            dur_hint = f"各场景时长：{dur_list}（合计 {total} 秒）"
        else:
            dur_hint = f"目标总时长：{total_duration} 秒（用于把握节奏，不要求每句等长）"
        style_hint = style.strip() if style and style.strip() else "未指定，请采用通用电影质感写实风格"
        system_prompt = self._prompt(
            zh_text="""你是一位诗意的视觉艺术家兼诗词分镜导演，专精将中国古典诗词转化为视频分镜。
请将整首诗拆分为若干场景，每个场景一行，严格使用如下格式：

原诗句 | 画面描述

规则：
- 「|」左侧为该场景对应的原诗片段（朗诵文案），严格保留原诗文字，不要改写、翻译或意译；
- 「|」右侧为该场景的视频画面描述（只描述视觉，15-30 字，不描述声音或文字）；
- 按诗意的自然段落或意象切分场景，使每段画面连贯且独立；
- 每行一个场景，不要编号、不要解释、不要输出 JSON、不要使用代码块围栏。
示例：
春眠不觉晓，处处闻啼鸟。 | 春日清晨薄雾，枝头鸟鸣
夜来风雨声，花落知多少。 | 夜雨敲窗，落花满地青石""",
            en_text="""You are a poetic visual artist and poetry storyboard director specializing in transforming classical poetry into video scenes.
Split the whole poem into scenes, ONE scene per line, strict format:

original verse | visual description

Rules:
- Left of "|" is the original poem line(s) verbatim (narration); do NOT rewrite/translate/paraphrase.
- Right of "|" is the visual description (15-30 words, visuals only, no audio/text).
- Split by stanzas or imagery; coherent, self-contained scenes.
- One scene per line; no numbering, no explanation, no JSON, no code fences.
Example:
Spring sleeps, unaware of dawn... | Misty spring morning, birds singing
Night rain, wind sounds... | Rain on window, petals on stone""",
        )
        user_prompt = f"""<poem>
{poem_text}
</poem>

<requirements>
- 场景数量：{count_hint}
- {dur_hint}
- 视觉风格：{style_hint}（请将该风格融入每个场景的画面描述）
</requirements>
"""
        return system_prompt, user_prompt

    def _parse_poetry_scene_lines(self, raw: str) -> List[dict]:
        """把 LLM 返回的行格式文本解析为场景列表。

        每行 ``原诗句 | 画面描述``；无 ``|`` 的行视为纯画面描述（诗句留空）。
        自动去除编号/标签前缀（如「1. 」「场景1：」），对外部 LLM 的输出更鲁棒。
        """
        if not raw:
            return []
        raw = strip_code_fence(raw)
        scenes: List[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # 跳过纯场景标签行（如「场景 1（00:00 - 00:10）」），避免被误判为分镜
            if re.match(r"^(场景|Scene)\s*\d+", line, re.IGNORECASE):
                continue
            # 去掉可能的编号/标签前缀
            line = re.sub(r"^[\d]+\s*[\.、)]\s*", "", line)
            line = re.sub(r"^(场景|Scene)\s*\d*\s*[:：]\s*", "", line, flags=re.IGNORECASE)
            if "|" in line:
                verse, _, prompt = line.partition("|")
                verse = verse.strip()
                prompt = prompt.strip()
            else:
                # 纯画面描述（无诗句）：诗句留空，由调用方处理
                verse, prompt = "", line
            if not prompt:
                continue
            scenes.append({"narration": verse, "scene_prompt": prompt})
        return scenes
