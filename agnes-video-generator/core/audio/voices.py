"""core.audio.voices — 音色目录与兼容性

Kokoro 本地静态音色目录（Edge-TTS 已移除，不再有远程音色目录可拉取），
并内置跨语言兼容性矩阵与「文本脚本 → 兼容性」检测，供后端 /api/voices* 接口与
任务创建时的 voice/text 校验复用。

设计背景见 docs/voice_selector_design.md。核心结论：
- 同一文字体系内互通，跨体系基本不通（CJK→en 是唯一例外）。
- 跨体系调用会在 Kokoro 生成阶段失败（不同语言需要不同的 misaki G2P 扩展），
  因此仍需前置校验。
"""

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 项目语言定义
# ═══════════════════════════════════════════════════

# code -> (展示名, 文字体系)
# 文字体系: cjk / latin / cyrillic
PROJECT_LANGUAGES = {
    "zh": {"label": "中文", "script": "cjk"},
    "en": {"label": "English", "script": "latin"},
    "ja": {"label": "日本語", "script": "cjk"},
    "ko": {"label": "한국어", "script": "cjk"},
    "ru": {"label": "Русский", "script": "cyrillic"},
    "de": {"label": "Deutsch", "script": "latin"},
    "fr": {"label": "Français", "script": "latin"},
    "nl": {"label": "Nederlands", "script": "latin"},
    "es": {"label": "Español", "script": "latin"},
    "pt": {"label": "Português", "script": "latin"},
    "it": {"label": "Italiano", "script": "latin"},
    "id": {"label": "Bahasa Indonesia", "script": "latin"},
    "ms": {"label": "Bahasa Melayu", "script": "latin"},
}

# 拉丁体系包含的全部项目语言（彼此完全互通）
_LATIN_LANGS = [c for c, v in PROJECT_LANGUAGES.items() if v["script"] == "latin"]

# ═══════════════════════════════════════════════════
# 预设试听文本（与音色语言严格匹配）
# ═══════════════════════════════════════════════════

VOICE_PREVIEW_TEXTS = {
    "zh": "你好，我是{name}，这是一段音色试听。",
    "en": "Hello, I'm {name}, this is a voice preview sample.",
    "ja": "こんにちは、{name}です。これはボイスプレビューです。",
    "ko": "안녕하세요, 저는 {name}입니다. 이것은 음성 미리보기입니다.",
    "ru": "Здравствуйте, я {name}, это образец голоса.",
    "de": "Hallo, ich bin {name}, dies ist eine Sprachvorschau.",
    "fr": "Bonjour, je suis {name}, ceci est un aperçu vocal.",
    "nl": "Hallo, ik ben {name}, dit is een stemvoorbeeld.",
    "es": "Hola, soy {name}, esta es una muestra de voz.",
    "pt": "Olá, eu sou {name}, esta é uma amostra de voz.",
    "it": "Ciao, sono {name}, questo è un esempio vocale.",
    "id": "Halo, saya {name}, ini adalah sampel suara.",
    "ms": "Helo, saya {name}, ini adalah sampel suara.",
}

# ═══════════════════════════════════════════════════
# 兼容性矩阵（语言级）
# ═══════════════════════════════════════════════════
# 每个语言可读的目标语言集合（实测结论见设计文档 2.2 节）。
# 拉丁体系 9 种语言完全互通，故共享同一集合。

_LATIN_COMPAT = list(_LATIN_LANGS)  # 自身 + 其他 8 种拉丁语言

LANG_COMPAT = {
    "zh": ["zh", "en"],
    "en": list(_LATIN_COMPAT),
    "ja": ["ja", "zh", "en"],
    "ko": ["ko", "zh", "en"],
    "ru": ["ru"],
    "de": list(_LATIN_COMPAT),
    "fr": list(_LATIN_COMPAT),
    "nl": list(_LATIN_COMPAT),
    "es": list(_LATIN_COMPAT),
    "pt": list(_LATIN_COMPAT),
    "it": list(_LATIN_COMPAT),
    "id": list(_LATIN_COMPAT),
    "ms": list(_LATIN_COMPAT),
}


# ═══════════════════════════════════════════════════
# 文本脚本检测（用于任务提交时校验任意文本）
# ═══════════════════════════════════════════════════

# 各文字体系对应的可读 voice 语言集合
_SCRIPT_COMPAT_VOICES = {
    "zh": {"zh", "ja", "ko"},          # 汉字 → 中/日/韩音色
    "ja": {"ja"},                       # 假名 → 仅日语音色
    "ko": {"ko"},                       # 谚文 → 仅韩语音色
    "latin": set(_LATIN_LANGS) | {"zh", "ja", "ko"},  # 拉丁字母 → 全部拉丁 + CJK(均可读英文)
    "ru": {"ru"},                       # 西里尔 → 仅俄文
}


def detect_text_script(text: str) -> str:
    """粗略判断文本的 dominant 文字体系。

    Returns: 'zh' | 'ja' | 'ko' | 'latin' | 'ru' | 'unknown'
    """
    if not text or not text.strip():
        return "unknown"
    # 优先级：谚文 > 假名 > 汉字 > 西里尔 > 拉丁
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "latin"
    return "unknown"


# ═══════════════════════════════════════════════════
# voice id 解析
# ═══════════════════════════════════════════════════

# Kokoro 音色命名约定：<lang><gender>_<name>（如 af_heart, bm_george），
# 单字母语言前缀，与旧 edge_tts 的 "xx-YY-NameNeural" 完全不同的规则。
_KOKORO_LANG_PREFIX = {
    "a": "en", "b": "en",  # American / British English
    "e": "es", "f": "fr", "i": "it", "j": "ja", "p": "pt", "z": "zh",
}
_KOKORO_VOICE_ID_RE = re.compile(r"^([a-z])[fm]_")


def get_voice_lang(voice_id: str):
    """从 voice id 解析项目语言 code。

    优先按 Kokoro 命名约定解析（如 af_heart -> en）；解析不出时，兼容旧的
    edge_tts 风格 id（如 zh-CN-XiaoxiaoNeural -> zh），仅用于历史数据/日志
    里可能还残留的旧 voice 字符串，不代表项目又支持 edge_tts。

    返回 PROJECT_LANGUAGES 中的 code，无法识别时返回 None。
    """
    if not voice_id:
        return None
    m = _KOKORO_VOICE_ID_RE.match(voice_id)
    if m:
        code = _KOKORO_LANG_PREFIX.get(m.group(1))
        if code and code in PROJECT_LANGUAGES:
            return code
    lang_part = voice_id.split("-")[0].lower()
    return lang_part if lang_part in PROJECT_LANGUAGES else None


# ═══════════════════════════════════════════════════
# 兼容性判定
# ═══════════════════════════════════════════════════

def is_voice_compatible(voice_id: str, target_lang: str) -> bool:
    """语言级兼容性：voice 能否朗读 target_lang 语言的内容。"""
    vlang = get_voice_lang(voice_id)
    if vlang is None or target_lang not in PROJECT_LANGUAGES:
        # 未知 voice 或未知目标语言：仅当完全相同时视为兼容
        return vlang == target_lang
    supported = LANG_COMPAT.get(vlang, [vlang])
    return target_lang in supported


def is_voice_compatible_with_text(voice_id: str, text: str) -> bool:
    """文本级兼容性：voice 能否朗读给定文本的 dominant 文字体系。

    用于任务提交时校验（稿件正文已知，创意/诗歌等由 LLM 按页面语言生成）。
    """
    vlang = get_voice_lang(voice_id)
    if vlang is None:
        return True  # 未知音色不阻断
    script = detect_text_script(text)
    allowed = _SCRIPT_COMPAT_VOICES.get(script)
    if allowed is None:
        return True  # unknown 脚本不阻断
    return vlang in allowed


# ═══════════════════════════════════════════════════
# Kokoro 静态音色目录（Edge-TTS 已移除，不再动态拉取音色列表）
# ═══════════════════════════════════════════════════
#
# NOTE: Kokoro-82M 官方语音包目前只覆盖英语（美/英）质量有保障、开箱即用；
# 其它语言（法/日/中/西/印地/意/葡）需要额外的 misaki 语言扩展和各自的
# G2P 依赖，本项目未安装/未验证，因此这里不列出，避免让用户选到一个实际
# 会在生成时报错的音色。之前基于 edge_tts.list_voices() 的 13 语言目录
# 已随 Edge-TTS 一起移除 —— 这是移除 Edge-TTS 的直接代价，请知悉。
# 完整 Kokoro 音色列表见: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

_KOKORO_VOICES = [
    # American English (af_* / am_*) — lang_code "a"
    {"id": "af_heart", "name": "Heart", "local_name": "Heart", "region": "en-US",
     "region_code": "en-US", "gender": "female", "style_tags": ["Warm", "Default"],
     "preview_text": "Hello, I'm Heart, this is a voice preview sample.", "lang": "en"},
    {"id": "af_bella", "name": "Bella", "local_name": "Bella", "region": "en-US",
     "region_code": "en-US", "gender": "female", "style_tags": [],
     "preview_text": "Hello, I'm Bella, this is a voice preview sample.", "lang": "en"},
    {"id": "af_nicole", "name": "Nicole", "local_name": "Nicole", "region": "en-US",
     "region_code": "en-US", "gender": "female", "style_tags": [],
     "preview_text": "Hello, I'm Nicole, this is a voice preview sample.", "lang": "en"},
    {"id": "af_sarah", "name": "Sarah", "local_name": "Sarah", "region": "en-US",
     "region_code": "en-US", "gender": "female", "style_tags": [],
     "preview_text": "Hello, I'm Sarah, this is a voice preview sample.", "lang": "en"},
    {"id": "af_sky", "name": "Sky", "local_name": "Sky", "region": "en-US",
     "region_code": "en-US", "gender": "female", "style_tags": [],
     "preview_text": "Hello, I'm Sky, this is a voice preview sample.", "lang": "en"},
    {"id": "am_adam", "name": "Adam", "local_name": "Adam", "region": "en-US",
     "region_code": "en-US", "gender": "male", "style_tags": [],
     "preview_text": "Hello, I'm Adam, this is a voice preview sample.", "lang": "en"},
    {"id": "am_michael", "name": "Michael", "local_name": "Michael", "region": "en-US",
     "region_code": "en-US", "gender": "male", "style_tags": [],
     "preview_text": "Hello, I'm Michael, this is a voice preview sample.", "lang": "en"},
    # British English (bf_* / bm_*) — lang_code "b"
    {"id": "bf_emma", "name": "Emma", "local_name": "Emma", "region": "en-GB",
     "region_code": "en-GB", "gender": "female", "style_tags": [],
     "preview_text": "Hello, I'm Emma, this is a voice preview sample.", "lang": "en"},
    {"id": "bm_george", "name": "George", "local_name": "George", "region": "en-GB",
     "region_code": "en-GB", "gender": "male", "style_tags": [],
     "preview_text": "Hello, I'm George, this is a voice preview sample.", "lang": "en"},
]

_KOKORO_INDEX = {v["id"]: v for v in _KOKORO_VOICES}


def _build_kokoro_catalog() -> dict:
    return {
        "languages": [
            {"code": "en", "label": PROJECT_LANGUAGES["en"]["label"],
             "count": len(_KOKORO_VOICES), "voices": _KOKORO_VOICES},
        ],
        "compat_hint": LANG_COMPAT,
        "fallback": False,
        "engine": "kokoro",
    }


async def load_voice_catalog(force: bool = False) -> dict:
    """构建 Kokoro 静态音色目录，结果缓存到模块级变量。

    Kokoro 是本地模型，没有类似 edge_tts.list_voices() 的远程目录可拉取，
    所以这里直接返回内置的静态列表（同步逻辑，async 签名只是为了不改动
    调用方的 await 用法）。
    """
    global _VOICE_CATALOG, _VOICE_INDEX
    if _VOICE_CATALOG is not None and not force:
        return _VOICE_CATALOG

    _VOICE_CATALOG = _build_kokoro_catalog()
    _VOICE_INDEX = dict(_KOKORO_INDEX)
    logger.info(f"[Voices] Loaded Kokoro catalog: {len(_KOKORO_VOICES)} voices")
    return _VOICE_CATALOG


def get_voice_catalog() -> dict:
    """同步获取目录（已在服务启动时加载；未加载时直接构建静态目录）。"""
    if _VOICE_CATALOG is None:
        logger.info("[Voices] Catalog not loaded yet; building Kokoro static catalog")
        return _build_kokoro_catalog()
    return _VOICE_CATALOG


def get_voice_by_id(voice_id: str) -> dict | None:
    """按 id 查询单个音色条目。"""
    if _VOICE_INDEX is None:
        get_voice_catalog()
    return (_VOICE_INDEX or _KOKORO_INDEX).get(voice_id)


# 模块级缓存
_VOICE_CATALOG: dict | None = None
_VOICE_INDEX: dict | None = None


def warmup_voice_catalog():
    """在同步上下文（如程序导入时）预加载目录。失败不抛异常。"""
    try:
        asyncio.run(load_voice_catalog())
    except Exception as e:
        logger.warning(f"[Voices] warmup failed ({e}); will use static Kokoro catalog")
