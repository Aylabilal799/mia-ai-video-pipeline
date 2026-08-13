import sys

path = "/root/deepseekyt/agnes-video-generator/core/screenwriter/scenes.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

original = content

anchor_1 = '''def _call_openrouter(system_prompt: str, user_prompt: str) -> str:'''

gemini_block = '''# --- Optional Gemini primary provider (falls back to OpenRouter above) ------
# If GEMINI_API_KEY is set, generate_scene_prompt_for_paragraph() tries
# Gemini FIRST (via Google's OpenAI-compatible endpoint), and only falls
# back to OpenRouter (if configured) if the Gemini call fails for any
# reason (rate limit, error, empty response, etc). If neither is
# configured, Agnes's own built-in LLM is used, same as before.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

if GEMINI_API_KEY:
    _gmasked = GEMINI_API_KEY[:6] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 12 else "***"
    logger.info(
        f"[Screenwriter] Gemini ENABLED as primary provider for scene-prompt generation: "
        f"model={GEMINI_MODEL}, key={_gmasked}"
    )


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        GEMINI_URL,
        headers={
            "Authorization": f"Bearer {GEMINI_API_KEY}",
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


def _call_openrouter(system_prompt: str, user_prompt: str) -> str:'''

if anchor_1 not in content:
    print("ERROR: anchor_1 (_call_openrouter def) not found. Aborting, no changes made.")
    sys.exit(1)

content = content.replace(anchor_1, gemini_block, 1)

old_selection_block = '''        logger.info(f"[Screenwriter] Generating scene prompt for paragraph ({len(text)} chars)...")
        if OPENROUTER_API_KEY:
            logger.info(
                f"[Screenwriter] Provider: OpenRouter (model={OPENROUTER_MODEL})"
            )
            try:
                raw = _call_openrouter(system_prompt, user_prompt)
            except Exception as e:
                # Do NOT silently fall back to Agnes's built-in LLM here --
                # if OpenRouter is configured, a failure must surface as a
                # real error (visible in server.log / Discord) rather than
                # quietly degrading to a different provider than the one
                # that was explicitly configured.
                logger.error(
                    f"[Screenwriter] OpenRouter call FAILED (model={OPENROUTER_MODEL}): {e}"
                )
                raise
            logger.info(f"[Screenwriter] OpenRouter response received ({len(raw)} chars)")
        else:
            logger.info(
                "[Screenwriter] Provider: Agnes built-in LLM "
                "(OPENROUTER_API_KEY not set)"
            )
            raw = self._chat(system_prompt, user_prompt)'''

new_selection_block = '''        logger.info(f"[Screenwriter] Generating scene prompt for paragraph ({len(text)} chars)...")
        raw = None
        if GEMINI_API_KEY:
            logger.info(f"[Screenwriter] Provider: Gemini (model={GEMINI_MODEL})")
            try:
                raw = _call_gemini(system_prompt, user_prompt)
                logger.info(f"[Screenwriter] Gemini response received ({len(raw)} chars)")
            except Exception as e:
                logger.warning(
                    f"[Screenwriter] Gemini call FAILED (model={GEMINI_MODEL}): {e}. "
                    f"Falling back..."
                )
                raw = None

        if raw is None and OPENROUTER_API_KEY:
            logger.info(
                f"[Screenwriter] Provider: OpenRouter (model={OPENROUTER_MODEL})"
            )
            try:
                raw = _call_openrouter(system_prompt, user_prompt)
                logger.info(f"[Screenwriter] OpenRouter response received ({len(raw)} chars)")
            except Exception as e:
                # If Gemini was never configured, OpenRouter was the
                # explicitly chosen provider -- a failure here must surface
                # as a real error rather than quietly degrading further.
                logger.error(
                    f"[Screenwriter] OpenRouter call FAILED (model={OPENROUTER_MODEL}): {e}"
                )
                if not GEMINI_API_KEY:
                    raise
                raw = None

        if raw is None:
            if GEMINI_API_KEY or OPENROUTER_API_KEY:
                raise RuntimeError(
                    "All configured LLM providers (Gemini/OpenRouter) failed "
                    "to generate a scene prompt."
                )
            logger.info(
                "[Screenwriter] Provider: Agnes built-in LLM "
                "(no GEMINI_API_KEY / OPENROUTER_API_KEY set)"
            )
            raw = self._chat(system_prompt, user_prompt)'''

if old_selection_block not in content:
    print("ERROR: old_selection_block not found. Aborting, no changes made.")
    sys.exit(1)

content = content.replace(old_selection_block, new_selection_block, 1)

if content == original:
    print("ERROR: no changes were actually made (unexpected). Aborting write.")
    sys.exit(1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
