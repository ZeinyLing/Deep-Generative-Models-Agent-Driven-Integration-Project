import json
import re
import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
)


# CLIP text encoder used by Stable Diffusion v1.x supports about 77 tokens.
# Keep prompts short to avoid truncation warnings.
MAX_PROMPT_WORDS = 55
MAX_NEGATIVE_WORDS = 35


def _limit_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def fallback_prompt(task_instruction: str):
    """
    Short rule-based prompt builder.
    This avoids CLIP 77-token truncation.
    """
    task_instruction = (task_instruction or "").strip().lower()

    if "scratch" in task_instruction or "damage" in task_instruction or "repair" in task_instruction:
        positive_prompt = (
            "repair the masked damaged area, match surrounding texture, "
            "natural realistic restoration, preserve unmasked image"
        )
    elif "watermark" in task_instruction:
        positive_prompt = (
            "remove the masked watermark, restore natural background, "
            "match surrounding texture and lighting, preserve unmasked image"
        )
    else:
        positive_prompt = (
            "restore the masked area naturally, match surrounding texture, "
            "preserve unmasked image"
        )

    negative_prompt = (
        "text, watermark, logo, blur, artifacts, distortion, color mismatch, "
        "new object, person, face, portrait"
    )

    return (
        _limit_words(positive_prompt, MAX_PROMPT_WORDS),
        _limit_words(negative_prompt, MAX_NEGATIVE_WORDS),
    )


def extract_json_from_text(text: str):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def generate_prompt_with_llm(task_instruction: str):
    """
    Use OpenRouter if available.
    If the API key/model is invalid, fallback to a short static prompt.
    """
    if not OPENROUTER_API_KEY:
        return fallback_prompt(task_instruction)

    system_prompt = """
Generate short Stable Diffusion inpainting prompts.
Return valid JSON only with keys: positive_prompt, negative_prompt.
Rules:
- positive_prompt under 45 words
- negative_prompt under 25 words
- local inpainting only
- preserve unmasked image
- no new people, faces, text, logos, or unrelated objects
"""

    user_prompt = f"Task: {task_instruction}"

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = extract_json_from_text(content)

        positive_prompt = parsed.get("positive_prompt", "").strip()
        negative_prompt = parsed.get("negative_prompt", "").strip()

        if not positive_prompt:
            return fallback_prompt(task_instruction)

        if not negative_prompt:
            negative_prompt = (
                "text, watermark, logo, blur, artifacts, distortion, "
                "new object, person, face"
            )

        return (
            _limit_words(positive_prompt, MAX_PROMPT_WORDS),
            _limit_words(negative_prompt, MAX_NEGATIVE_WORDS),
        )

    except Exception as e:
        print("[Warning] LLM prompt generation failed. Using short fallback prompt.")
        print(e)
        return fallback_prompt(task_instruction)
