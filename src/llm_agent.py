import json
import re
import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
)


def fallback_prompt(task_instruction: str):
    """
    Rule-based prompt generation.
    Used when no LLM API is available.
    """
    task_instruction = task_instruction.strip()
    if task_instruction == "":
        task_instruction = "restore only the masked region naturally"

    positive_prompt = (
        f"{task_instruction}. "
        "Edit only the masked region. "
        "Preserve all unmasked regions exactly. "
        "Fill the masked area with contextually matching content only. "
        "Natural restoration, seamless background, coherent lighting, realistic texture, "
        "high quality, visually consistent with the surrounding area. "
        "Do not create new people, faces, portraits, text, logos, or unrelated objects."
    )

    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT
        + ", person, face, portrait, human, extra object, unrelated content, full image regeneration"
    )

    return positive_prompt, negative_prompt



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
    if not OPENROUTER_API_KEY:
        return fallback_prompt(task_instruction)

    system_prompt = """
You are an expert prompt engineer for diffusion-based image inpainting.

Given a user instruction, generate:
1. positive_prompt
2. negative_prompt

Rules:
- Focus on local inpainting only.
- Preserve the original image style and composition.
- Preserve all unmasked regions exactly.
- Fill only the masked region with content that matches nearby context.
- Do NOT generate new people, faces, portraits, or unrelated objects unless explicitly requested.
- Avoid fantasy, painting, cartoon, anime, or unrelated style unless the user explicitly asks for it.
- Make the positive prompt suitable for Stable Diffusion Inpainting.
- Make the negative prompt describe unwanted artifacts and unrelated hallucinations.
- Output valid JSON only.
- JSON keys must be exactly: positive_prompt, negative_prompt.
"""

    user_prompt = f"""
User instruction:
{task_instruction}

Generate suitable prompts for Stable Diffusion Inpainting.
"""

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

        if positive_prompt == "":
            return fallback_prompt(task_instruction)

        if negative_prompt == "":
            negative_prompt = (
                DEFAULT_NEGATIVE_PROMPT
                + ", person, face, portrait, human, extra object, unrelated content, full image regeneration"
            )

        return positive_prompt, negative_prompt

    except Exception as e:
        print("[Warning] LLM prompt generation failed. Using fallback prompt.")
        print(e)
        return fallback_prompt(task_instruction)
