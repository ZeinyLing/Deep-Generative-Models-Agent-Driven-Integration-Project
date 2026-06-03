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
        task_instruction = "restore the masked region naturally"

    positive_prompt = (
        f"{task_instruction}, restore the masked region naturally, "
        f"clean seamless background, natural texture, realistic image restoration, "
        f"coherent lighting, high quality, detailed, visually consistent with the surrounding area"
    )

    negative_prompt = DEFAULT_NEGATIVE_PROMPT

    return positive_prompt, negative_prompt


def extract_json_from_text(text: str):
    """
    Try to extract JSON from LLM response.
    This makes the parser more robust when the model wraps JSON in markdown.
    """
    text = text.strip()

    # Remove ```json ... ```
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def generate_prompt_with_llm(task_instruction: str):
    """
    Use OpenRouter OpenAI-compatible API to generate prompts.
    If the API key is missing or the API call fails, fallback prompt generation is used.
    """
    if not OPENROUTER_API_KEY:
        return fallback_prompt(task_instruction)

    system_prompt = """
You are an expert prompt engineer for diffusion-based image inpainting.

Given a user instruction, generate:
1. positive_prompt
2. negative_prompt

Rules:
- Focus on natural restoration of the masked region.
- Preserve the original image style.
- Avoid changing unmasked regions.
- Avoid fantasy, painting, cartoon, anime, or unrelated style unless the user explicitly asks for it.
- Make the positive prompt suitable for Stable Diffusion Inpainting.
- Make the negative prompt describe unwanted artifacts.
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
                "temperature": 0.3,
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
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        return positive_prompt, negative_prompt

    except Exception as e:
        print("[Warning] LLM prompt generation failed. Using fallback prompt.")
        print(e)
        return fallback_prompt(task_instruction)