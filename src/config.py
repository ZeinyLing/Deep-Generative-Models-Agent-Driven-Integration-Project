import os
import torch
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "outputs"

MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMAGE_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# OpenRouter is optional.
# If OPENROUTER_API_KEY is empty, the app uses a short static fallback prompt.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# The previous default "opencode/big-pickle" may return 400 on OpenRouter.
# Use a valid OpenRouter model if you want LLM prompt generation.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

HF_TOKEN = os.getenv("HF_TOKEN", "")

DEFAULT_NEGATIVE_PROMPT = (
    "watermark, text, logo, blur, artifacts, distorted, color mismatch, "
    "low quality, noisy, rough edges"
)
