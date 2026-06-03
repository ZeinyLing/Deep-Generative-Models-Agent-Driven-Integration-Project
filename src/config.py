import os
import torch
from dotenv import load_dotenv

# =========================================================
# Load .env
# =========================================================
load_dotenv()

# =========================================================
# Paths
# =========================================================
OUTPUT_DIR = "outputs"

# =========================================================
# Diffusion model config
# =========================================================
MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMAGE_SIZE = 256

# =========================================================
# Runtime config
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =========================================================
# OpenRouter LLM config
# =========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "opencode/big-pickle")

# =========================================================
# Hugging Face config
# Usually optional for runwayml/stable-diffusion-inpainting
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN", "")

# =========================================================
# Default negative prompt
# =========================================================
DEFAULT_NEGATIVE_PROMPT = (
    "watermark, text, words, logo, blur, artifacts, distorted shapes, "
    "color mismatch, noisy pixels, low quality, patchy texture, rough edges, "
    "unrealistic texture, duplicated objects, deformed background"
)