import os
import time
import torch

from diffusers import StableDiffusionInpaintPipeline

from src.config import (
    MODEL_ID,
    DEVICE,
    DTYPE,
    OUTPUT_DIR,
    HF_TOKEN,
)

from src.image_utils import (
    preprocess_input_image,
    preprocess_mask,
    force_same_size,
    create_comparison_image,
)

from src.llm_agent import generate_prompt_with_llm


_pipe = None


def load_pipeline():
    """
    Lazy load Stable Diffusion Inpainting pipeline.
    """
    global _pipe

    if _pipe is not None:
        return _pipe

    print(f"[Info] Loading diffusion model: {MODEL_ID}")
    print(f"[Info] Device: {DEVICE}")

    kwargs = {
        "torch_dtype": DTYPE,
        "safety_checker": None,
    }

    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        **kwargs,
    )

    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("[Info] xFormers not available. Continue without it.")

    _pipe = pipe
    return pipe



def run_inpainting(
    image,
    mask,
    task_instruction,
    steps=40,
    guidance_scale=5.5,
    seed=42,
):
    """
    Main inference function.
    - image, mask, result, and comparison are all unified to the same canvas size.
    """
    if image is None:
        raise ValueError("Please upload an input image.")

    if mask is None:
        raise ValueError("Please provide or draw a mask.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = preprocess_input_image(image)
    mask = preprocess_mask(mask)

    positive_prompt, negative_prompt = generate_prompt_with_llm(task_instruction)

    pipe = load_pipeline()
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    with torch.inference_mode():
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            width=image.width,
            height=image.height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]

    result = force_same_size(result, image)

    timestamp = int(time.time())
    result_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.png")
    comparison_path = os.path.join(OUTPUT_DIR, f"comparison_{timestamp}.png")

    result.save(result_path)

    comparison = create_comparison_image(image, mask, result)
    comparison.save(comparison_path)

    return {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "input_image": image,
        "processed_mask": mask,
        "result_image": result,
        "comparison_image": comparison,
        "result_path": result_path,
        "comparison_path": comparison_path,
    }
