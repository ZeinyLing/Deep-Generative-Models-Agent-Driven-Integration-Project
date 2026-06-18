import numpy as np
from PIL import Image, ImageFilter

from src.config import IMAGE_SIZE


def preprocess_input_image(image: Image.Image) -> Image.Image:
    """
    Convert uploaded image to RGB and resize it to fixed resolution.
    """
    if image is None:
        raise ValueError("No input image provided.")

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image



def preprocess_mask(mask: Image.Image, blur_radius: float = 0.0) -> Image.Image:
    """
    Convert mask image to binary grayscale mask.

    White region = area to restore.
    Black region = area to preserve.
    """
    if mask is None:
        raise ValueError("No mask image provided.")

    mask = mask.convert("L")
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))

    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np).convert("L")

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask



def _safe_rgba_to_mask(layer: Image.Image) -> np.ndarray:
    """
    Convert one editor layer to a binary mask robustly.

    Important fix:
    - Do NOT use RGB alone directly because some Gradio versions create a full-size
      white canvas with zero alpha. Using RGB threshold on that turns the entire image
      into a mask.
    - Prefer alpha channel.
    - Only fallback to RGB when alpha carries almost no signal.
    """
    layer_rgba = layer.convert("RGBA")
    layer_np = np.array(layer_rgba)

    alpha = layer_np[:, :, 3]
    rgb = layer_np[:, :, :3]

    # Primary path: use alpha only.
    mask_alpha = (alpha > 20).astype(np.uint8) * 255

    # If alpha is empty or nearly empty, fallback carefully.
    if mask_alpha.sum() == 0:
        rgb_sum = rgb.sum(axis=2)
        # Only count clearly drawn pixels. This is safer than the previous generic rule.
        mask_rgb = (rgb_sum > 500).astype(np.uint8) * 255
        return mask_rgb

    return mask_alpha



def extract_mask_from_editor(editor_value):
    """
    Extract hand-drawn mask from Gradio ImageEditor output.

    The returned mask is binary:
    - white = inpaint area
    - black = keep area
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)
    layers = editor_value.get("layers", [])

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    if not layers:
        raise ValueError("Please draw a mask on the image.")

    bg_w, bg_h = background.size
    final_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue

        layer = layer.resize((bg_w, bg_h))
        mask_np = _safe_rgba_to_mask(layer)
        final_mask = np.maximum(final_mask, mask_np)

    if final_mask.sum() == 0:
        raise ValueError("No valid mask was detected. Please draw again.")

    # Safety check: if most of the image becomes masked, warn the user.
    mask_ratio = float((final_mask > 0).mean())
    if mask_ratio > 0.60:
        raise ValueError(
            f"The detected mask covers {mask_ratio:.1%} of the image, which is too large. "
            "This usually means the editor layer was parsed incorrectly. "
            "Please clear and redraw a smaller mask."
        )

    return preprocess_mask(Image.fromarray(final_mask).convert("L"))



def extract_image_from_editor(editor_value):
    """
    Extract background image from Gradio ImageEditor output.

    Prefer the editor background and strip alpha. This avoids feeding the painted mask
    strokes into the inpainting model.
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    return preprocess_input_image(background)



def create_comparison_image(original: Image.Image, mask: Image.Image, result: Image.Image) -> Image.Image:
    """
    Create side-by-side comparison: input | mask | result
    """
    original = original.convert("RGB")
    mask_rgb = mask.convert("RGB")
    result = result.convert("RGB")

    w, h = original.size
    comparison = Image.new("RGB", (w * 3, h), color=(255, 255, 255))
    comparison.paste(original, (0, 0))
    comparison.paste(mask_rgb, (w, 0))
    comparison.paste(result, (w * 2, 0))
    return comparison
