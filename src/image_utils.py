import numpy as np
from PIL import Image

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


def preprocess_mask(mask: Image.Image) -> Image.Image:
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

    return Image.fromarray(mask_np).convert("L")


def extract_mask_from_editor(editor_value):
    """
    Extract hand-drawn mask from Gradio ImageEditor output.

    Gradio ImageEditor usually returns a dictionary containing:
    - background
    - layers
    - composite

    The drawn mask is usually stored in layers.
    We convert the visible painted area into a binary mask.
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)
    layers = editor_value.get("layers", [])

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    if len(layers) == 0:
        raise ValueError("Please draw a mask on the image.")

    # Use all layers to build one mask
    final_mask = None

    for layer in layers:
        if layer is None:
            continue

        layer = layer.convert("RGBA")
        layer = layer.resize((IMAGE_SIZE, IMAGE_SIZE))

        layer_np = np.array(layer)

        # Alpha channel indicates painted region
        alpha = layer_np[:, :, 3]

        # Some Gradio versions may store brush strokes in RGB without strong alpha
        rgb = layer_np[:, :, :3]
        rgb_sum = rgb.sum(axis=2)

        mask_np = ((alpha > 10) | (rgb_sum > 30)).astype(np.uint8) * 255

        if final_mask is None:
            final_mask = mask_np
        else:
            final_mask = np.maximum(final_mask, mask_np)

    if final_mask is None:
        raise ValueError("No valid mask layer found.")

    final_mask = (final_mask > 127).astype(np.uint8) * 255

    return Image.fromarray(final_mask).convert("L")


def extract_image_from_editor(editor_value):
    """
    Extract background image from Gradio ImageEditor output.
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    return preprocess_input_image(background)


def create_comparison_image(
    original: Image.Image,
    mask: Image.Image,
    result: Image.Image
) -> Image.Image:
    """
    Create side-by-side comparison:
    input | mask | result
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