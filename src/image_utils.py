import numpy as np
from PIL import Image, ImageChops, ImageFilter

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
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255

    mask = Image.fromarray(mask_np).convert("L")

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask


def _mask_from_composite_difference(background: Image.Image, composite: Image.Image) -> Image.Image:
    """
    Best method for Gradio ImageEditor.

    We compare:
    - original background
    - editor composite after brush drawing

    Only changed pixels are treated as mask.
    This avoids the common bug where the whole editor layer has alpha=255
    and the full image becomes a mask.
    """
    bg = background.convert("RGB")
    comp = composite.convert("RGB")

    if bg.size != comp.size:
        comp = comp.resize(bg.size)

    diff = ImageChops.difference(bg, comp)
    diff_np = np.array(diff).astype(np.int16)

    # Color difference between original image and painted brush strokes.
    score = np.abs(diff_np).sum(axis=2)

    # Threshold: brush stroke should be strongly different from background.
    mask_np = (score > 25).astype(np.uint8) * 255

    return Image.fromarray(mask_np).convert("L")


def _mask_from_layers_fallback(background: Image.Image, layers) -> Image.Image:
    """
    Fallback method.

    Some Gradio versions provide actual transparent brush layers.
    Some versions provide a full white canvas. For that case, we avoid using
    full alpha directly because it makes the whole image a mask.
    """
    bg_w, bg_h = background.size
    final_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue

        layer = layer.resize((bg_w, bg_h)).convert("RGBA")
        layer_np = np.array(layer)

        alpha = layer_np[:, :, 3]
        rgb = layer_np[:, :, :3]

        alpha_ratio = float((alpha > 20).mean())

        # Case 1: normal transparent brush layer.
        # If alpha does not cover almost the whole canvas, alpha is reliable.
        if 0 < alpha_ratio < 0.80:
            mask_np = (alpha > 20).astype(np.uint8) * 255

        # Case 2: full alpha canvas; detect colored brush pixels only.
        else:
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)

            # App uses red brush by default, so detect red strokes robustly.
            red_stroke = (r > 150) & (r - g > 50) & (r - b > 50)

            # Also support white brush strokes, but avoid treating a white canvas
            # as mask by requiring local difference from the background.
            bg_rgb = np.array(background.convert("RGB"))
            color_diff = np.abs(rgb.astype(np.int16) - bg_rgb.astype(np.int16)).sum(axis=2)
            white_stroke = (rgb.sum(axis=2) > 650) & (color_diff > 25)

            mask_np = (red_stroke | white_stroke).astype(np.uint8) * 255

        final_mask = np.maximum(final_mask, mask_np)

    return Image.fromarray(final_mask).convert("L")


def _validate_mask(mask: Image.Image) -> Image.Image:
    """
    Validate mask size and convert to target resolution.
    """
    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 127).astype(np.uint8) * 255

    mask_ratio = float((mask_np > 0).mean())

    if mask_ratio == 0:
        raise ValueError(
            "No valid mask was detected. Please draw again with the red brush."
        )

    if mask_ratio > 0.60:
        raise ValueError(
            f"The detected mask covers {mask_ratio:.1%} of the image. "
            "This is too large and usually means the editor layer was parsed incorrectly. "
            "Please clear the layer and draw with the red brush again."
        )

    return preprocess_mask(Image.fromarray(mask_np).convert("L"))


def extract_mask_from_editor(editor_value):
    """
    Extract hand-drawn mask from Gradio ImageEditor output.

    Priority:
    1. composite - background difference
    2. layer fallback

    Output:
    - white = inpaint area
    - black = keep area
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)
    composite = editor_value.get("composite", None)
    layers = editor_value.get("layers", [])

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    background = background.convert("RGB")

    # First try composite difference. This is most stable across Gradio versions.
    if composite is not None:
        mask = _mask_from_composite_difference(background, composite)
        mask_np = np.array(mask)
        if (mask_np > 0).mean() > 0:
            return _validate_mask(mask)

    # Fallback to layer parsing.
    if not layers:
        raise ValueError("Please draw a mask on the image.")

    mask = _mask_from_layers_fallback(background, layers)
    return _validate_mask(mask)


def extract_image_from_editor(editor_value):
    """
    Extract original background image from Gradio ImageEditor output.
    Do not use composite here, because composite contains brush strokes.
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
