import numpy as np
from PIL import Image, ImageChops, ImageFilter

from src.config import IMAGE_SIZE


# =========================================================
# Common resize helpers
# =========================================================

def _resize_keep_aspect_with_padding(image: Image.Image, target_size: int, fill):
    """
    Resize to target_size x target_size while preserving aspect ratio.
    Pad remaining area instead of stretching.
    """
    mode = "L" if image.mode == "L" else "RGB"
    image = image.convert(mode)

    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resample = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.BILINEAR
    resized = image.resize((new_w, new_h), resample)

    canvas = Image.new(mode, (target_size, target_size), fill)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


# =========================================================
# Image / mask preprocessing
# =========================================================

def preprocess_input_image(image: Image.Image) -> Image.Image:
    """
    Convert uploaded image to RGB and place it on a fixed square canvas.
    Aspect ratio is preserved.
    """
    if image is None:
        raise ValueError("No input image provided.")

    image = image.convert("RGB")
    return _resize_keep_aspect_with_padding(image, IMAGE_SIZE, fill=(255, 255, 255))



def preprocess_mask(mask: Image.Image, blur_radius: float = 0.0) -> Image.Image:
    """
    Convert mask image to binary grayscale mask on the same square canvas.

    White region = area to restore.
    Black region = area to preserve.
    """
    if mask is None:
        raise ValueError("No mask image provided.")

    mask = mask.convert("L")
    mask = _resize_keep_aspect_with_padding(mask, IMAGE_SIZE, fill=0)

    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np).convert("L")

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask


# =========================================================
# Manual mask extraction from Gradio ImageEditor
# =========================================================

def _mask_from_composite_difference(background: Image.Image, composite: Image.Image) -> Image.Image:
    """
    Preferred method for Gradio ImageEditor.
    Compare the original background and the composite preview to detect brush strokes.
    """
    bg = background.convert("RGB")
    comp = composite.convert("RGB")

    if bg.size != comp.size:
        comp = comp.resize(bg.size)

    diff = ImageChops.difference(bg, comp)
    diff_np = np.array(diff).astype(np.int16)
    score = np.abs(diff_np).sum(axis=2)

    mask_np = (score > 25).astype(np.uint8) * 255
    return Image.fromarray(mask_np).convert("L")



def _mask_from_layers_fallback(background: Image.Image, layers) -> Image.Image:
    """
    Fallback for Gradio versions where composite may not be reliable.
    """
    bg_w, bg_h = background.size
    final_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    bg_rgb = np.array(background.convert("RGB"))

    for layer in layers:
        if layer is None:
            continue

        layer = layer.resize((bg_w, bg_h)).convert("RGBA")
        layer_np = np.array(layer)

        alpha = layer_np[:, :, 3]
        rgb = layer_np[:, :, :3]
        alpha_ratio = float((alpha > 20).mean())

        # Normal transparent brush layer
        if 0 < alpha_ratio < 0.80:
            mask_np = (alpha > 20).astype(np.uint8) * 255
        else:
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)

            red_stroke = (r > 150) & (r - g > 50) & (r - b > 50)
            color_diff = np.abs(rgb.astype(np.int16) - bg_rgb.astype(np.int16)).sum(axis=2)
            white_stroke = (rgb.sum(axis=2) > 650) & (color_diff > 25)
            mask_np = (red_stroke | white_stroke).astype(np.uint8) * 255

        final_mask = np.maximum(final_mask, mask_np)

    return Image.fromarray(final_mask).convert("L")



def _validate_raw_mask(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask_ratio = float((mask_np > 0).mean())

    if mask_ratio == 0:
        raise ValueError("No valid mask was detected. Please draw again with the red brush.")

    if mask_ratio > 0.60:
        raise ValueError(
            f"The detected mask covers {mask_ratio:.1%} of the image. "
            "This is too large and usually means the editor layer was parsed incorrectly. "
            "Please clear the layer and redraw a smaller mask."
        )

    return Image.fromarray(mask_np).convert("L")



def extract_mask_from_editor(editor_value):
    """
    Extract hand-drawn mask from Gradio ImageEditor.
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

    if composite is not None:
        mask = _mask_from_composite_difference(background, composite)
        mask_np = np.array(mask)
        if (mask_np > 0).mean() > 0:
            return preprocess_mask(_validate_raw_mask(mask))

    if not layers:
        raise ValueError("Please draw a mask on the image.")

    mask = _mask_from_layers_fallback(background, layers)
    return preprocess_mask(_validate_raw_mask(mask))



def extract_image_from_editor(editor_value):
    """
    Extract the original background image from Gradio ImageEditor.
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    return preprocess_input_image(background)


# =========================================================
# Auto damage / scratch detection
# =========================================================

def auto_detect_damage_mask(
    image: Image.Image,
    bright_threshold: int = 210,
    min_local_contrast: int = 35,
    line_thickness: int = 3,
) -> Image.Image:
    """
    Auto-detect bright scratch / damage lines.

    Best for:
    - white scratches
    - bright thin cracks
    - obvious high-contrast damage marks

    Strategy:
    1. convert to grayscale
    2. find very bright pixels
    3. require local contrast against surrounding area
    4. slightly dilate / thicken detected lines for inpainting
    """
    if image is None:
        raise ValueError("No input image provided.")

    gray = image.convert("L")
    gray_np = np.array(gray).astype(np.int16)

    # local mean via blur to estimate surrounding tone
    blur = gray.filter(ImageFilter.GaussianBlur(radius=3))
    blur_np = np.array(blur).astype(np.int16)

    bright = gray_np >= bright_threshold
    high_contrast = (gray_np - blur_np) >= min_local_contrast

    # Detect bright thin damage marks.
    mask_np = (bright & high_contrast).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np).convert("L")

    # Thicken a little so inpainting covers the full scratch.
    for _ in range(max(1, line_thickness)):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))

    # Optional light cleanup
    mask = mask.filter(ImageFilter.MedianFilter(size=3))

    # Validate and then preprocess to the final square canvas.
    try:
        validated = _validate_raw_mask(mask)
    except ValueError:
        raise ValueError(
            "Auto damage detection could not find a clear damage region. "
            "Please switch to Manual Mask mode and draw the damaged area yourself."
        )

    return preprocess_mask(validated)


# =========================================================
# Output helpers
# =========================================================

def force_same_size(result: Image.Image, reference: Image.Image) -> Image.Image:
    result = result.convert("RGB")
    reference = reference.convert("RGB")

    if result.size != reference.size:
        result = result.resize(reference.size, Image.Resampling.BILINEAR)

    return result



def create_comparison_image(original: Image.Image, mask: Image.Image, result: Image.Image) -> Image.Image:
    """
    Create side-by-side comparison: input | mask | result
    All panels use the same size.
    """
    original = original.convert("RGB")
    mask_rgb = mask.convert("RGB")
    result = force_same_size(result, original)

    w, h = original.size
    comparison = Image.new("RGB", (w * 3, h), color=(255, 255, 255))
    comparison.paste(original, (0, 0))
    comparison.paste(mask_rgb.resize((w, h), Image.Resampling.NEAREST), (w, 0))
    comparison.paste(result, (w * 2, 0))
    return comparison
