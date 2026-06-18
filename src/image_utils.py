import numpy as np
from PIL import Image, ImageChops, ImageFilter

from src.config import IMAGE_SIZE


def _resize_keep_aspect_with_padding(image: Image.Image, target_size: int, fill):
    """
    Resize image to target_size x target_size while keeping aspect ratio.
    Padding is added instead of stretching/cropping.

    This makes input / mask / result use the same canvas.
    """
    image = image.convert("RGB") if image.mode != "L" else image.convert("L")
    w, h = image.size

    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resample = Image.Resampling.BILINEAR if image.mode != "L" else Image.Resampling.NEAREST
    resized = image.resize((new_w, new_h), resample)

    canvas = Image.new(image.mode, (target_size, target_size), fill)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))

    return canvas


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
    Convert mask image to binary grayscale mask and place it on the same square canvas.

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


def _mask_from_composite_difference(background: Image.Image, composite: Image.Image) -> Image.Image:
    """
    Detect brush region by comparing original background and editor composite.
    This avoids the Gradio layer-alpha issue where the whole canvas becomes mask.
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
    Fallback mask extraction from editor layers.
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

        if 0 < alpha_ratio < 0.80:
            mask_np = (alpha > 20).astype(np.uint8) * 255
        else:
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)

            red_stroke = (r > 150) & (r - g > 50) & (r - b > 50)

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
        raise ValueError("No valid mask was detected. Please draw again with the red brush.")

    if mask_ratio > 0.60:
        raise ValueError(
            f"The detected mask covers {mask_ratio:.1%} of the image. "
            "This is too large. Please clear the layer and draw a smaller mask."
        )

    return preprocess_mask(Image.fromarray(mask_np).convert("L"))


def extract_mask_from_editor(editor_value):
    """
    Extract hand-drawn mask from Gradio ImageEditor output.

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
            return _validate_mask(mask)

    if not layers:
        raise ValueError("Please draw a mask on the image.")

    mask = _mask_from_layers_fallback(background, layers)
    return _validate_mask(mask)


def extract_image_from_editor(editor_value):
    """
    Extract original background image from Gradio ImageEditor output.
    Do not use composite because composite contains brush strokes.
    """
    if editor_value is None:
        raise ValueError("No image editor input provided.")

    if not isinstance(editor_value, dict):
        raise ValueError("ImageEditor output format is invalid.")

    background = editor_value.get("background", None)

    if background is None:
        raise ValueError("Please upload an image in the editor.")

    return preprocess_input_image(background)


def force_same_size(result: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Force model result to match reference size exactly.
    """
    result = result.convert("RGB")
    reference = reference.convert("RGB")

    if result.size != reference.size:
        result = result.resize(reference.size, Image.Resampling.BILINEAR)

    return result


def create_comparison_image(
    original: Image.Image,
    mask: Image.Image,
    result: Image.Image
) -> Image.Image:
    """
    Create fixed-size side-by-side comparison:
    input | mask | result

    All three panels are forced to the same size.
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
