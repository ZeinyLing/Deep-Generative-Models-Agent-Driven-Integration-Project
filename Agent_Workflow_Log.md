# Agent Collaboration Workflow Log

## Project: AI-Powered Image Restoration Assistant

This document records a structured development workflow for the **AI-Powered Image Restoration Assistant**.  
It focuses on a reasonable engineering process, including requirement analysis, tool usage, technical decisions, and problems solved with the support of an AI agent.

---

## 1. Project Objective

The goal of this project is to build an interactive image restoration system using **Stable Diffusion Inpainting**.

The system supports two main tasks:

- **Remove Watermark**
- **Restore Damage / Scratch**

Users can upload an image, select a restoration task, provide a mask manually or automatically, and generate a restored image.

---

## 2. Development Workflow

```text
Requirement Analysis
        ↓
System Architecture Design
        ↓
Interface and Task Design
        ↓
Mask Processing Design
        ↓
Inpainting Pipeline Integration
        ↓
Prompt Preparation Design
        ↓
Testing and Debugging
        ↓
Documentation and Workflow Visualization
```

---

## 3. Requirement Analysis

### Main Requirements

| Requirement | Description |
|---|---|
| Image upload | Allow users to upload an image |
| Task selection | Provide restoration task options |
| Manual mask | Allow users to mark the region to restore |
| Auto detection | Automatically detect obvious scratches |
| Inpainting | Restore the selected region using a diffusion model |
| Output display | Show input, mask, restored image, and comparison |
| Download | Allow users to download output results |

### Key Prompt

```text
Design an image restoration assistant that supports watermark removal and damage restoration using mask-based inpainting.
```

---

## 4. System Architecture Design

The project was separated into several modules.

```text
app.py
├── Gradio interface
├── Task selection
├── User input handling
└── Output display

src/config.py
├── Model configuration
├── Device setting
├── Image size setting
├── API key loading
└── Token loading

src/image_utils.py
├── Image preprocessing
├── Manual mask extraction
├── Auto scratch mask detection
├── Size unification
└── Comparison image generation

src/llm_agent.py
├── Prompt preparation
├── Optional LLM prompt generation
└── Rule-based fallback prompt

src/inpaint_pipeline.py
├── Stable Diffusion Inpainting loading
├── Inference execution
├── Result postprocessing
└── Output saving
```

### Tool Combination

```text
ChatGPT + Python + Gradio + Diffusers + PIL + NumPy
```

---

## 5. Interface and Task Design

The interface was designed to be simple and task-oriented.

Only two task categories were kept:

```python
TASK_PRESETS = {
    "Remove Watermark": "remove watermark and restore the background naturally",
    "Restore Damage / Scratch": "repair the damaged or scratched area and restore natural texture",
}
```

| Task | Purpose |
|---|---|
| Remove Watermark | Remove watermark, text, or logo-like unwanted regions |
| Restore Damage / Scratch | Repair damaged areas, scratches, and visible defects |

### Key Prompt

```text
Simplify the restoration system into two main tasks: Remove Watermark and Restore Damage / Scratch.
```

---

## 6. Mask Processing Design

Since Stable Diffusion Inpainting requires both an image and a mask, mask processing is a key part of the system.

### Manual Mask

Users can draw over the region that needs restoration using the Gradio ImageEditor brush.

### Auto Scratch Detection

For damage restoration, the system also supports automatic scratch detection.  
This mode is suitable for bright, high-contrast scratches or cracks.

Basic logic:

```python
gray = image.convert("L")
blur = gray.filter(ImageFilter.GaussianBlur(radius=3))

bright = gray_np >= bright_threshold
high_contrast = (gray_np - blur_np) >= min_local_contrast

mask_np = (bright & high_contrast).astype(np.uint8) * 255
```

### Tool Combination

```text
PIL + NumPy + image thresholding + image filtering
```

---

## 7. Mask Debugging and Improvement

### Problem

The mask drawn in the image editor may be incorrectly interpreted.  
In some cases, the entire image can be detected as the mask.

### Cause

Some Gradio ImageEditor layers may store alpha information in an unreliable format.

### Solution

The system compares the original background image with the composite image after drawing.  
Only changed pixels are treated as the mask.

```python
diff = ImageChops.difference(background, composite)
score = np.abs(diff_np).sum(axis=2)
mask_np = (score > 25).astype(np.uint8) * 255
```

A mask area check was also added. If the detected mask covers too much of the image, the system asks the user to redraw the mask.

### Key Prompt

```text
Improve mask extraction so only the brush-painted region is used as the inpainting mask.
```

---

## 8. Image Size Unification

### Problem

The input image, mask, restored result, and comparison image may have different sizes or aspect ratios.

### Solution

The system uses aspect-ratio-preserving resizing with padding.

```python
def resize_keep_aspect_with_padding(image, target_size):
    scale = min(target_size / width, target_size / height)
    resized = image.resize((new_width, new_height))
    canvas = Image.new(image.mode, (target_size, target_size))
    canvas.paste(resized, (left, top))
    return canvas
```

### Result

All outputs are unified:

```text
Processed Input: IMAGE_SIZE x IMAGE_SIZE
Extracted Mask:  IMAGE_SIZE x IMAGE_SIZE
Restored Result: IMAGE_SIZE x IMAGE_SIZE
Comparison:      3*IMAGE_SIZE x IMAGE_SIZE
```

---

## 9. Inpainting Pipeline Integration

The restoration model is based on:

```python
StableDiffusionInpaintPipeline
```

### Pipeline Input

```text
Input image
Mask image
Positive prompt
Negative prompt
Inference steps
Guidance scale
Seed
```

### Pipeline Output

```text
Restored image
Comparison image
Saved output files
```

### Tool Combination

```text
Diffusers + PyTorch + Stable Diffusion Inpainting
```

---

## 10. Prompt Preparation Design

The system supports two prompt preparation modes.

### OpenRouter LLM Prompt Generation

If an OpenRouter API key is available, the system can use an LLM to generate the positive and negative prompts.

### Rule-based Fallback Prompt

If the API key is unavailable or the request fails, the system uses a short rule-based prompt.

Example for watermark removal:

```text
remove the masked watermark, restore natural background, match surrounding texture and lighting, preserve unmasked image
```

Example for scratch restoration:

```text
repair the masked damaged area, match surrounding texture, natural realistic restoration, preserve unmasked image
```

The interface uses the label **Prompt Used** instead of **Generated Prompt** to avoid confusion when the fallback prompt is used.

---

## 11. Runtime Warning Handling

| Warning | Cause | Handling |
|---|---|---|
| OpenRouter 400 | Invalid model name or API setting | Use a valid model or fallback prompt |
| Hugging Face warning | HF token is not set | Optional; add `HF_TOKEN` if needed |
| CLIP token limit | Prompt is too long | Use shorter prompts |
| xFormers warning | xFormers is not installed | Non-fatal; continue without it |
| safetensors warning | Model may use `.bin` weights | Non-fatal for local testing |

### Key Prompt

```text
Analyze runtime warnings and revise the program so the restoration pipeline remains stable.
```

---

## 12. Final System Workflow

```text
User Uploads Image
        ↓
Select Task
        ↓
Remove Watermark          Restore Damage / Scratch
        ↓                         ↓
Manual Mask          Manual Mask / Auto Detect Scratch
        ↓                         ↓
Image and Mask Preprocessing
        ↓
Prompt Preparation
        ↓
Stable Diffusion Inpainting
        ↓
Generate Restored Image
        ↓
Outputs:
- Processed Input
- Extracted Mask
- Restored Result
- Comparison Image
```

---

## 13. Key Tools Used

| Tool | Purpose |
|---|---|
| ChatGPT | System planning, debugging, code generation, documentation |
| Python | Main implementation language |
| Gradio | Web interface and image editor |
| PIL | Image preprocessing and mask processing |
| NumPy | Pixel-level mask calculation |
| Diffusers | Stable Diffusion Inpainting pipeline |
| PyTorch | Model execution |
| OpenRouter API | Optional prompt generation |
| Hugging Face Hub | Model download |
| Markdown | Documentation |
| Image generation tool | Workflow diagram creation |

---

## 14. Final Features

The final system includes:

- Task selection interface
- Manual mask drawing
- Automatic scratch detection
- Stable Diffusion Inpainting
- Short prompt preparation
- Negative prompt support
- Unified output size
- Restored image download
- Comparison image download
- README documentation
- System workflow diagram

---

## 15. Conclusion

The AI agent supported the project through a complete engineering workflow, including requirement analysis, system design, implementation, debugging, and documentation.  
The final project is a structured image restoration assistant that combines a Gradio interface, mask-based preprocessing, automatic scratch detection, prompt preparation, and Stable Diffusion Inpainting.
