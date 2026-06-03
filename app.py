import argparse
import gradio as gr

from src.config import DEVICE, OPENROUTER_API_KEY, OPENROUTER_MODEL
from src.image_utils import (
    extract_image_from_editor,
    extract_mask_from_editor,
)
from src.inpaint_pipeline import run_inpainting


def gradio_inference(
    editor_value,
    task_instruction,
    steps,
    guidance_scale,
    seed,
):
    try:
        image = extract_image_from_editor(editor_value)
        mask = extract_mask_from_editor(editor_value)

        output = run_inpainting(
            image=image,
            mask=mask,
            task_instruction=task_instruction,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return (
            output["positive_prompt"],
            output["negative_prompt"],
            output["input_image"],
            output["processed_mask"],
            output["result_image"],
            output["comparison_image"],
            output["result_path"],
            output["comparison_path"],
        )

    except Exception as e:
        raise gr.Error(str(e))


def build_app():
    llm_status = "OpenRouter API enabled" if OPENROUTER_API_KEY else "Fallback prompt generator enabled"

    with gr.Blocks(title="AI Image Restoration Assistant") as demo:
        gr.Markdown(
            f"""
# AI-Powered Image Restoration Assistant

This app combines:

- **Hand-drawn Mask**
- **LLM Prompt Assistant**
- **Stable Diffusion Inpainting**

Current device: `{DEVICE}`  
LLM mode: `{llm_status}`  
OpenRouter model: `{OPENROUTER_MODEL}`

## How to use

1. Upload an image.
2. Use the brush to draw over the region you want to restore.
3. Enter a task instruction.
4. Click Generate Restoration.

Mask rule:

- Painted area = region to restore
- Unpainted area = preserved region
"""
        )

        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="Upload Image and Draw Mask",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                    brush=gr.Brush(
                        default_size=30,
                        colors=["#FFFFFF"],
                        color_mode="fixed",
                    ),
                    height=520,
                )

                task_instruction = gr.Textbox(
                    label="Task Instruction",
                    placeholder="Example: remove watermark and restore natural background",
                    lines=3,
                    value="remove watermark and restore the background naturally",
                )

                with gr.Row():
                    steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Inference Steps",
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.0,
                        step=0.5,
                        label="Guidance Scale",
                    )

                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="Seed",
                )

                run_button = gr.Button(
                    "Generate Restoration",
                    variant="primary",
                )

            with gr.Column():
                positive_prompt = gr.Textbox(
                    label="Generated Positive Prompt",
                    lines=4,
                )

                negative_prompt = gr.Textbox(
                    label="Generated Negative Prompt",
                    lines=4,
                )

                with gr.Row():
                    processed_input = gr.Image(
                        label="Processed Input",
                        type="pil",
                    )

                    processed_mask = gr.Image(
                        label="Extracted Mask",
                        type="pil",
                    )

                result_image = gr.Image(
                    label="Restored Result",
                    type="pil",
                )

                comparison_image = gr.Image(
                    label="Comparison: Input | Mask | Result",
                    type="pil",
                )

                with gr.Row():
                    result_file = gr.File(
                        label="Download Result",
                    )

                    comparison_file = gr.File(
                        label="Download Comparison",
                    )

        run_button.click(
            fn=gradio_inference,
            inputs=[
                editor,
                task_instruction,
                steps,
                guidance_scale,
                seed,
            ],
            outputs=[
                positive_prompt,
                negative_prompt,
                processed_input,
                processed_mask,
                result_image,
                comparison_image,
                result_file,
                comparison_file,
            ],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link.",
    )

    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
    )

    args = parser.parse_args()

    demo = build_app()

    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
    )