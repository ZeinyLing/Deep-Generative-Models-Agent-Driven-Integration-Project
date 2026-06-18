import argparse
import gradio as gr

from src.config import DEVICE, OPENROUTER_API_KEY, OPENROUTER_MODEL
from src.image_utils import (
    extract_image_from_editor,
    extract_mask_from_editor,
)
from src.inpaint_pipeline import run_inpainting


TASK_PRESETS = {
    "Remove Watermark": "remove watermark and restore the background naturally",
    "Remove Text / Logo": "remove text or logo and restore the background naturally",
    "Remove Object": "remove the selected object and fill the region with a natural background",
    "Restore Damage / Scratch": "repair the damaged or scratched area and restore natural texture",
    "Clean Background": "clean the selected background region and make it visually consistent",
    "Custom": "",
}


def build_task_instruction(task_type, custom_instruction):
    task_type = task_type or "Remove Watermark"
    custom_instruction = (custom_instruction or "").strip()
    preset_instruction = TASK_PRESETS.get(task_type, "")

    if task_type == "Custom":
        if custom_instruction:
            return custom_instruction
        return "restore only the masked region naturally"

    if custom_instruction:
        return f"{preset_instruction}. Additional requirement: {custom_instruction}"

    return preset_instruction


def gradio_inference(
    editor_value,
    task_type,
    custom_instruction,
    steps,
    guidance_scale,
    seed,
):
    try:
        image = extract_image_from_editor(editor_value)
        mask = extract_mask_from_editor(editor_value)

        task_instruction = build_task_instruction(task_type, custom_instruction)

        output = run_inpainting(
            image=image,
            mask=mask,
            task_instruction=task_instruction,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return (
            task_instruction,
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

Current device: `{DEVICE}`  
LLM mode: `{llm_status}`  
OpenRouter model: `{OPENROUTER_MODEL}`

## How to use

1. Upload an image.
2. Use the **red brush** to draw over the region you want to restore.
3. Choose the restoration task.
4. Optionally enter extra requirements.
5. Click Generate Restoration.

All outputs are unified to the same canvas size:

- Processed Input
- Extracted Mask
- Restored Result
- Comparison
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
                        default_size=20,
                        colors=["#FF0000"],
                        color_mode="fixed",
                    ),
                    height=520,
                )

                task_type = gr.Dropdown(
                    label="Select Task",
                    choices=list(TASK_PRESETS.keys()),
                    value="Remove Text / Logo",
                )

                custom_instruction = gr.Textbox(
                    label="Custom Instruction / Extra Requirement",
                    placeholder=(
                        "Optional. Example: preserve original structure, keep grass texture, "
                        "do not create unrelated objects"
                    ),
                    lines=3,
                    value="preserve the original image structure and restore only the masked region",
                )

                with gr.Row():
                    steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=40,
                        step=1,
                        label="Inference Steps",
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=5.5,
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
                final_instruction = gr.Textbox(
                    label="Final Task Instruction",
                    lines=2,
                )

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
                        height=320,
                    )

                    processed_mask = gr.Image(
                        label="Extracted Mask",
                        type="pil",
                        height=320,
                    )

                result_image = gr.Image(
                    label="Restored Result",
                    type="pil",
                    height=320,
                )

                comparison_image = gr.Image(
                    label="Comparison: Input | Mask | Result",
                    type="pil",
                    height=320,
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
                task_type,
                custom_instruction,
                steps,
                guidance_scale,
                seed,
            ],
            outputs=[
                final_instruction,
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
