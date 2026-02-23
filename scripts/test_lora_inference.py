"""Test LoRA inference with rob-character on SDXL Base 1.0.

Usage:
    source .venv/bin/activate
    python scripts/test_lora_inference.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
LORA_PATH = BASE / "output/loras/character/rob-character.safetensors"
OUTPUT_DIR = BASE / "output/lora_tests"
PROMPT = "rob_char, man standing on a beach, golden hour lighting"
NEGATIVE = "blurry, deformed, low quality, ugly, bad anatomy"
LORA_SCALE = 0.8
SEED = 42


def main() -> None:
    if not LORA_PATH.exists():
        print(f"ERROR: LoRA not found at {LORA_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SDXL Base 1.0...")
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(desc="Generating")

    print(f"Loading LoRA: {LORA_PATH.name}  (scale={LORA_SCALE})")
    pipe.load_lora_weights(str(LORA_PATH))

    print(f"Prompt: {PROMPT}")
    generator = torch.Generator("cuda").manual_seed(SEED)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": LORA_SCALE},
        generator=generator,
    ).images[0]

    out_path = OUTPUT_DIR / f"rob_char_seed{SEED}.png"
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
