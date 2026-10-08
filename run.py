"""Minimal Qwen-Image-2.1 generation with one sampled attention read."""

import argparse
from pathlib import Path

import torch

from attention.controller import AttentionController
from attention.hook import AttentionHook
from attention.store import AttentionStore
from models.qwen_image import load_pipeline, processor_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/sample.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-layer", type=int, default=0, help="DiT layer index, 0 to 31")
    parser.add_argument("--capture-step", type=int, default=0, help="Zero-based denoising step")
    args = parser.parse_args()

    if args.steps < 2:
        parser.error("--steps must be at least 2 for the local FlowMatch scheduler")
    if args.capture_step < 0 or args.capture_step >= args.steps:
        parser.error("--capture-step must be between 0 and --steps - 1")
    if args.capture_layer < 0 or args.capture_layer >= 32:
        parser.error("--capture-layer must be between 0 and 31")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        parser.error("--width and --height must be positive multiples of 16")

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-2.1 inference requires CUDA in this prototype")
    pipe = load_pipeline()
    layer = f"transformer_blocks.{args.capture_layer}.attn"
    controller = AttentionController(capture_layer=layer, capture_step=args.capture_step)
    controller.timestep_provider = lambda: pipe._current_timestep
    store = AttentionStore()

    def on_step_end(_pipe, step, _timestep, tensors):
        controller.set_step(step + 1)
        return tensors

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with AttentionHook(pipe.transformer, controller, store, processor_factory):
        image = pipe(
            prompt=args.prompt, width=args.width, height=args.height,
            num_inference_steps=args.steps, generator=generator,
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    image.save(output)
    print(f"[run] saved {output} ({image.size[0]}x{image.size[1]})", flush=True)
    if not store.records:
        raise RuntimeError("No attention captured: check --capture-layer and --capture-step")


if __name__ == "__main__":
    main()
