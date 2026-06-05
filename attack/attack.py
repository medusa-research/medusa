"""Medusa SVD attack entrypoint."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.dirname(__file__))

from sgm.util import append_dims
from utils.attention import DEFAULT_ATTACK_BLOCKS, TemporalAttentionHook
from utils.svd import (
    denoise_step,
    encode_first_stage_with_grad,
    generate_video_auto_sharded,
    load_model_auto_sharded,
    prepare_conditioning,
    resolve_repo_path,
)


def load_image_tensor(image_path: str, device: str) -> torch.Tensor:
    """Load an image, resize to SVD-compatible dimensions, and scale to [-1, 1]."""
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        width, height = image.size
        if height % 64 != 0 or width % 64 != 0:
            width, height = (width - width % 64, height - height % 64)
            image = image.resize((width, height))
        array = np.array(image)

    return (ToTensor()(array) * 2.0 - 1.0).unsqueeze(0).to(device)


def save_image_tensor(image: torch.Tensor, path: str) -> None:
    """Save a [-1, 1] image tensor as an RGB PNG."""
    image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
    array = (image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def run_attention_attack(
    model,
    model_info: Dict,
    image_path: str,
    num_frames: int = 14,
    num_steps: int = 25,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    cond_aug: float = 0.02,
    seed: int = 23,
    learning_rate: float = 0.02,
    epsilon: float = 16 / 255,
    iterations: int = 50,
    target_timestep: int = 4,
):
    """Optimize an input image by minimizing temporal attention nuclear norm."""
    primary_device = model_info["primary_device"]
    attention_device = model_info.get("attention_device", primary_device)

    torch.manual_seed(seed)
    source_image = load_image_tensor(image_path, primary_device)
    adversarial_image = source_image.clone().detach().requires_grad_(True)

    sigmas = model.sampler.discretization(num_steps, device=primary_device)
    target_sigma = sigmas[target_timestep]

    attention_hook = TemporalAttentionHook(model_info.get("attack_blocks", DEFAULT_ATTACK_BLOCKS))
    hook_count = attention_hook.register_hooks(model)
    if hook_count == 0:
        raise RuntimeError("No temporal attention hooks were registered; cannot run the attack")

    print("\nStarting medusa attack")
    print(f"  Iterations: {iterations}, epsilon: {epsilon}, lr: {learning_rate}")

    try:
        for iteration in tqdm(range(iterations), desc="Attack"):
            if adversarial_image.grad is not None:
                adversarial_image.grad.zero_()
            attention_hook.clear()

            latent = encode_first_stage_with_grad(model, adversarial_image)
            latent_frames = latent.repeat(num_frames, 1, 1, 1)
            torch.manual_seed(seed)
            noise = torch.randn(
                num_frames,
                4,
                source_image.shape[2] // 8,
                source_image.shape[3] // 8,
                device=primary_device,
            )
            noisy_latent = latent_frames + noise * append_dims(target_sigma, latent_frames.ndim)

            conditioning = prepare_conditioning(
                model,
                adversarial_image,
                num_frames,
                motion_bucket_id,
                fps_id,
                cond_aug,
                primary_device,
            )
            denoised = denoise_step(model, noisy_latent, target_sigma, conditioning, num_frames)

            loss = attention_hook.compute_nuclear_loss(attention_device).to(primary_device)
            loss.backward()

            if adversarial_image.grad is not None:
                with torch.no_grad():
                    adversarial_image -= learning_rate * adversarial_image.grad.sign()
                    delta = torch.clamp(adversarial_image - source_image, -epsilon, epsilon)
                    adversarial_image.copy_(torch.clamp(source_image + delta, -1.0, 1.0))

            if iteration % 10 == 0 or iteration == iterations - 1:
                print(f"  Iter {iteration}: medusa_loss={loss.item():.4f}")

            del denoised, noisy_latent, conditioning, latent_frames, latent
            torch.cuda.empty_cache()
    finally:
        attention_hook.remove_hooks()

    print("\nAttack complete")
    return adversarial_image.detach(), source_image


def iter_input_images(input_path: str):
    """Return one image path or all image files in a directory."""
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return sorted(
            str(item)
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
    raise ValueError(f"Invalid input path: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Medusa SVD attention nuclear-norm attack")
    parser.add_argument("--input_path", type=str, default="assets/examples")
    parser.add_argument("--attack_image_folder", type=str, default="assets/attack")
    parser.add_argument("--output_folder", type=str, default="assets/outputs")
    parser.add_argument("--model_config", type=str, default="scripts/sampling/configs/svd.yaml")

    parser.add_argument("--devices", type=str, default=None, help="Comma-separated CUDA devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--max_memory", type=str, default=None, help="Per-device caps, e.g. cuda:0=70GiB,cuda:1=70GiB")

    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps_id", type=int, default=6)
    parser.add_argument("--cond_aug", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=23)

    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--target_timestep", type=int, default=4)
    parser.add_argument("--save_video", action="store_true", help="Generate an adversarial video after saving the image")

    args = parser.parse_args()
    args.input_path = resolve_repo_path(args.input_path)
    args.attack_image_folder = resolve_repo_path(args.attack_image_folder)
    args.output_folder = resolve_repo_path(args.output_folder)
    args.model_config = resolve_repo_path(args.model_config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this attack")
    print(f"Found {torch.cuda.device_count()} visible GPU(s)")

    os.makedirs(args.attack_image_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    model, model_info = load_model_auto_sharded(
        args.model_config,
        args.num_frames,
        args.num_steps,
        devices=args.devices,
        max_memory=args.max_memory,
        attack_blocks=DEFAULT_ATTACK_BLOCKS,
    )

    image_paths = iter_input_images(args.input_path)
    print(f"\nFound {len(image_paths)} image(s) to process")

    for index, input_img_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {index}/{len(image_paths)}: {input_img_path}")
        adversarial_image, _ = run_attention_attack(
            model=model,
            model_info=model_info,
            image_path=input_img_path,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            motion_bucket_id=args.motion_bucket_id,
            fps_id=args.fps_id,
            cond_aug=args.cond_aug,
            seed=args.seed,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon,
            iterations=args.iterations,
            target_timestep=args.target_timestep,
        )

        base_name = Path(input_img_path).stem
        adv_path = os.path.join(args.attack_image_folder, f"{base_name}_adv.png")
        save_image_tensor(adversarial_image, adv_path)
        print(f"Saved adversarial image: {adv_path}")

        if args.save_video:
            video = generate_video_auto_sharded(
                model,
                model_info,
                adversarial_image,
                args.num_frames,
                args.num_steps,
                args.motion_bucket_id,
                args.fps_id,
                args.cond_aug,
                seed=args.seed,
            )
            video_path = os.path.join(args.output_folder, f"{base_name}_adv.mp4")
            frames = (rearrange(video, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
            imageio.mimwrite(video_path, frames, fps=args.fps_id)
            print(f"Saved adversarial video: {video_path}")


if __name__ == "__main__":
    main()
