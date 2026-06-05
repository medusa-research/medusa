"""SVD model and conditioning helpers for the Medusa attack."""

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

from utils.model_parallel import parse_devices, parse_max_memory, setup_auto_sharded_model
from sgm.util import append_dims, instantiate_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str) -> str:
    """Resolve relative CLI paths from the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def get_batch(keys, value_dict, shape, num_frames, device):
    """Build the conditioning batch expected by SVD's conditioner."""
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(shape)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(shape)))
        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(shape))
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=shape[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=shape[0])

    batch["num_video_frames"] = num_frames

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_uc[key] = value.clone()

    return batch, batch_uc


def prepare_conditioning(model, image, num_frames, motion_bucket_id, fps_id, cond_aug, device):
    """Prepare SVD conditional and unconditional embeddings for one image."""
    value_dict = {
        "cond_frames_without_noise": image,
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames": image + cond_aug * torch.randn_like(image),
    }

    embedder_keys = list({embedder.input_key for embedder in model.conditioner.embedders})
    batch, batch_uc = get_batch(embedder_keys, value_dict, [1, num_frames], num_frames, device)

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
    )

    for key in ["crossattn", "concat"]:
        if key in uc:
            uc[key] = repeat(uc[key], "b ... -> b t ...", t=num_frames)
            uc[key] = rearrange(uc[key], "b t ... -> (b t) ...", t=num_frames)
        if key in c:
            c[key] = repeat(c[key], "b ... -> b t ...", t=num_frames)
            c[key] = rearrange(c[key], "b t ... -> (b t) ...", t=num_frames)

    return {"c": c, "uc": uc, "batch": batch}


def denoise_step(model, noisy_latent, sigma, conditioning, num_frames):
    """Run one guided denoising step; attention hooks fire inside this call."""
    c = conditioning["c"]
    uc = conditioning["uc"]
    batch = conditioning.get("batch", {})
    additional_model_inputs = {
        "image_only_indicator": torch.zeros(2, num_frames).to(noisy_latent.device),
        "num_video_frames": batch.get("num_video_frames", num_frames),
    }

    def denoiser(input, sigma_in, cond):
        return model.denoiser(model.model, input, sigma_in, cond, **additional_model_inputs)

    sigma_in = torch.ones([noisy_latent.shape[0]], device=noisy_latent.device) * sigma
    x_in, sigma_in, cond_in = model.sampler.guider.prepare_inputs(noisy_latent, sigma_in, c, uc)
    denoised = denoiser(x_in, sigma_in, cond_in)
    return model.sampler.guider(denoised, sigma_in)


def encode_first_stage_with_grad(model, image):
    """Encode an image through the VAE while preserving gradients to the image."""
    with torch.autocast("cuda", enabled=False):
        return model.scale_factor * model.first_stage_model.encode(image)


def generate_video_auto_sharded(
    model,
    model_info,
    image,
    num_frames,
    num_steps,
    motion_bucket_id,
    fps_id,
    cond_aug,
    decoding_t=7,
    seed=23,
):
    """Generate a video from an image with the sharded model layout."""
    primary_device = model_info["primary_device"]
    decoder_device = model_info["decoder_device"]

    image = image.to(primary_device)
    height, width = image.shape[2:]
    latent_shape = (num_frames, 4, height // 8, width // 8)

    torch.manual_seed(seed)
    conditioning = prepare_conditioning(model, image, num_frames, motion_bucket_id, fps_id, cond_aug, primary_device)

    with torch.no_grad(), torch.autocast("cuda"):
        additional_model_inputs = {
            "image_only_indicator": torch.zeros(2, num_frames).to(primary_device),
            "num_video_frames": num_frames,
        }

        def denoiser(input, sigma, cond):
            return model.denoiser(model.model, input, sigma, cond, **additional_model_inputs)

        torch.manual_seed(seed)
        randn = torch.randn(latent_shape, device=primary_device)
        samples_z = model.sampler(denoiser, randn, cond=conditioning["c"], uc=conditioning["uc"])
        samples_z = samples_z.to(decoder_device)

        z = samples_z / model.scale_factor
        from sgm.modules.autoencoding.temporal_ae import VideoDecoder

        is_video_decoder = isinstance(model.first_stage_model.decoder, VideoDecoder)
        outputs = []
        for start in range(0, z.shape[0], decoding_t):
            z_batch = z[start : start + decoding_t]
            kwargs = {"timesteps": z_batch.shape[0]} if is_video_decoder else {}
            outputs.append(model.first_stage_model.decode(z_batch, **kwargs))

        samples = torch.cat(outputs, dim=0)
        return torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)


def load_model_auto_sharded(
    config_path: str,
    num_frames: int,
    num_steps: int,
    devices: Optional[str] = None,
    max_memory: Optional[str] = None,
    attack_blocks: Sequence[int] = (5,),
):
    """Load SVD on CPU first, freeze model parameters, then shard it across GPUs."""
    print(f"Loading model from {config_path}...")
    config_path = resolve_repo_path(config_path)
    config = OmegaConf.load(config_path)

    ckpt_path = config.model.params.get("ckpt_path")
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            repo_ckpt_path = REPO_ROOT / ckpt_path
            config_ckpt_path = Path(config_path).resolve().parent / ckpt_path
            ckpt_path = repo_ckpt_path if repo_ckpt_path.exists() else config_ckpt_path
        config.model.params.ckpt_path = str(ckpt_path)

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = "cpu"

    model = instantiate_from_config(config.model).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model_info = setup_auto_sharded_model(
        model,
        devices=parse_devices(devices),
        max_memory=parse_max_memory(max_memory),
        attack_blocks=attack_blocks,
    )
    return model, model_info
