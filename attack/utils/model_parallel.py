"""Automatic model-parallel placement for Medusa's SVD attack."""

from typing import Dict, Iterable, List, Optional, Sequence

import torch

from sgm.modules.diffusionmodules.util import timestep_embedding

CHECKPOINT_BLOCK_RADIUS = 2


def parse_devices(devices: Optional[str]) -> List[str]:
    """Return CLI-selected CUDA devices, or every visible GPU."""
    if devices:
        parsed = [device.strip() for device in devices.split(",") if device.strip()]
        if not parsed:
            raise ValueError("--devices was provided but no valid device was found")
        return parsed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for automatic sharding")
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def parse_max_memory(max_memory: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse memory caps like 'cuda:0=70GiB,cuda:1=70GiB' into bytes."""
    if not max_memory:
        return None

    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
    }
    result: Dict[str, int] = {}
    for item in max_memory.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid --max_memory item: {item!r}")
        device, value = [part.strip() for part in item.split("=", 1)]
        number = "".join(ch for ch in value if ch.isdigit() or ch == ".")
        unit = value[len(number):].strip().lower() or "b"
        if unit not in units or not number:
            raise ValueError(f"Invalid memory value: {value!r}")
        result[device] = int(float(number) * units[unit])
    return result


def module_nbytes(module: torch.nn.Module) -> int:
    """Approximate a module's parameter and buffer footprint in bytes."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    )


def available_device_memory(devices: List[str], max_memory: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Estimate per-device memory budgets while leaving room for attack activations."""
    budgets: Dict[str, int] = {}
    for device in devices:
        if max_memory and device in max_memory:
            budgets[device] = max_memory[device]
            continue
        if not device.startswith("cuda"):
            budgets[device] = 0
            continue
        idx = torch.device(device).index
        if idx is None:
            idx = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        budgets[device] = int(min(free_bytes, total_bytes) * 0.85)
    return budgets


def choose_devices_for_stages(stage_sizes: List[int], devices: List[str], budgets: Dict[str, int]) -> List[str]:
    """Greedily assign ordered UNet stages to devices without splitting a stage."""
    if not devices:
        raise ValueError("At least one CUDA device is required")
    if len(devices) == 1:
        return [devices[0] for _ in stage_sizes]

    total_size = sum(stage_sizes)
    total_budget = sum(max(1, budgets.get(device, 1)) for device in devices)
    target_used = {device: total_size * max(1, budgets.get(device, 1)) / total_budget for device in devices}

    assignments: List[str] = []
    device_idx = 0
    used_on_device = 0
    remaining_size = total_size

    for stage_idx, size in enumerate(stage_sizes):
        remaining_stages = len(stage_sizes) - stage_idx
        remaining_devices = len(devices) - device_idx
        device = devices[device_idx]

        should_advance = (
            remaining_devices > 1
            and assignments
            and used_on_device > 0
            and used_on_device + size > target_used[device]
            and remaining_stages >= remaining_devices
            and remaining_size > size
        )
        if should_advance:
            device_idx += 1
            device = devices[device_idx]
            used_on_device = 0

        assignments.append(device)
        used_on_device += size
        remaining_size -= size

    return assignments


def build_unet_stages(unet) -> List[Dict]:
    """Represent VideoUNet as ordered stages matching its forward pass."""
    stages = [{"name": "time_embed", "module": unet.time_embed}]
    for idx, block in enumerate(unet.input_blocks):
        stages.append({"name": f"input_blocks.{idx}", "module": block})
    stages.append({"name": "middle_block", "module": unet.middle_block})
    for idx, block in enumerate(unet.output_blocks):
        stages.append({"name": f"output_blocks.{idx}", "module": block})
    stages.append({"name": "out", "module": unet.out})
    return stages


def move_optional_tensor(value, device):
    """Move tensors across shard boundaries while leaving non-tensors unchanged."""
    return value.to(device) if isinstance(value, torch.Tensor) and value.device != torch.device(device) else value


def create_sharded_forward(unet, stage_devices: Dict[str, str], primary_device: str):
    """Create a VideoUNet forward that follows the generated device map."""

    def sharded_forward(
        x,
        timesteps,
        context=None,
        y=None,
        time_context=None,
        num_video_frames=None,
        image_only_indicator=None,
    ):
        assert (y is not None) == (
            unet.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"

        time_device = stage_devices["time_embed"]
        x = move_optional_tensor(x, time_device)
        timesteps = move_optional_tensor(timesteps, time_device)
        y = move_optional_tensor(y, time_device)

        emb = unet.time_embed(timestep_embedding(timesteps, unet.model_channels, repeat_only=False))
        if unet.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + unet.label_emb(y)

        h = x
        skip_connections = []
        for idx, module in enumerate(unet.input_blocks):
            device = stage_devices[f"input_blocks.{idx}"]
            h = move_optional_tensor(h, device)
            h = module(
                h,
                move_optional_tensor(emb, device),
                context=move_optional_tensor(context, device),
                image_only_indicator=move_optional_tensor(image_only_indicator, device),
                time_context=move_optional_tensor(time_context, device),
                num_video_frames=num_video_frames,
            )
            skip_connections.append(h)

        middle_device = stage_devices["middle_block"]
        h = move_optional_tensor(h, middle_device)
        h = unet.middle_block(
            h,
            move_optional_tensor(emb, middle_device),
            context=move_optional_tensor(context, middle_device),
            image_only_indicator=move_optional_tensor(image_only_indicator, middle_device),
            time_context=move_optional_tensor(time_context, middle_device),
            num_video_frames=num_video_frames,
        )

        for idx, module in enumerate(unet.output_blocks):
            device = stage_devices[f"output_blocks.{idx}"]
            h = move_optional_tensor(h, device)
            skip = move_optional_tensor(skip_connections.pop(), device)
            h = torch.cat([h, skip], dim=1)
            h = module(
                h,
                move_optional_tensor(emb, device),
                context=move_optional_tensor(context, device),
                image_only_indicator=move_optional_tensor(image_only_indicator, device),
                time_context=move_optional_tensor(time_context, device),
                num_video_frames=num_video_frames,
            )

        out_device = stage_devices["out"]
        h = move_optional_tensor(h, out_device)
        return move_optional_tensor(unet.out(h.type(x.dtype)), primary_device)

    return sharded_forward


def checkpoint_block_window(num_blocks: int, block_indices: Sequence[int], radius: int = CHECKPOINT_BLOCK_RADIUS) -> List[int]:
    """Return input block indices whose checkpoints should be disabled around attacked blocks."""
    selected = set()
    for block_idx in block_indices:
        start = max(0, block_idx - radius)
        stop = min(num_blocks, block_idx + radius + 1)
        selected.update(range(start, stop))
    return sorted(selected)


def disable_checkpoints_near_blocks(model, block_indices: Sequence[int], radius: int = CHECKPOINT_BLOCK_RADIUS) -> List[int]:
    """Disable checkpointing near attacked blocks so attention tensors keep gradients."""
    unet = model.model.diffusion_model
    disabled_blocks = checkpoint_block_window(len(unet.input_blocks), block_indices, radius)
    disabled_count = 0

    for block_idx in disabled_blocks:
        for module in unet.input_blocks[block_idx].modules():
            if hasattr(module, "checkpoint") and module.checkpoint:
                module.checkpoint = False
                disabled_count += 1
            if hasattr(module, "use_checkpoint") and module.use_checkpoint:
                module.use_checkpoint = False
                disabled_count += 1

    print(f"  Disabled {disabled_count} checkpoint flag(s) in input_blocks{disabled_blocks}")
    return disabled_blocks


def find_attention_device(stage_devices: Dict[str, str], attack_blocks: Sequence[int]) -> str:
    """Return the device where the first attacked input block runs."""
    for block_idx in attack_blocks:
        name = f"input_blocks.{block_idx}"
        if name in stage_devices:
            return stage_devices[name]
    return next(iter(stage_devices.values()))


def setup_auto_sharded_model(
    model,
    devices: Optional[List[str]] = None,
    max_memory: Optional[Dict[str, int]] = None,
    attack_blocks: Sequence[int] = (5,),
):
    """Move model components to an automatic model-parallel layout."""
    devices = devices or parse_devices(None)
    if not devices:
        raise ValueError("No CUDA devices available")

    primary_device = devices[0]
    budgets = available_device_memory(devices, max_memory)
    unet = model.model.diffusion_model
    stages = build_unet_stages(unet)
    stage_sizes = [module_nbytes(stage["module"]) for stage in stages]
    assigned_devices = choose_devices_for_stages(stage_sizes, devices, budgets)
    stage_devices = {stage["name"]: device for stage, device in zip(stages, assigned_devices)}

    model.conditioner.to(primary_device)
    model.first_stage_model.encoder.to(primary_device)

    decoder_device = assigned_devices[-1]
    model.first_stage_model.decoder.to(decoder_device)

    for stage in stages:
        stage["module"].to(stage_devices[stage["name"]])
    if hasattr(unet, "label_emb") and unet.label_emb is not None:
        unet.label_emb.to(stage_devices["time_embed"])

    disabled_checkpoint_blocks = disable_checkpoints_near_blocks(model, attack_blocks)
    attention_device = find_attention_device(stage_devices, attack_blocks)

    model_info = {
        "devices": devices,
        "primary_device": primary_device,
        "decoder_device": decoder_device,
        "attention_device": attention_device,
        "stage_devices": stage_devices,
        "device_budgets": budgets,
        "attack_blocks": tuple(attack_blocks),
        "disabled_checkpoint_blocks": disabled_checkpoint_blocks,
        "auto_sharded": True,
    }

    print("=" * 60)
    print("Setting up automatic model-parallel sharding")
    print(f"  Devices: {', '.join(devices)}")
    print(f"  Primary device: {primary_device}")
    print(f"  VAE decoder device: {decoder_device}")
    print(f"  Attack blocks: {list(attack_blocks)}")
    print("  UNet stage placement:")
    for stage, size in zip(stages, stage_sizes):
        print(f"    - {stage['name']:<18} -> {stage_devices[stage['name']]} ({size / 1024**2:.1f} MiB)")

    unet._original_forward = unet.forward
    unet.forward = create_sharded_forward(unet, stage_devices, primary_device)
    print("\nAutomatic sharded setup complete!")
    print("=" * 60)
    return model_info
