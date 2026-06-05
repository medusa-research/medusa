"""Temporal attention capture for the Medusa attack."""

import torch
import torch.nn.functional as F
from einops import rearrange

DEFAULT_ATTACK_BLOCKS = (5,)


class TemporalAttentionHook:
    """Capture temporal self-attention matrices and compute their nuclear norm."""

    def __init__(self, blocks=DEFAULT_ATTACK_BLOCKS):
        self.blocks = tuple(blocks)
        self.attention_weights = []
        self.hooks = []
        self.q_outputs = {}
        self.k_outputs = {}
        self.attn_modules = {}

    def clear(self):
        """Clear attention tensors captured during the previous forward pass."""
        self.attention_weights = []
        self.q_outputs = {}
        self.k_outputs = {}

    def register_hooks(self, model):
        """Register Q/K hooks on the configured temporal attention input blocks."""
        from sgm.modules.video_attention import VideoTransformerBlock

        unet = model.model.diffusion_model
        attn_count = 0

        for block_idx in self.blocks:
            if block_idx >= len(unet.input_blocks):
                continue

            for module in unet.input_blocks[block_idx].modules():
                if module.__class__.__name__ != "SpatialVideoTransformer":
                    continue
                if not hasattr(module, "time_stack"):
                    continue

                for time_module in module.time_stack:
                    if not isinstance(time_module, VideoTransformerBlock):
                        continue
                    if not hasattr(time_module, "attn1") or time_module.attn1 is None:
                        continue

                    attn_module = time_module.attn1
                    if attn_module.__class__.__name__ != "MemoryEfficientCrossAttention":
                        continue

                    attn_id = f"block{block_idx}_attn{attn_count}"
                    self.attn_modules[attn_id] = attn_module

                    def make_q_hook(aid, attn_mod):
                        def hook_fn(module, inputs, output):
                            heads = attn_mod.heads
                            self.q_outputs[aid] = rearrange(output, "b n (h d) -> b h n d", h=heads)

                        return hook_fn

                    def make_k_hook(aid, attn_mod):
                        def hook_fn(module, inputs, output):
                            heads = attn_mod.heads
                            k = rearrange(output, "b n (h d) -> b h n d", h=heads)
                            self.k_outputs[aid] = k

                            if aid not in self.q_outputs:
                                return

                            q = self.q_outputs[aid]
                            attn = torch.matmul(q, k.transpose(-2, -1)) * (attn_mod.dim_head ** -0.5)
                            attn_weights = F.softmax(attn, dim=-1)
                            if attn_weights.requires_grad:
                                attn_weights.retain_grad()
                            self.attention_weights.append(attn_weights)

                        return hook_fn

                    self.hooks.append(attn_module.to_q.register_forward_hook(make_q_hook(attn_id, attn_module)))
                    self.hooks.append(attn_module.to_k.register_forward_hook(make_k_hook(attn_id, attn_module)))
                    attn_count += 1

        if attn_count == 0:
            print("  Warning: no temporal attention hooks registered")
        else:
            print(f"  Registered {attn_count} temporal attention hook(s) at input_blocks{list(self.blocks)}")

        return attn_count

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_nuclear_loss(self, device):
        """Return the mean nuclear norm of captured conditional temporal attention."""
        if not self.attention_weights:
            return torch.tensor(0.0, device=device, requires_grad=True)

        losses = []
        for attn_weights in self.attention_weights:
            batch_size = attn_weights.shape[0] // 2
            conditional_attention = attn_weights[batch_size:]
            batch, heads, seq_len, _ = conditional_attention.shape
            attn_2d = conditional_attention.reshape(batch * heads, seq_len, seq_len)
            losses.append(torch.linalg.svdvals(attn_2d).sum() / (batch * heads))

        return torch.stack(losses).mean()
