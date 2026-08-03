"""PyTorch modeling code for Barbet.

Mirrors the Open Formosa R2 reference architecture: hybrid global/sliding
attention + Mamba-style mixer blocks, SwiGLU MLPs, QK RMSNorm, tied
embedding/LM head, and an optional multi-token prediction training loss.

Incremental decoding uses :class:`BarbetCache`, a hybrid cache holding
attention K/V states (a rolling window for sliding layers) and the causal-conv
tail state for Mamba-style layers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_barbet import BarbetConfig

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as MambaRMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except Exception:
    MambaRMSNormGated = None
    mamba_chunk_scan_combined = None


class BarbetCache:
    """Hybrid per-layer cache for incremental decoding.

    Attention layers store un-repeated GQA key/value states; sliding-window
    layers keep only the most recent ``sliding_window_size`` positions. Mamba
    layers store the trailing ``d_conv - 1`` causal-conv inputs. Duck-types the
    parts of the transformers ``Cache`` interface that ``generate()`` touches
    for stateful models (``get_seq_length`` and ``reorder_cache``).
    """

    def __init__(self, config: BarbetConfig) -> None:
        self.sliding_window_size = config.sliding_window_size
        num_layers = config.num_hidden_layers
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self.conv_cache: list[torch.Tensor | None] = [None] * num_layers
        self.ssm_cache: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def update_attention(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sliding: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is not None:
            key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        # Return the full states for the current block (early queries in a
        # prefill chunk still need keys beyond the window tail); store only the
        # rolling window for future steps.
        if sliding and self.sliding_window_size > 0 and key_states.shape[2] > self.sliding_window_size:
            self.key_cache[layer_idx] = key_states[:, :, -self.sliding_window_size :]
            self.value_cache[layer_idx] = value_states[:, :, -self.sliding_window_size :]
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def update_conv(self, layer_idx: int, conv_inputs: torch.Tensor, tail_len: int) -> torch.Tensor:
        """Prepend the cached conv tail (zeros initially) and store the new tail.

        ``conv_inputs`` has shape ``(batch, channels, seq_len)``; the returned
        tensor has ``tail_len`` extra leading positions so a valid (unpadded)
        causal conv produces exactly ``seq_len`` outputs.
        """
        previous = self.conv_cache[layer_idx]
        if previous is None:
            previous = conv_inputs.new_zeros(conv_inputs.shape[0], conv_inputs.shape[1], tail_len)
        full = torch.cat([previous, conv_inputs], dim=-1)
        self.conv_cache[layer_idx] = full[..., full.shape[-1] - tail_len :]
        return full

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for tensors in (self.key_cache, self.value_cache, self.conv_cache, self.ssm_cache):
            for idx, tensor in enumerate(tensors):
                if tensor is not None:
                    tensors[idx] = tensor.index_select(0, beam_idx.to(tensor.device))


class BarbetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return hidden_states.to(dtype=self.weight.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class BarbetRotaryEmbedding(nn.Module):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.dim = config.head_dim
        self.base = config.rope_theta
        # Only linear position scaling has a reference implementation upstream;
        # yarn/longrope entries are config metadata for external runtimes.
        scale = None
        if config.rope_scaling:
            scaling_type = str(config.rope_scaling.get("type", "linear")).lower()
            factor = config.rope_scaling.get("factor")
            if scaling_type == "linear" and factor:
                scale = float(factor)
        self.scale = scale

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.float()
        if self.scale and self.scale > 1.0:
            positions = positions / self.scale
        # Computed per call instead of a non-persistent buffer: meta-device
        # checkpoint loading materializes such buffers uninitialized.
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device) / self.dim)
        )
        freqs = torch.einsum("bs,d->bsd", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[:, None, :, :]) + (rotate_half(x) * sin[:, None, :, :])


class BarbetAttention(nn.Module):
    def __init__(self, config: BarbetConfig, layer_idx: int, sliding_window: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window_size = config.sliding_window_size if sliding_window else None
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = BarbetRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = BarbetRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.rotary_emb = BarbetRotaryEmbedding(config)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads)) if config.attention_sink else None

    def _shape(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _attention_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        total = past_len + q_len
        q_pos = torch.arange(past_len, total, device=device)[:, None]
        # Cached keys are the most recent kv_len positions, in order.
        k_pos = torch.arange(total - kv_len, total, device=device)[None, :]
        mask = k_pos <= q_pos
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            mask &= k_pos >= (q_pos - self.sliding_window_size + 1)
        mask = mask.view(1, 1, q_len, kv_len).expand(batch_size, 1, q_len, kv_len)
        if attention_mask is not None:
            if attention_mask.shape[1] < total:
                # Mask covers only the newest tokens; treat older history as visible.
                pad = attention_mask.new_ones(batch_size, total - attention_mask.shape[1])
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
            key_mask = attention_mask[:, -kv_len:][:, None, None, :].bool()
            mask = mask & key_mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BarbetCache | None = None,
        past_len: int = 0,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        query_states = self._shape(self.q_proj(hidden_states), self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value_states = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = self.rotary_emb(position_ids, query_states.dtype)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update_attention(
                self.layer_idx, key_states, value_states, sliding=self.sliding_window_size is not None
            )

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        kv_len = key_states.shape[2]

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.config.qk_logit_clip:
            threshold = float(self.config.qk_clip_threshold)
            attn_weights = threshold * torch.tanh(attn_weights / threshold)

        allowed = self._attention_mask(
            batch_size, seq_len, kv_len, past_len, attention_mask, hidden_states.device
        )
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~allowed, min_value)

        if self.sink_logits is None:
            softmax_input = attn_weights if attn_weights.is_cuda else attn_weights.float()
            attn_probs = torch.softmax(softmax_input, dim=-1).to(query_states.dtype)
        else:
            sink = self.sink_logits.view(1, self.num_heads, 1, 1).float()
            max_score = torch.maximum(attn_weights.float().max(dim=-1, keepdim=True).values, sink)
            real_exp = torch.exp(attn_weights.float() - max_score)
            sink_exp = torch.exp(sink - max_score)
            attn_probs = (real_exp / (real_exp.sum(dim=-1, keepdim=True) + sink_exp)).to(query_states.dtype)

        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), attn_probs if output_attentions else None


class BarbetMambaMixer(nn.Module):
    """Megatron Mamba2-compatible mixer with a PyTorch selective-scan path."""

    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.inner_size = config.hidden_size * config.mamba_expand
        self.d_state = max(config.mamba_d_state, 1)
        self.d_conv = config.mamba_d_conv
        self.head_dim = config.head_dim
        self.num_heads = self.inner_size // self.head_dim
        self.num_groups = config.num_key_value_heads
        if self.inner_size % self.head_dim != 0:
            raise ValueError("mamba inner size must be divisible by head_dim")
        if self.num_heads % self.num_groups != 0:
            raise ValueError("mamba heads must be divisible by mamba groups")
        self.group_size = self.inner_size // self.num_groups

        self.in_proj_z = nn.Linear(config.hidden_size, self.inner_size, bias=False)
        self.in_proj_x = nn.Linear(config.hidden_size, self.inner_size, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_groups * self.d_state, bias=False)
        self.in_proj_c = nn.Linear(config.hidden_size, self.num_groups * self.d_state, bias=False)
        self.in_proj_dt = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.conv_x = nn.Conv1d(
            self.inner_size,
            self.inner_size,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.inner_size,
        )
        self.conv_b = nn.Conv1d(
            self.num_groups * self.d_state,
            self.num_groups * self.d_state,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.num_groups * self.d_state,
        )
        self.conv_c = nn.Conv1d(
            self.num_groups * self.d_state,
            self.num_groups * self.d_state,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.num_groups * self.d_state,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        if MambaRMSNormGated is not None:
            self.norm = MambaRMSNormGated(
                self.inner_size,
                eps=1.0e-5,
                group_size=self.group_size,
                norm_before_gate=False,
            )
        else:
            self.norm = BarbetRMSNorm(self.inner_size, eps=1.0e-5)
        self.out_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

    def _conv_full(self, conv: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        seq_len = values.shape[1]
        values = values.transpose(1, 2)
        values = conv(values)[..., :seq_len]
        return F.silu(values.transpose(1, 2))

    def _rmsnorm_gated(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if MambaRMSNormGated is not None and isinstance(self.norm, MambaRMSNormGated):
            return self.norm(hidden_states, gate)
        hidden_states = hidden_states * F.silu(gate)
        shape = hidden_states.shape
        grouped = hidden_states.view(*shape[:-1], self.num_groups, self.group_size)
        variance = grouped.float().pow(2).mean(dim=-1, keepdim=True)
        grouped = grouped.float() * torch.rsqrt(variance + self.norm.eps)
        weight = self.norm.weight.view(1, 1, self.num_groups, self.group_size)
        return (grouped.to(dtype=self.norm.weight.dtype) * weight).view(shape)

    def _selective_scan(
        self,
        x: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        dt: torch.Tensor,
        z: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b_proj = b_proj.view(batch_size, seq_len, self.num_groups, self.d_state)
        c_proj = c_proj.view(batch_size, seq_len, self.num_groups, self.d_state)
        z = z.view(batch_size, seq_len, self.num_heads, self.head_dim)

        state = initial_state
        if state is None:
            state = x.new_zeros(batch_size, self.num_heads, self.head_dim, self.d_state)
        else:
            state = state.to(dtype=dtype)

        heads_per_group = self.num_heads // self.num_groups
        group_for_head = torch.arange(self.num_heads, device=x.device) // heads_per_group
        a = -torch.exp(self.A_log.float()).to(dtype=dtype)
        d = self.D.to(dtype=dtype)
        dt_bias = self.dt_bias.to(dtype=dtype)
        outputs: list[torch.Tensor] = []
        for pos in range(seq_len):
            dt_pos = F.softplus(dt[:, pos] + dt_bias)
            d_a = torch.exp(dt_pos * a)
            b_pos = b_proj[:, pos].index_select(1, group_for_head)
            c_pos = c_proj[:, pos].index_select(1, group_for_head)
            x_pos = x[:, pos]
            state = state * d_a[:, :, None, None] + (
                dt_pos[:, :, None, None] * b_pos[:, :, None, :] * x_pos[:, :, :, None]
            )
            y = (state * c_pos[:, :, None, :]).sum(dim=-1)
            y = y + d[None, :, None] * x_pos
            outputs.append(y.reshape(batch_size, self.inner_size))
        y = torch.stack(outputs, dim=1)
        return self._rmsnorm_gated(y, z.reshape(batch_size, seq_len, self.inner_size)), state

    def _selective_scan_kernel(
        self,
        x: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        dt: torch.Tensor,
        z: torch.Tensor,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mamba_chunk_scan_combined is None or not x.is_cuda:
            raise RuntimeError("mamba_ssm selective-scan kernel is unavailable")
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        b_proj = b_proj.view(batch_size, seq_len, self.num_groups, self.d_state).contiguous()
        c_proj = c_proj.view(batch_size, seq_len, self.num_groups, self.d_state).contiguous()
        y = mamba_chunk_scan_combined(
            x,
            dt.contiguous(),
            -torch.exp(self.A_log.float()),
            b_proj,
            c_proj,
            chunk_size=128,
            D=self.D,
            z=None,
            dt_bias=self.dt_bias.float(),
            dt_softplus=True,
            return_final_states=return_final_state,
        )
        final_state = None
        if return_final_state:
            y, final_state = y
        y = y.reshape(batch_size, seq_len, self.inner_size)
        return self._rmsnorm_gated(y, z), final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: BarbetCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        z = self.in_proj_z(hidden_states)
        x = self.in_proj_x(hidden_states)
        b_proj = self.in_proj_b(hidden_states)
        c_proj = self.in_proj_c(hidden_states)
        dt = self.in_proj_dt(hidden_states)

        conv_inputs = torch.cat([x, b_proj, c_proj], dim=-1)
        use_step_cache = (
            past_key_values is not None
            and past_key_values.conv_cache[layer_idx] is not None
            and past_key_values.ssm_cache[layer_idx] is not None
            and hidden_states.shape[1] == 1
        )
        if use_step_cache:
            conv_state = past_key_values.conv_cache[layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = conv_inputs[:, 0, :]
            weights = torch.cat([self.conv_x.weight, self.conv_b.weight, self.conv_c.weight], dim=0)
            bias = torch.cat([self.conv_x.bias, self.conv_b.bias, self.conv_c.bias], dim=0)
            conv_out = (conv_state * weights.squeeze(1)[None, :, :]).sum(dim=-1) + bias
            conv_out = F.silu(conv_out).unsqueeze(1).to(dtype=hidden_states.dtype)
            past_key_values.conv_cache[layer_idx] = conv_state
        else:
            x = self._conv_full(self.conv_x, x)
            b_proj = self._conv_full(self.conv_b, b_proj)
            c_proj = self._conv_full(self.conv_c, c_proj)
            conv_out = torch.cat([x, b_proj, c_proj], dim=-1)
            if past_key_values is not None:
                padded = F.pad(conv_inputs.transpose(1, 2), (max(self.d_conv - conv_inputs.shape[1], 0), 0))
                past_key_values.conv_cache[layer_idx] = padded[..., -self.d_conv :]

        x, b_proj, c_proj = torch.split(
            conv_out,
            [self.inner_size, self.num_groups * self.d_state, self.num_groups * self.d_state],
            dim=-1,
        )
        if (
            mamba_chunk_scan_combined is not None
            and not use_step_cache
            and hidden_states.is_cuda
        ):
            y, final_state = self._selective_scan_kernel(
                x, b_proj, c_proj, dt, z, return_final_state=past_key_values is not None
            )
        else:
            initial_state = past_key_values.ssm_cache[layer_idx] if use_step_cache else None
            y, final_state = self._selective_scan(x, b_proj, c_proj, dt, z, initial_state=initial_state)
        if past_key_values is not None:
            past_key_values.ssm_cache[layer_idx] = final_state.detach()
        return self.out_proj(y)


class BarbetMLP(nn.Module):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class BarbetDecoderLayer(nn.Module):
    def __init__(self, config: BarbetConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.input_layernorm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.layer_type == "mamba":
            self.mixer = BarbetMambaMixer(config)
        else:
            self.mixer = BarbetAttention(config, layer_idx, sliding_window=self.layer_type == "sliding_attention")
        self.post_attention_layernorm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = BarbetMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BarbetCache | None = None,
        past_len: int = 0,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if isinstance(self.mixer, BarbetAttention):
            mixed, attn = self.mixer(
                normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                past_len=past_len,
                output_attentions=output_attentions,
            )
        else:
            mixed = self.mixer(normed, past_key_values=past_key_values, layer_idx=self.layer_idx)
            attn = None
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, attn


class BarbetPreTrainedModel(PreTrainedModel):
    config_class = BarbetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BarbetDecoderLayer"]

    def mark_tied_weights_as_initialized(self, loading_info: Any) -> None:
        # transformers >= 5 additionally drops declared tie targets from
        # missing_keys for remote-code models (a module-tying heuristic), which
        # then stops tie_weights() from re-tying lm_head after loading a
        # deduplicated checkpoint. Barbet only ties parameters explicitly via
        # _tied_weights_keys, so keep the init-skip flag and skip that cleanup.
        for tied_param in getattr(self, "all_tied_weights_keys", {}):
            self.get_parameter(tied_param)._is_hf_initialized = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class BarbetModel(BarbetPreTrainedModel):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__(config)
        padding_idx = config.pad_token_id
        if padding_idx is not None and padding_idx >= config.vocab_size:
            padding_idx = None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx)
        self.layers = nn.ModuleList([BarbetDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = BarbetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BarbetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape

        if use_cache and not isinstance(past_key_values, BarbetCache):
            past_key_values = BarbetCache(self.config)
        if not use_cache:
            past_key_values = None
        past_len = past_key_values.seen_tokens if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, past_len + seq_len, dtype=torch.bool, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                past_len=past_len,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions += (attn,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if past_key_values is not None:
            past_key_values.seen_tokens += seq_len

        if not return_dict:
            return tuple(
                v for v in (hidden_states, past_key_values, all_hidden_states, all_attentions) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class BarbetMTPHead(nn.Module):
    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.offsets = list(config.mtp_offsets)
        self.weights = {int(k): float(v) for k, v in config.mtp_loss_weights.items()}
        self.proj = nn.ModuleDict(
            {
                str(offset): nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.SiLU(),
                )
                for offset in self.offsets
            }
        )

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Linear) -> dict[int, torch.Tensor]:
        return {
            offset: F.linear(self.proj[str(offset)](hidden_states), lm_head.weight)
            for offset in self.offsets
        }


class BarbetForCausalLM(BarbetPreTrainedModel, GenerationMixin):
    # {target: source} mapping (transformers >= 5); 4.x ties via
    # get_output_embeddings() and only iterates these keys for bookkeeping.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # The hybrid cache (rolling sliding-window K/V + Mamba conv state) cannot
    # roll back, so generate() must not create a DynamicCache for this model.
    _is_stateful = True

    def __init__(self, config: BarbetConfig) -> None:
        super().__init__(config)
        self.model = BarbetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = BarbetMTPHead(config) if config.mtp_enabled else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def _shifted_loss(self, logits: torch.Tensor, labels: torch.Tensor, offset: int = 1) -> torch.Tensor:
        if offset <= 0:
            raise ValueError("offset must be positive")
        shifted_labels = torch.full_like(labels, -100)
        if offset < labels.shape[1]:
            shifted_labels[:, :-offset] = labels[:, offset:]
        valid = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~valid, 0)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        return (loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BarbetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self._shifted_loss(logits, labels, offset=1)
            if self.mtp is not None:
                for offset, mtp_logits in self.mtp(hidden_states, self.lm_head).items():
                    loss = loss + self.mtp.weights.get(offset, 1.0) * self._shifted_loss(
                        mtp_logits, labels, offset=offset
                    )

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: BarbetCache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        past_len = past_key_values.seen_tokens if isinstance(past_key_values, BarbetCache) else 0
        if past_len > 0:
            input_ids = input_ids[:, past_len:]
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1] :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache if use_cache is not None else True,
        }


BarbetModel.register_for_auto_class("AutoModel")
BarbetForCausalLM.register_for_auto_class("AutoModelForCausalLM")
