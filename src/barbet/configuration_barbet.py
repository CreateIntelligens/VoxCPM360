"""Configuration for Barbet models.

Barbet mirrors the Open Formosa R2 architecture (Taiwan-Omni R2): a hybrid
decoder with a repeating ``global, sliding, sliding, mamba2`` block motif,
tied embedding/LM-head, and the frozen openformosa/PangolinTokenizer vocabulary.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

# Canonical PangolinTokenizer contract (mirrors open_formosa special_tokens.py).
# The base BPE vocab is 114688, special tokens run to id 114821 (effective
# 114822), and the Megatron-padded embedding size is 114944 (multiple of 128).
BASE_VOCAB_SIZE = 114688
EFFECTIVE_VOCAB_SIZE = 114822
MEGATRON_PADDED_VOCAB_SIZE = 114944

UNK_TOKEN_ID = 114688
BOS_TOKEN_ID = 114689
EOS_TOKEN_ID = 114690
PAD_TOKEN_ID = 114691

ROPE_SCALING_TYPES = ("none", "linear", "yarn", "longrope")


class BarbetConfig(PretrainedConfig):
    """Configuration class for Barbet decoder-only causal language models."""

    model_type = "barbet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = MEGATRON_PADDED_VOCAB_SIZE,
        hidden_size: int = 1536,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        max_position_embeddings: int = 262144,
        rope_theta: float = 10000000.0,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window_size: int = 8192,
        global_attention_layers: list[int] | tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24),
        mamba_layers: list[int] | tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27),
        qk_norm: bool = True,
        qk_logit_clip: bool = False,
        qk_clip_alpha: float = 0.5,
        qk_clip_threshold: float = 100.0,
        attention_sink: bool = False,
        rms_norm_eps: float = 1.0e-6,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mtp_enabled: bool = True,
        mtp_offsets: list[int] | tuple[int, ...] = (2, 3),
        mtp_loss_weights: dict[str, float] | dict[int, float] | None = None,
        pad_token_id: int | None = PAD_TOKEN_ID,
        bos_token_id: int | None = BOS_TOKEN_ID,
        eos_token_id: int | None = EOS_TOKEN_ID,
        unk_token_id: int | None = UNK_TOKEN_ID,
        **kwargs: Any,
    ) -> None:
        rope_scaling = self._normalize_rope_scaling(rope_scaling, kwargs.pop("rope_parameters", None))
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window_size = sliding_window_size
        self.global_attention_layers = [int(layer) for layer in global_attention_layers]
        self.mamba_layers = [int(layer) for layer in mamba_layers]
        self.qk_norm = qk_norm
        self.qk_logit_clip = qk_logit_clip
        self.qk_clip_alpha = qk_clip_alpha
        self.qk_clip_threshold = qk_clip_threshold
        self.attention_sink = attention_sink
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mtp_enabled = mtp_enabled
        self.mtp_offsets = [int(offset) for offset in mtp_offsets]
        weights = mtp_loss_weights or {2: 0.2, 3: 0.1}
        self.mtp_loss_weights = {str(int(k)): float(v) for k, v in dict(weights).items()}
        self.unk_token_id = unk_token_id
        self._validate()

    @classmethod
    def barbet_1b(
        cls,
        vocab_size: int = MEGATRON_PADDED_VOCAB_SIZE,
        max_position_embeddings: int = 262144,
    ) -> "BarbetConfig":
        """Taiwan-Omni-1B-R2 shape: tied vocab tables fund a 28-layer body."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=1536,
            intermediate_size=5120,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=8192,
            global_attention_layers=(0, 4, 8, 12, 16, 20, 24),
            mamba_layers=(3, 7, 11, 15, 19, 23, 27),
            qk_logit_clip=False,
            attention_sink=False,
            tie_word_embeddings=True,
        )

    @classmethod
    def barbet_1b_1m_extension(cls) -> "BarbetConfig":
        """1M research extension: linear RoPE scaling x4 from the 256K base."""
        config = cls.barbet_1b(max_position_embeddings=1048576)
        config.rope_scaling = {
            "type": "linear",
            "factor": 4.0,
            "original_context_length": 262144,
        }
        return config

    def layer_type(self, layer_idx: int) -> str:
        if layer_idx in self.global_attention_layers:
            return "global_attention"
        if layer_idx in self.mamba_layers:
            return "mamba"
        return "sliding_attention"

    def _validate(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_attention_heads * self.head_dim < self.hidden_size:
            raise ValueError("num_attention_heads * head_dim must cover hidden_size")
        layer_ids = set(range(self.num_hidden_layers))
        scheduled = set(self.global_attention_layers) | set(self.mamba_layers)
        invalid = scheduled - layer_ids
        if invalid:
            raise ValueError(f"scheduled layers out of range: {sorted(invalid)}")
        overlap = set(self.global_attention_layers) & set(self.mamba_layers)
        if overlap:
            raise ValueError(f"layers cannot be both attention and mamba: {sorted(overlap)}")
        if self.rope_scaling:
            scaling_type = str(self.rope_scaling.get("type", "linear"))
            if scaling_type not in ROPE_SCALING_TYPES:
                raise ValueError(
                    f"rope_scaling.type must be one of {ROPE_SCALING_TYPES}, got {scaling_type!r}"
                )

    @staticmethod
    def _normalize_rope_scaling(
        rope_scaling: dict[str, Any] | None,
        rope_parameters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        scaling = rope_scaling if rope_scaling is not None else rope_parameters
        if not scaling:
            return None
        normalized = dict(scaling)
        rope_type = normalized.pop("rope_type", None)
        if "type" not in normalized and rope_type is not None:
            normalized["type"] = rope_type
        normalized.setdefault("type", "linear")
        return normalized


BarbetConfig.register_for_auto_class()
