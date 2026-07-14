import logging

from transformers.configuration_utils import PretrainedConfig
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape

logger = logging.getLogger(__name__)


class Glm5NextVisionConfig(Glm4vVisionConfig):
    model_type = "glm5next_vision"


class Glm5NextConfig(PretrainedConfig):
    r"""Config for GLM-5 Next; flat for text-only, nested (text_config + vision_config) for VLM."""

    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    # Wrapper-owned / identity fields that must not be promoted from text_config.
    _NO_PROMOTE = frozenset(
        {
            "model_type",
            "architectures",
            "text_config",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "image_start_token_id",
            "image_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
        }
    )

    def __init__(
        self,
        model_type: str | None = "glm5_next",
        vocab_size: int | None = 154880,
        hidden_size: int | None = 4096,
        head_dim: int | None = None,
        intermediate_size: int | None = 12288,
        num_hidden_layers: int | None = 45,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        rms_norm_eps: float | None = 1e-05,
        pad_token_id: int | None = 151329,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        rope_theta: float | None = 10000.0,
        rope_scaling: dict | None = None,
        max_position_embeddings: int | None = 4196,
        tie_word_embeddings: bool | None = False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool | None = True,
        scoring_func: str | None = "sigmoid",
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        n_shared_experts: int | None = 1,
        routed_scaling_factor: float | None = 2.5,
        first_k_dense_replace: int | None = 0,
        moe_layer_freq: int | None = 1,
        use_grouped_topk: bool | None = True,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        norm_topk_prob: bool | None = True,
        mla: bool | None = True,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_nope: bool | None = True,
        num_nextn_predict_layers: int | None = 0,
        linear_attn_config: dict | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_dsa_use_layernorm: bool | None = True,
        index_kpool: int | None = 1,
        index_kpool_compress: bool | None = False,
        index_kpool_always_select_tail: bool | None = False,
        linear_conv_kernel_dim: int | None = 4,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_allow_neg_eigval: bool | None = False,
        mhc: bool | None = False,
        hc_mult: int | None = 4,
        hc_eps: float | None = 1e-06,
        hc_sinkhorn_iters: int | None = 20,
        hc_post_mult_value: float | None = 2.0,
        swiglu_limit: float | None = None,
        # Vision-language wrapper fields (present only for VLM checkpoints).
        text_config: dict | None = None,
        vision_config: dict | None = None,
        image_token_id: int | None = 151363,
        video_token_id: int | None = 151364,
        image_start_token_id: int | None = 151339,
        image_end_token_id: int | None = 151340,
        video_start_token_id: int | None = 151341,
        video_end_token_id: int | None = 151342,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # mla config
        self.mla = mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_nope = mla_nope
        # moe config
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.moe_renormalize = moe_renormalize
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        assert self.scoring_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval
        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        # dsa index config
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_dsa_use_layernorm = index_dsa_use_layernorm
        # Downstream uses ``index_kpool > 1`` as the single
        # "kpool enabled" condition; collapse the degenerate combo here.
        if index_kpool > 1 and not index_kpool_compress:
            logger.warning(
                "index_kpool=%d with compress=False -> disable kpool.",
                index_kpool,
            )
            index_kpool = 1
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        # mhc config
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_post_mult_value = (
            float(hc_post_mult_value) if hc_post_mult_value is not None else None
        )

        self.swiglu_limit = swiglu_limit

        # Only VLM checkpoints carry text_config/vision_config/token ids; text-only stays flat for multimodal detection.
        is_vlm = text_config is not None or vision_config is not None
        if is_vlm:
            # Promote nested text_config fields onto self for flat attribute access.
            if text_config is not None:
                text_conf_cls = self.sub_configs["text_config"]
                if isinstance(text_config, text_conf_cls):
                    self.text_config = text_config
                elif isinstance(text_config, PretrainedConfig):
                    self.text_config = text_conf_cls(**text_config.to_dict())
                else:
                    self.text_config = text_conf_cls(**text_config)
                for k, v in self.text_config.__dict__.items():
                    if k.startswith("_") or k in self._NO_PROMOTE:
                        continue
                    setattr(self, k, v)

                # For VLM the real token ids live in text_config, but the
                # top-level args usually stay at their defaults. Pull them from
                # text_config so super().__init__ below doesn't clobber the
                # promoted values with those defaults.
                pad_token_id = getattr(self.text_config, "pad_token_id", pad_token_id)
                bos_token_id = getattr(self.text_config, "bos_token_id", bos_token_id)
                eos_token_id = getattr(self.text_config, "eos_token_id", eos_token_id)
                tie_word_embeddings = getattr(
                    self.text_config, "tie_word_embeddings", tie_word_embeddings
                )

            if isinstance(vision_config, dict):
                self.vision_config = self.sub_configs["vision_config"](**vision_config)
            else:
                self.vision_config = vision_config

            # Multimodal token ids.
            self.image_token_id = image_token_id
            self.video_token_id = video_token_id
            self.video_start_token_id = video_start_token_id
            self.video_end_token_id = video_end_token_id
            self.image_start_token_id = image_start_token_id
            self.image_end_token_id = image_end_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_nope is True
        )

    @property
    def is_moe(self):
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and layer_idx in self.linear_attn_config["kda_layers"]
        )

    @property
    def linear_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_size,
            get_attention_tp_size,
        )

        # Shard mamba state by the same factor KDA shards heads (CP under NSA prefill CP, else TP).
        head_shard_size = (
            get_attention_cp_size()
            if is_nsa_enable_prefill_cp()
            else get_attention_tp_size()
        )

        shape = KimiLinearStateShape.create(
            tp_world_size=head_shard_size,
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
        )

        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)


class Glm5NextTextConfig(Glm5NextConfig):
    model_type = "glm5next_text"


# Registered after definition: Glm5NextTextConfig subclasses Glm5NextConfig, so it can't be referenced inside the class body above.
Glm5NextConfig.sub_configs["text_config"] = Glm5NextTextConfig
