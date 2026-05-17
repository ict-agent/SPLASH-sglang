from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape


class GlmLinearConfig(PretrainedConfig):
    model_type = "glm4_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="glm4_moe",
        vocab_size=154880,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=12288,
        num_hidden_layers=45,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        rms_norm_eps=1e-05,
        pad_token_id=151329,
        bos_token_id=None,
        eos_token_id=None,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4196,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        scoring_func: str = "sigmoid",
        n_routed_experts: int | None = None,
        num_experts_per_token: int | None = None,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 2.5,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        mla: bool = True,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_nope: bool | None = True,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_dsa_use_layernorm: bool = True,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_allow_neg_eigval: bool | None = False,
        mhc: bool | None = True,
        mhc_num_residual_streams: int = 4,
        hc_eps: float | None = 1e-06,
        mhc_tau: float = 0.05,
        hres_vwnstyle: bool | None = True,
        mhc_no_norm_weight: bool | None = False,
        mhc_sinkhorn_iterations: int | None = 20,
        mhc_post_mult_value: float | None = 2.0,
        swiglu_clamp_limit: float | None = None,
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
        self.num_experts_per_token = num_experts_per_token
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

        # mhc config
        self.mhc = mhc
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_tau = mhc_tau
        self.hres_vwnstyle = hres_vwnstyle
        self.hc_eps = hc_eps
        self.mhc_no_norm_weight = mhc_no_norm_weight
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_post_mult_value = float(mhc_post_mult_value)

        self.swiglu_clamp_limit = swiglu_clamp_limit

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
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = KimiLinearStateShape.create(
            tp_world_size=get_attention_tp_size(),
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
        )

        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)
