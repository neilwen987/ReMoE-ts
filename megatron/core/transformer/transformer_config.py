# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch.nn.functional as F

from megatron.core.transformer.enums import AttnBackend

from ..model_parallel_config import ModelParallelConfig
from ..utils import get_te_version, init_method_normal, is_te_min_version, scaled_init_method_normal


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    ####################
    # model architecture
    ####################
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    first_pipeline_num_layers: int = None
    """Number of transformer layers on first pipeline stage. 
    None implies equal layer division across PP ranks."""

    last_pipeline_num_layers: int = None
    """Number of transformer layers on last pipeline stage. 
    None implies equal layer division across PP ranks."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    attention_backend: AttnBackend = AttnBackend.auto
    """Attention backend to run. By default we let transformer engine
    decide the best backend to run (except in the case of local).
    If attention backend is local we use the local pytorch implementation in mcore. 
    Users can specify exact backend by changing this config. """

    num_query_groups: int = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: int = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    kv_channels: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop compuatation."""

    num_moe_experts: int = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    normalization: bool = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply LayerNorm to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    ####################
    # initialization
    ####################
    init_method: Callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: bool = None
    """If True, distribute recomputed activations across the model parallel group."""

    ####################
    # fp8 related
    ####################
    fp8: str = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
    Controls how often the scaling factor is recomputed.
    """

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation
    in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    tp_only_amax_red: bool = False
    """When set to True, reduce the FP8 AMAX only in the TP or TP-CP domain"""

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: int = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared epxerts execute after the routed experts."""

    moe_layer_freq: int = 1
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A string containing a Python list expression that defines a custom pattern, e.g.:
    "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
    where 1 indicates an expert layer and 0 indicates a dense layer."""

    moe_ffn_hidden_size: int = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. 
    By default, softmax is done after top-k."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_use_legacy_grouped_gemm: bool = False
    """Use legacy GroupedMLP rather than TEGroupedMLP.
    Note: The legacy one will be deprecated soon."""

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: float = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: float = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False  # TODO: Support token dropping.
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather' and 'alltoall'."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: float = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    moe_granularity: int = 1
    """Granularity of fine-grained MoE. Please refer to https://arxiv.org/abs/2402.07871 for more details."""

    moe_sample_routing: bool = False
    """Use Sample Top-k routing for MoE."""
    
    moe_relu_routing: bool = False
    """Use ReLU as the routing function for MoE."""

    moe_relu_l1_reg_coeff_init: float = 1e-8
    """Initial value of L1 regularization coefficient for ReLU routing (\lambda_0 in the paper)."""

    moe_relu_l1_reg_coeff_multiplier: float = 1.2
    """Multiplier for the L1 regularization coefficient for ReLU routing (\alpha in the paper)."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: Union[str, List[str]] = None
    """Inter-gpu communication type for context parallelism.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    cp_comm_type of each layer can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
    "p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
    overlapped with attention compute.
    "all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
    async, and cannot be overlapped.
    "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
    full sequence of QKV.
    "a2a+p2p": A hierarchical implementation of context parallelism to attention. 
    It uses A2A communications in low-level CP groups (e.g., via NVLink),
    and P2P communications in high-level CP groups (e.g., via IBLink).
    """

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with a CUDA graphed version."""

    external_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with user provided CUDA graphs."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    flash_decode: bool = False
    """ Use the optimized flash decoding kernel during inference. """

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f'moe_shared_expert_intermediate_size must be '
                    f'num_shared_experts * ffn_size_of_each_shared_expert, '
                    f'but got {self.moe_shared_expert_intermediate_size}'
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall"
            ]:
                raise ValueError(
                    f'moe_shared_expert_overlap only works with alltoall token dispatcher.'
                )

        if self.moe_expert_capacity_factor is not None:
            if self.moe_token_dispatcher_type not in ["alltoall", "alltoall_seq"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full"'
                    'or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so '
                    'recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be between '
                    '1 and num_layers_per_pipeline_rank: '
                    f'{self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be '
                    f'false when sequence parallel is enabled: {self.sequence_parallel}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by '
                        f'virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either "
                    "gelu or swiglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )

        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")

        if self.apply_rope_fusion:
            if self.rotary_interleaved:
                raise ValueError("rotary_interleaved does not work with apply_rope_fusion.")

            from megatron.core.models.common.embeddings.rope_utils import HAVE_APPLY_ROPE_FUSION

            if not HAVE_APPLY_ROPE_FUSION:
                raise ValueError(
                    "apply_rope_fusion is not available. Please install TE >= 1.4 or Apex."
                )

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )

        if (
            self.moe_token_dispatcher_type == "alltoall_seq"
            and self.tensor_model_parallel_size != self.expert_tensor_parallel_size
        ):
            raise ValueError(
                "alltoall_seq dispatcher not support different TP size for MoE and Dense layer."
            )

        if self.num_moe_experts and self.fp8:
            # TE version below 1.7.0 will raise Error when handle zeros tokens for expert
            if not is_te_min_version("1.7.0.dev0"):
                raise ValueError(
                    "Only transformer-engine>=1.7.0 supports MoE FP8 training, "
                    f"but your version is {get_te_version()}."
                )

            if self.moe_grouped_gemm and not is_te_min_version("1.11.0"):
                raise ValueError(
                    "Only transformer-engine>=1.11.0 supports FP8 grouped gemm, "
                    f"but your version is {get_te_version()}."
                )

        if self.flash_decode and self.fp8:
            raise ValueError("FP8 inference is currently not support with flash decoding.")

        if self.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
            if self.variable_seq_lengths is True:
                raise ValueError(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, please use alltoall dispatcher instead."
                )

        if self.cp_comm_type is not None:
            if isinstance(self.cp_comm_type, list):
                assert len(self.cp_comm_type) == self.num_layers, (
                    f"Length of cp_comm_type ({len(self.cp_comm_type)}) should equal to "
                    f"the total number of transformer layers ({self.num_layers})!"
                )
            else:
                assert isinstance(
                    self.cp_comm_type, str
                ), "Unsupported communication type for context parallelism!"

        if self.moe_granularity > 1 and self.num_moe_experts:
            if self.moe_ffn_hidden_size % self.moe_granularity != 0:
                raise ValueError(
                    f"moe_ffn_hidden_size ({self.moe_ffn_hidden_size}) must be divisible by "
                    f"moe_granularity ({self.moe_granularity})."
                )
            print(f"Using fine-grained MoE with granularity: {self.moe_granularity}, "
                    f"increase num_moe_experts to {self.num_moe_experts * self.moe_granularity}, "
                    f"increase moe_router_topk to {self.moe_router_topk * self.moe_granularity}, "
                    f"decrease moe_ffn_hidden_size to {self.moe_ffn_hidden_size // self.moe_granularity}.")
            self.num_moe_experts *= self.moe_granularity
            self.moe_router_topk *= self.moe_granularity
            self.moe_ffn_hidden_size //= self.moe_granularity


@dataclass
class MLATransformerConfig(TransformerConfig):
    """Configuration object for megatron-core Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    multi_latent_attention: bool = True
    """Whether to use Multi-Latent Attention."""

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings."""

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    max_position_embeddings: int = 163840
    """Maximum position embeddings for the original model."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE."""

    mscale: float = 0.707
    """Mscale for YaRN RoPE in Multi-Latent Attention."""

    mscale_all_dim: float = 0.707
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention."""
