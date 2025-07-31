# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    sp_topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig

import contextlib

@contextlib.contextmanager
def temporary_eval_topk(model, eval_topk: int):
    """Temporarily set evaluation top-k for all MoE layers."""
    original_topks = []
    
    # 保存并设置新的top-k值
    for model_module in model:
        for layer in model_module.modules():
            if hasattr(layer, 'router') and hasattr(layer.router, 'set_eval_topk'):
                original_topks.append(layer.router.topk)
                layer.router.set_eval_topk(eval_topk)
    
    try:
        yield
    finally:
        # 恢复原始top-k值
        topk_idx = 0
        for model_module in model:
            for layer in model_module.modules():
                if hasattr(layer, 'router') and hasattr(layer.router, 'set_eval_topk'):
                    layer.router.set_eval_topk(original_topks[topk_idx])
                    topk_idx += 1


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        self.eval_topk = getattr(config, 'moe_router_eval_topk', self.topk)
    
    def get_current_topk(self):
        if self.training:
            return self.topk
        else:
            return self.eval_topk
    
    def set_eval_topk(self, eval_topk:int):
        self.eval_topk = eval_topk


    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """
        current_topk = self.get_current_topk()
        def _sinkhorn_activation(logits):
            if current_topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=current_topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=current_topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            indices (torch.Tensor): The mask of token to experts assignment.
        """
        current_topk = self.get_current_topk()
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            current_topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
        return probs, routing_map

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token.
                [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
                [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            current_topk = self.get_current_topk()
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                current_topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                deterministic_mode=self.config.deterministic_mode,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


class ReLURouter(Router):
    """Route each token to the experts with non-zero relu outputs."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the relu router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        # self.target_sparsity = 1 - self.topk / self.num_experts
        self.input_jitter = None

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def l1_reg_load_balancing(self, logits: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        probs = torch.relu(logits)
        routing_map = probs > 0
        if self.training and torch.is_grad_enabled():
            num_local_tokens_per_expert = routing_map.sum(dim=0)
            # Apply l1 regularization
            probs = self.apply_l1_reg(probs, num_local_tokens_per_expert, activation=probs)
            # Record the sparsity of the ReLU output
            sparsity = 1 - routing_map.sum().float() / routing_map.numel()
            self.config.moe_relu_sparsity += sparsity
        return probs, routing_map

    def apply_l1_reg(self, probs: torch.Tensor, num_local_tokens_per_expert: torch.Tensor, activation: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            probs (torch.Tensor): The probs output by the router for each token.
                [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
                [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        l1_reg_coeff = self.config.moe_relu_l1_reg_coeff.item()

        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            l1_reg_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        # L1 regularization with load balancing shares the same formula with switch load balancing loss:
        # l1_reg = sum((probs_per_expert/num_tokens) *
        # (tokens_per_expert/(num_tokens*topk))) * num_experts * l1_reg_coeff.
        l1_reg = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            l1_reg_coeff,
            sequence_partition_group=sequence_partition_group,
        )

        save_to_aux_losses_tracker(
            "l1_reg_loss",
            l1_reg / l1_reg_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, l1_reg)
        return activation

    def routing(self, logits: torch.Tensor):
        """ReLU routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        scores, routing_map = self.l1_reg_load_balancing(logits)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


class SampleTopKRouter(TopKRouter):
    """Sample Top-k router."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

    def aux_loss_load_balancing(self, logits: torch.Tensor,B=None,S=None):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            indices (torch.Tensor): The mask of token to experts assignment.
        """
        current_topk = self.get_current_topk()
        probs, routing_map, tokens_per_expert = sp_topk_softmax_with_capacity(
            logits,
            current_topk,
            batch_size = B,
            Seq_length = S,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,

        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Sample Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        S,B,E = logits.shape
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits, B, S,)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                deterministic_mode=self.config.deterministic_mode,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        # logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map
