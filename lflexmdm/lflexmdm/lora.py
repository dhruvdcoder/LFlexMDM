"""
Lightweight LoRA (Low-Rank Adaptation) implementation for DDiT layers.

This module provides LoRA adapters that can be applied to the linear layers
in DDiTLayer without modifying the original layer implementation.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA adapter for a linear layer.

    Computes: output = (x @ lora_A @ lora_B) * scaling
    where lora_A has shape (in_features, rank) and lora_B has shape (rank, out_features).

    The output is meant to be ADDED to the original linear layer's output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input dimension of the linear layer being adapted.
            out_features: Output dimension of the linear layer being adapted.
            rank: Rank of the low-rank decomposition.
            alpha: Scaling factor. The final scaling is alpha / rank.
            dropout: Dropout rate applied to input before LoRA computation.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA weight matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA weights following the original paper."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros so that LoRA starts as identity
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA output to be added to the original linear output.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA output tensor of shape (..., out_features)
        """
        x = self.dropout(x)
        # (... , in_features) @ (in_features, rank) @ (rank, out_features)
        # = (..., out_features)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class DDiTLayerLoRA(nn.Module):
    """
    LoRA adapters for all linear layers in a DDiTLayer.

    This module contains LoRA adapters for:
    - attn_qkv: QKV projection (d_model -> 3*d_model)
    - o_proj: Output projection (d_model -> d_model)
    - mlp_fc1: First MLP layer (d_model -> dim_feedforward)
    - mlp_fc2: Second MLP layer (dim_feedforward -> d_model)
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension.
            dim_feedforward: Feedforward dimension (typically 4 * d_model).
            rank: LoRA rank for all adapters.
            alpha: LoRA alpha (scaling factor) for all adapters.
            dropout: Dropout rate for LoRA layers.
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        # Attention LoRA adapters
        self.attn_qkv_lora = LoRALinear(
            d_model, 3 * d_model, rank=rank, alpha=alpha, dropout=dropout
        )
        self.o_proj_lora = LoRALinear(
            d_model, d_model, rank=rank, alpha=alpha, dropout=dropout
        )

        # MLP LoRA adapters
        self.mlp_fc1_lora = LoRALinear(
            d_model, dim_feedforward, rank=rank, alpha=alpha, dropout=dropout
        )
        self.mlp_fc2_lora = LoRALinear(
            dim_feedforward, d_model, rank=rank, alpha=alpha, dropout=dropout
        )

    def forward_attn_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA addition for attn_qkv."""
        return self.attn_qkv_lora(x)

    def forward_o_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA addition for o_proj."""
        return self.o_proj_lora(x)

    def forward_mlp_fc1(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA addition for MLP first layer."""
        return self.mlp_fc1_lora(x)

    def forward_mlp_fc2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA addition for MLP second layer."""
        return self.mlp_fc2_lora(x)


class DDiTLoRAStack(nn.Module):
    """
    Stack of DDiTLayerLoRA modules, one for each layer in the backbone encoder.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        dim_feedforward: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_layers: Number of DDiT layers in the backbone.
            d_model: Model dimension.
            dim_feedforward: Feedforward dimension.
            rank: LoRA rank for all adapters.
            alpha: LoRA alpha for all adapters.
            dropout: Dropout rate for LoRA layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DDiTLayerLoRA(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

    def __getitem__(self, idx: int) -> DDiTLayerLoRA:
        return self.layers[idx]

    def __len__(self) -> int:
        return self.num_layers


