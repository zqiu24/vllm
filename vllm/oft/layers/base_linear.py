# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import torch
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig
from vllm.distributed.utils import divide
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform

from .base import BaseLayerWithOFT
from .utils import _get_oft_device


class BaseLinearLayerWithOFT(BaseLayerWithOFT):

    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_oft_device(self.base_layer)
        self.oft_bias_stacked: Optional[tuple[torch.Tensor, ...]] = None
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

    def create_oft_weights(
        self,
        max_ofts: int,
        oft_config: OFTConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.oft_config = oft_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            oft_r_dim = oft_config.max_oft_block_size * (oft_config.max_oft_block_size - 1) // 2
            oft_r_num = self.input_size // oft_r_dim

        # shard the output dimension
        elif isinstance(self.base_layer, ColumnParallelLinear):
            oft_r_dim = oft_config.max_oft_block_size * (oft_config.max_oft_block_size - 1) // 2
            oft_r_num = self.input_size // oft_r_dim

        # shard the input dimension
        elif isinstance(self.base_layer, RowParallelLinear):
            oft_r_dim = oft_config.max_oft_block_size * (oft_config.max_oft_block_size - 1) // 2
            tmp_num = self.input_size // oft_r_dim
            oft_r_num = tmp_num if not oft_config.fully_sharded_ofts else tmp_num // self.tp_size

        else:
            raise NotImplementedError

        self.oft_r_stacked = tuple(
            torch.zeros(
                max_ofts,
                1,
                oft_r_num,
                oft_r_dim,
                dtype=oft_config.oft_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        if oft_config.bias_enabled:
            oft_bias_out_size = self.output_size
            self.oft_bias_stacked = tuple(
                torch.zeros(
                    max_ofts,
                    1,
                    oft_bias_out_size,
                    dtype=oft_config.oft_dtype,
                    device=self.device,
                ) for _ in range(self.n_slices))
        # self.output_slices = (self.lora_b_stacked[0].shape[2], )

    def reset_oft(self, index: int):
        for s_index in range(self.n_slices):
            self.oft_r_stacked[s_index][index] = 0
            if self.oft_config.bias_enabled:
                # Make mypy happy
                self.oft_bias_stacked = cast(tuple[torch.Tensor, ...],
                                              self.oft_bias_stacked)
                self.oft_bias_stacked[s_index][index] = 0

    def set_oft(
        self,
        index: int,
        oft_r: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        oft_bias: Optional[torch.Tensor] = None,
    ):
        # Except for QKVParallelLinearWithOFT and
        # MergedColumnParallelLinearWithOFT, all other linear OFT layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert len(self.oft_r_stacked) == self.n_slices == 1

        self.reset_oft(index)
        if self.tp_size > 1:
            oft_r = self.slice_oft_r(oft_r)
            if oft_bias is not None:
                oft_bias = self.slice_bias(oft_bias)

        self.oft_r_stacked[0][index,
                               0, :oft_r.shape[0], :oft_r.shape[1]].copy_(
                                   oft_r, non_blocking=True)

        if oft_bias is not None:

            self.oft_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.oft_bias_stacked)
            assert len(self.oft_bias_stacked)
            self.oft_bias_stacked[0][index, 0, :oft_bias.shape[0]].copy_(
                oft_bias, non_blocking=True)

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Assuming single active adapter at index 0 for now (vLLM usually manages mapping via metadata, 
        # but for this manual implementation we default to 0).  

        # 1. Apply OFT rotation (Input Side)
        # If merged QKV layer, this returns a stacked tensor: shape (3 * batch, input_dim)
        #   [Batch 0: rotated by R_q]
        #   [Batch 1: rotated by R_k]
        #   [Batch 2: rotated by R_v]
        x = apply_oft_linear(x, self.oft_R_stacked, adapter_idx=0)

        # 2. Base Layer Forward Pass (Black Box)
        # It processes the huge batch. 
        # Output shape: (3 * batch, output_dim_total) where output_dim_total = Q_dim + K_dim + V_dim
        # Output shape: (2 * batch, output_dim_total) for gate-up projection layer
        output_all = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # 3. Selection & Stitching (The "Deactivation" Step)
        # If we expanded the batch (merged layer), we must recover the correct outputs.
        if isinstance(self.oft_R_stacked, (tuple, list)) and len(self.oft_R_stacked) > 1:
            num_slices = len(self.oft_R_stacked)
            
            # A. Split the batch back into 3 separate chunks
            # Each chunk has shape (batch, output_dim_total)
            chunks = torch.chunk(output_all, num_slices, dim=0)
            
            final_parts = []
            start_col = 0
            
            # self.output_slices stores [Q_dim, K_dim, V_dim]
            for i, chunk in enumerate(chunks):
                width = self.output_slices[i]
                
                # B. Select the valid columns for this slice
                # Chunk 0 (Q-input) -> Keep only Q columns [0 : Q_dim]
                # Chunk 1 (K-input) -> Keep only K columns [Q_dim : Q_dim+K_dim]
                # ...
                valid_part = chunk[..., start_col : start_col + width]
                final_parts.append(valid_part)
                
                # Advance the column pointer
                start_col += width
            
            # C. Stitch them together to form the final result
            # Shape: (batch, Q_dim + K_dim + V_dim)
            output = torch.cat(final_parts, dim=-1)
        else:
            # Standard single layer case
            output = output_all

        return output

    @property
    def weight(self) -> torch.Tensor:

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
