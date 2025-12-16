# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.utils import divide
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.platforms import current_platform

from .base_linear import BaseLinearLayerWithOFT
from .utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace


def _mcp_apply(x, bias, layer: "ColumnParallelLinearWithOFT"):
    """
    For `ColumnParallelLinearWithOFT` or classes that inherit from
    `ColumnParallelLinearWithOFT`, they share the same `apply` logic.
    """
    assert (
        layer.n_slices
        == len(layer.lora_a_stacked)
        == len(layer.lora_b_stacked)
        == len(layer.output_slices)
    )

    output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape

    # Since communication is needed, the buffer is directly initialized as a
    # tensor rather than a tuple of tensor.
    buffers = torch.zeros(
        (layer.n_slices, x.shape[0], layer.lora_a_stacked[0].shape[2]),
        dtype=torch.float32,
        device=x.device,
    )

    shrunk_buffers: torch.Tensor | None = layer.punica_wrapper.add_shrink(
        buffers, x, layer.lora_a_stacked, 1.0
    )

    if not current_platform.can_update_inplace():
        buffers = shrunk_buffers

    buffers = tensor_model_parallel_all_gather(buffers)

    lora_output: torch.Tensor | None = layer.punica_wrapper.add_expand(
        output,
        buffers,
        layer.lora_b_stacked,
        layer.output_slices,
        offset_start=0,
        add_input=True,
    )

    if not current_platform.can_update_inplace():
        output = lora_output

    output = output.view(*out_orig_shape)
    # now have column partitioned and packed output
    return output


class ColumnParallelLinearWithOFT(BaseLinearLayerWithOFT):
    """
    OFT on top of ColumnParallelLinear layer.
    OFT B is sliced for tensor parallelism.
    There are two types for the `base_layer`:
    1. ColumnParallelLinear, e.g.`dense_h_to_4h` in `FalconForCausalLM`.
    2. MergedColumnParallelLinear, e.g.`gate_up_proj` in `Phi3ForCausalLM`.
    """

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)
        # The base_layer type is ColumnParallelLinear or
        # MergedColumnParallelLinear, their weight sharding logic is
        # inconsistent when TP is greater than 1.
        self.is_merged_col_linear = type(base_layer) is MergedColumnParallelLinear
        self.output_size = self.base_layer.output_size_per_partition
        # There is only one OFT layer
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        # Applicable to cases where the base_layer is
        # MergedColumnParallelLinear.
        if self.is_merged_col_linear:
            shard_size = self.output_size // 2
            offset = lora_b.shape[0] // 2

            left_weight = lora_b[
                self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size, :
            ]
            right_weight = lora_b[
                offset + self.tp_rank * shard_size : offset
                + (self.tp_rank + 1) * shard_size,
                :,
            ]
            lora_b = torch.cat([left_weight, right_weight], dim=0)
        # Applicable to cases where the base_layer is
        # ColumnParallelLinear.
        else:
            shard_size = self.output_size
            start_idx = self.tp_rank * shard_size
            end_idx = (self.tp_rank + 1) * shard_size
            lora_b = lora_b[start_idx:end_idx, :]
        return lora_b

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.apply(input_, bias)
        if self.base_layer.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.base_layer.return_bias:
            return output

        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1
        )


class MergedColumnParallelLinearWithOFT(ColumnParallelLinearWithOFT):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (e.g. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 OFTs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(
        self, base_layer: MergedColumnParallelLinear | QKVParallelLinear
    ) -> None:
        super().__init__(base_layer)
        # There are two OFT layers
        # the output_sizes in MergedColumnParallelLinear is not sharded by tp
        # we need to divide it by the tp_size to get correct slices size
        output_sizes = self.base_layer.output_sizes
        self.output_slices = tuple(
            divide(output_size, self.tp_size) for output_size in output_sizes
        )
        self.n_slices = len(self.output_slices)
        self.output_ids = (self.tp_rank,) * self.n_slices

    def create_lora_weights(
        self,
        max_loras: int,
        oft_config: OFTConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """
        The main reason for overriding this function is to enhance  code
        maintainability.
        """
        self.oft_config = oft_config

        lora_a_output_size_per_partition = (
            oft_config.max_lora_rank
            if not oft_config.fully_sharded_loras
            else divide(oft_config.max_lora_rank, self.tp_size)
        )

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=oft_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                output_size,
                oft_config.max_lora_rank,
                dtype=oft_config.lora_dtype,
                device=self.device,
            )
            for output_size in self.output_slices
        )

    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        return lora_a

    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        sliced_lora_b = [None] * self.n_slices
        for i, (shard_id, shard_size) in enumerate(
            zip(self.output_ids, self.output_slices)
        ):
            if (lora_b_i := lora_b[i]) is not None:
                sliced_lora_b[i] = lora_b_i[
                    shard_size * shard_id : shard_size * (shard_id + 1), :
                ]
        return sliced_lora_b

    def set_oft(
        self,
        index: int,
        oft_R: torch.Tensor,
    ):
        self.reset_oft(index)

        if self.tp_size > 1:
            oft_R = self.slice_oft_R(oft_R)

        for i in range(self.n_slices):
            if (oft_R_i := oft_R[i]) is not None:
                self.oft_R_stacked[i][
                    index, 0, : oft_R_i.shape[0], : oft_R_i.shape[1]
                ].copy_(oft_R_i, non_blocking=True)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 2
        )


class QKVParallelLinearWithOFT(ColumnParallelLinearWithOFT):
    """
    ColumnParallelLinear layer that is specifically designed for
    qkv_proj. Certain models, such as chatglm3 and baichuan-7b,
    only contains a single OFT within their qkv_proj layer.

    During inference with Tensor Parallel, the weights of lora_b
    must be accurately partitioned according to the respective ranks.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.q_proj_total_size = (
            self.base_layer.total_num_heads * self.base_layer.head_size
        )
        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.kv_proj_total_size = (
            self.base_layer.total_num_kv_heads * self.base_layer.head_size
        )
        # There is only one OFT layer
        self.n_slices = 1

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas
        lora_b_q = lora_b[
            self.q_proj_shard_size * self.q_shard_id : self.q_proj_shard_size
            * (self.q_shard_id + 1),
            :,
        ]
        k_offset = self.q_proj_total_size
        lora_b_k = lora_b[
            k_offset + self.kv_proj_shard_size * self.kv_shard_id : k_offset
            + self.kv_proj_shard_size * (self.kv_shard_id + 1),
            :,
        ]
        v_offset = k_offset + self.kv_proj_total_size
        lora_b_v = lora_b[
            v_offset + self.kv_proj_shard_size * self.kv_shard_id : v_offset
            + self.kv_proj_shard_size * (self.kv_shard_id + 1),
            :,
        ]
        lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=0)
        return lora_b

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 1


class MergedQKVParallelLinearWithOFT(MergedColumnParallelLinearWithOFT):
    """MergedColumnParallelLinear layer that is composed of 3 sublayers (slices)
    packed together in qkv proj fashion
    (q_proj + k_proj + v_proj -> qkv_proj).

    This means we have 3 OFTs, each applied to one slice of the layer.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        # There are three OFT layer.
        self.n_slices = len(self.base_layer.output_sizes)

        self.q_proj_shard_size = self.base_layer.num_heads * self.base_layer.head_size
        self.kv_proj_shard_size = (
            self.base_layer.num_kv_heads * self.base_layer.head_size
        )
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.output_ids = (
            self.q_shard_id,
            self.kv_shard_id,
            self.kv_shard_id,
        )

    def create_lora_weights(
        self,
        max_loras: int,
        oft_config: OFTConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """
        The main reason for overloading this function is to handle inconsistent
        weight dimensions in qkv lora.
        """
        super().create_lora_weights(max_loras, oft_config, model_config)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is QKVParallelLinear and len(packed_modules_list) == 3


# These following layers are based on the tensor parallelism strategy given in
# Y. Sheng et al., S-OFT: Serving Thousands of Concurrent OFT Adapters. 2023,
# https://arxiv.org/abs/2311.03285.


class ColumnParallelLinearWithShardedOFT(ColumnParallelLinearWithOFT):
    """
    Differs from ColumnParallelLinearWithOFT by slicing OFT A also.

    Based on S-OFT, slicing happens along the rank dim.
    """

    # For all OFT layers where the `base_layer` is `ColumnParallelLinear`,
    # their `lora_a` and `lora_b` have different sharding patterns. After
    # completing the `lora_a` GEMM , a gather operation is performed.
    # Therefore, the sharding of `lora_a` only needs to correspond with the
    # gather operation.
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_a_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        lora_a = lora_a[start_idx : start_idx + shard_size, :]
        return lora_a

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            oft_config=oft_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class MergedColumnParallelLinearWithShardedOFT(MergedColumnParallelLinearWithOFT):
    """
    Differs from MergedColumnParallelLinearWithOFT by slicing the
    OFT A's also.

    Based on S-OFT, slicing happens along the rank dim.
    """

    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        # NOTE: lora_a contains 2 subloras, and each sublora could be None.
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        lora_a = [
            lora_a[0][output_start_idx : output_start_idx + output_shard_size, :]
            if lora_a[0] is not None
            else None,
            lora_a[1][output_start_idx : output_start_idx + output_shard_size, :]
            if lora_a[1] is not None
            else None,
        ]
        return lora_a

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            oft_config=oft_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class QKVParallelLinearWithShardedOFT(QKVParallelLinearWithOFT):
    """
    Differs from QKVParallelLinearWithOFT by slicing the
    OFT A's also.

    Based on S-OFT, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_a_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        lora_a = lora_a[start_idx : start_idx + shard_size, :]
        return lora_a

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            oft_config=oft_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class MergedQKVParallelLinearWithShardedOFT(MergedQKVParallelLinearWithOFT):
    """
    Differs from MergedQKVParallelLinearWithOFT by slicing the
    OFT A's also.

    Based on S-OFT, slicing happens along the rank dim.
    """

    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        # NOTE: lora_a contains 3 subloras, and each sublora could be None.
        shard_size = [self.lora_a_stacked[i].shape[2] for i in range(3)]
        start_idx = [self.tp_rank * shard_size[i] for i in range(3)]
        lora_a = [
            lora_a[0][start_idx[0] : start_idx[0] + shard_size[0], :]
            if lora_a[0] is not None
            else None,
            lora_a[1][start_idx[1] : start_idx[1] + shard_size[1], :]
            if lora_a[1] is not None
            else None,
            lora_a[2][start_idx[2] : start_idx[2] + shard_size[2], :]
            if lora_a[2] is not None
            else None,
        ]
        return lora_a

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            oft_config=oft_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )
