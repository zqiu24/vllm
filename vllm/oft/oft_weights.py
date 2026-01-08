# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence as GenericSequence
from typing import Optional

import torch
import torch.types

from vllm.oft.peft_helper import PEFTHelper
from vllm.utils import is_pin_memory_available


class OFTLayerWeights:
    """OFT weights for a layer composed of one orthogonal rotation matrix."""

    def __init__(
        self,
        module_name: str,
        input_dim: int,
        output_dim: int,
        block_size: int,
        oft_r: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        embeddings_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        self.module_name = module_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_size = block_size
        self.oft_r = oft_r
        self.bias = bias
        self.embeddings_tensor = embeddings_tensor

    def optimize(self) -> "OFTLayerWeights":
        """Does not apply for OFT."""
        return self

    @property
    def input_dim(self) -> int:
        return self.input_dim

    @property
    def output_dim(self) -> int:
        return self.output_dim

    @property
    def is_packed(self) -> bool:
        return False

    @property
    def extra_vocab_size(self) -> int:
        return self.embeddings_tensor.shape[
            0] if self.embeddings_tensor is not None else 0

    @classmethod
    def from_config(
        cls,
        module_name: str,
        input_dim: int,
        output_dim: int,
        peft_helper: PEFTHelper,
        embeddings_tensor: Optional[torch.Tensor] = None,
    ) -> "OFTLayerWeights":
        return cls(module_name, input_dim, output_dim, peft_helper.block_size, None, 
                   None, embeddings_tensor)

    @classmethod
    def create_dummy_oft_weights(
            cls,
            module_name: str,
            input_dim: int,
            output_dim: int,
            block_size: int,
            dtype: torch.dtype,
            device: torch.types.Device,
            embeddings_tensor_dim: Optional[int] = None,
            bias_enabled: Optional[bool] = False) -> "OFTLayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        oft_r_dim = block_size * (block_size - 1) // 2
        oft_r_num = input_dim // oft_r_dim
        oft_r = torch.zeros([oft_r_num, oft_r_dim],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        if bias_enabled:
            bias = torch.zeros([output_dim],
                               dtype=dtype,
                               device=device,
                               pin_memory=pin_memory)
        else:
            bias = None

        embeddings_tensor = torch.rand(
            10,
            embeddings_tensor_dim,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory) if embeddings_tensor_dim else None
        return cls(
            module_name,
            block_size=block_size,
            oft_r=oft_r,
            bias=bias,
            embeddings_tensor=embeddings_tensor,
        )


class PackedOFTLayerWeights(OFTLayerWeights):
    """OFT used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        input_dim: int,
        output_dim: int,
        block_size: int,
        oft_r: list[Optional[torch.Tensor]],
        bias: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> None:
        super().__init__(
            module_name=module_name,
            input_dim=input_dim,
            output_dim=output_dim,
            block_size=block_size,
            oft_r=oft_r,
            bias=bias,
            embeddings_tensor=None,
        )

    @classmethod
    def pack(
        cls, ofts: GenericSequence[Optional["OFTLayerWeights"]]
    ) -> "PackedOFTLayerWeights":
        """Pack a list of OFTs into a single OFT.

        If OFT is None, it signifies that the submodule does not have a OFT.
        """
        first_oft = next(oft for oft in ofts if oft is not None)
        for oft in ofts:
            if oft is None:
                continue
            oft.optimize()
        block_size = first_oft.block_size
        module_name = first_oft.module_name
        input_dim = first_oft.input_dim
        output_dim = first_oft.output_dim
        obj = cls(
            module_name,
            input_dim,
            output_dim,
            block_size,
            [oft.oft_r if oft is not None else None for oft in ofts],
            [oft.bias if oft is not None else None for oft in ofts]
            )
        return obj

    def optimize(self) -> "PackedOFTLayerWeights":
        """Does not apply for OFT."""
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True
