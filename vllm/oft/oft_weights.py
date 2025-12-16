# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence as GenericSequence
from typing import Optional

import torch
import torch.types

from vllm.oft.peft_helper import PEFTHelper
from vllm.utils.platform_utils import is_pin_memory_available


class OFTLayerWeights:
    """OFT weights for a layer composed of two block diagonal matrixes."""

    def __init__(
        self,
        module_name: str,
        block_size: int,
        oft_R: torch.Tensor,
    ) -> None:
        self.module_name = module_name
        self.block_size = block_size
        self.oft_R = oft_R

    def optimize(self) -> "OFTLayerWeights":
        """Not applicable for OFT."""
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

    @classmethod
    def from_config(
        cls,
        module_name: str,
        peft_helper: PEFTHelper,
    ) -> "OFTLayerWeights":
        # oft_R is set to None for config-based construction
        return cls(
            module_name,
            peft_helper.oft_block_size,
            None,
        )

    @classmethod
    def create_dummy_oft_weights(
        cls,
        module_name: str,
        oft_R_size: int,
        oft_R_block_num: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.types.Device,
    ) -> "OFTLayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        oft_R = torch.zeros(
            [oft_R_block_num, oft_R_size], dtype=dtype, device=device, pin_memory=pin_memory
        )

        return cls(
            module_name,
            block_size=block_size,
            oft_R=oft_R,
        )


class PackedOFTLayerWeights(OFTLayerWeights):
    """OFT used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        block_size: int,
        oft_R: list[torch.Tensor | None],
    ) -> None:
        super().__init__(
            module_name=module_name,
            block_size=block_size,
            oft_R=oft_R,
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
        obj = cls(
            module_name,
            block_size,
            [oft.oft_R if oft is not None else None for oft in ofts],
        )
        return obj

    def optimize(self) -> "PackedOFTLayerWeights":
        """Not applicable for OFT."""
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
