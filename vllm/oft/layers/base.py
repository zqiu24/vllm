# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig

if TYPE_CHECKING:
    from vllm.oft.punica_wrapper import PunicaWrapperBase


class BaseLayerWithOFT(nn.Module):
    def slice_oft_R(
        self, oft_R: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice oft R if splitting for tensor parallelism."""
        ...

    def create_oft_weights(
        self,
        max_ofts: int,
        oft_config: OFTConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes oft matrices."""
        ...

    def reset_oft(self, index: int):
        """Resets the oft weights at index back to 0."""
        ...

    def set_oft(
        self,
        index: int,
        oft_R: torch.Tensor,
    ):
        """Overwrites oft tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        """Returns True if the layer can be replaced by this OFT layer."""
        raise NotImplementedError
