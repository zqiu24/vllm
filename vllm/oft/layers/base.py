# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig

if TYPE_CHECKING:
    from vllm.oft.punica_wrapper import PunicaWrapperBase


class BaseLayerWithOFT(nn.Module):

    def slice_oft_r(
        self, oft_r: Union[torch.Tensor, list[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, list[Union[torch.Tensor, None]]]:
        """Slice oft_r if splitting for tensor parallelism."""
        ...

    def create_oft_weights(
        self,
        max_ofts: int,
        oft_config: OFTConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """Initializes oft matrices."""
        ...

    def reset_oft(self, index: int):
        """Resets the oft weights at index back to 0."""
        ...

    def set_oft(
        self,
        index: int,
        oft_r: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
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
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this OFT layer."""
        raise NotImplementedError
