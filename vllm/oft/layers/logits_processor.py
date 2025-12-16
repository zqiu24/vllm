# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

from .base import BaseLayerWithOFT


class LogitsProcessorWithOFT(BaseLayerWithOFT):
    """
    OFT wrapper for LogitsProcessor, with extra logic to handle the
    application of the OFT adapter and added OFT vocabulary.

    Args:
        base_layer: LogitsProcessor layer
        hidden_size: hidden size of the model
        dtype: data type of the model
        device: device of the model
        sharded_to_full_mapping: index mapping from sharded vocab to full vocab
            received from base_layer.get_sharded_to_full_mapping(). If None,
            no reindexing will be done.
    """

    def __init__(
        self,
        base_layer: LogitsProcessor,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        sharded_to_full_mapping: list[int] | None,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sharded_to_full_mapping = sharded_to_full_mapping

    @property
    def logits_as_input(self):
        return self.base_layer.logits_as_input

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def scale(self):
        return self.base_layer.scale

    @property
    def soft_cap(self):
        return self.base_layer.soft_cap

    @property
    def use_all_gather(self):
        return self.base_layer.use_all_gather

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    @property
    def should_modify_greedy_probs_inplace(self):
        return self.base_layer.should_modify_greedy_probs_inplace

    def create_oft_weights(
        self,
        max_ofts: int,
        oft_config: OFTConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        # TODO: Verify if this condition can be further relaxed
        if 32000 < self.base_layer.vocab_size > 257024:
            raise ValueError(
                "When using OFT, vocab size must be 32000 >= vocab_size <= 257024"
            )
        self.oft_R_stacked = torch.zeros(
            (
                max_ofts,
                1,
                oft_config.max_oft_block_size,
                self.hidden_size,
            ),
            dtype=oft_config.oft_dtype,
            device=self.device,
        )

        if self.sharded_to_full_mapping is not None:
            self.sharded_to_full_mapping_gpu = torch.tensor(
                self.sharded_to_full_mapping, device=self.device, dtype=torch.long
            )
        else:
            self.sharded_to_full_mapping_gpu = None

    def reset_oft(self, index: int):
        self.oft_R_stacked[index] = 0

    def set_oft(
        self,
        index: int,
        oft_R: torch.Tensor,
    ):
        self.reset_oft(index)
        self.oft_R_stacked[index, 0, : oft_R.shape[0], : oft_R.shape[1]].copy_(
            oft_R, non_blocking=True
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states)
        if embedding_bias is not None:
            logits += embedding_bias

        # Gather logits for TP
        logits = self.base_layer._gather_logits(logits)

        if logits is None:
            return None

        if self.sharded_to_full_mapping_gpu is not None:
            # Reindex full logits tensor to ensure 1:1 mapping between
            # index and token_id
            # Example for:
            #   org_vocab_size = 4
            #   added_vocab_size = 2
            #   pad_to_size = 8
            #   tp_size = 2

            # indices:  [0, 1, 2,  3, 4, 5, 6,  7]
            # token_id: [0, 1, 4, -1, 2, 3, 5, -1]

            # Therefore, the mapping is expected to be:
            # [0, 1, 4, 6, 2, 3, 5, 7] so that when we reindex,
            # we get:
            # indices:  [0, 1, 2, 3, 4, 5,  6,  7]
            # token_id: [0, 1, 2, 3, 4, 5, -1, -1]
            logits = logits[:, self.sharded_to_full_mapping_gpu]

        oft_output: torch.Tensor | None = self.punica_wrapper.add_oft_logits(
            logits, hidden_states, self.oft_R_stacked, 1.0
        )

        if not current_platform.can_update_inplace():
            logits = oft_output

        # Remove paddings in vocab (if any).
        logits = logits[:, : self.base_layer.vocab_size]
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        oft_config: OFTConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # Special handling for the LogitsProcessor.
        return False
