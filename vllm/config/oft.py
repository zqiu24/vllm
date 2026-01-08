# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

OFTDType = Literal["auto", "float16", "bfloat16"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OFTConfig:
    """Configuration for OFT."""

    max_oft_block_size: int = 16
    """Max OFT block size."""
    max_ofts: int = 1
    """Max number of OFTs in a single batch."""
    fully_sharded_ofts: bool = False
    """By default, only half of the OFT computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max block size or tensor parallel size, this is likely faster.
    """
    max_cpu_ofts: Optional[int] = None
    """Maximum number of OFTs to store in CPU memory. Must be >= than
    `max_ofts`."""
    oft_dtype: Union[torch.dtype, OFTDType] = "auto"
    """Data type for OFT. If auto, will default to base model dtype."""
    oft_extra_vocab_size: int = 256
    """(Deprecated) Maximum size of extra vocabulary that can be present in a 
    OFT adapter. Will be removed in v0.12.0."""
    oft_vocab_padding_size: ClassVar[int] = current_platform\
        .get_oft_vocab_padding_size()
    default_mm_ofts: Optional[dict[str, str]] = None
    """Dictionary mapping specific modalities to OFT model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a OFT to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own OFT, we do NOT apply
    default_mm_ofts because we currently only support one oft adapter
    per prompt. When run in offline mode, the oft IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""
    bias_enabled: bool = False
    """[DEPRECATED] Enable bias for OFT adapters. This option will be
    removed in v0.12.0."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_oft_block_size)
        factors.append(self.max_ofts)
        factors.append(self.fully_sharded_ofts)
        factors.append(self.oft_dtype)
        factors.append(self.oft_extra_vocab_size)
        factors.append(self.oft_vocab_padding_size)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # Deprecation warning for oft_extra_vocab_size
        logger.warning(
            "`oft_extra_vocab_size` is deprecated and will be removed "
            "in v0.12.0. Additional vocabulary support for "
            "OFT adapters is being phased out.")

        # Deprecation warning for enable_oft_bias
        if self.bias_enabled:
            logger.warning("`enable_oft_bias` is deprecated "
                           "and will be removed in v0.12.0.")

        # Setting the maximum block size to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_block_sizes = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_oft_extra_vocab_size = (256, 512)
        if self.max_oft_block_size not in possible_max_block_sizes:
            raise ValueError(
                f"max_oft_block_size ({self.max_oft_block_size}) must be one of "
                f"{possible_max_block_sizes}.")
        if self.oft_extra_vocab_size not in possible_oft_extra_vocab_size:
            raise ValueError(
                f"oft_extra_vocab_size ({self.oft_extra_vocab_size}) "
                f"must be one of {possible_oft_extra_vocab_size}.")
        if self.max_ofts < 1:
            raise ValueError(f"max_ofts ({self.max_ofts}) must be >= 1.")
        if self.max_cpu_ofts is None:
            self.max_cpu_ofts = self.max_ofts
        elif self.max_cpu_ofts < self.max_ofts:
            raise ValueError(
                f"max_cpu_ofts ({self.max_cpu_ofts}) must be >= "
                f"max_ofts ({self.max_ofts})")

    def verify_with_cache_config(self, cache_config: CacheConfig):
        if cache_config.cpu_offload_gb > 0 and not envs.VLLM_USE_V1:
            raise ValueError(
                "V0 OFT does not support CPU offload, please use V1.")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.oft_dtype in (None, "auto"):
            self.oft_dtype = model_config.dtype
        elif isinstance(self.oft_dtype, str):
            self.oft_dtype = getattr(torch, self.oft_dtype)
