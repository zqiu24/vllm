# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

OFTDType = Literal["auto", "float16", "bfloat16"]
MaxOFTBlockSize = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
OFTExtraVocabSize = Literal[256, 512]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OFTConfig:
    """Configuration for OFT."""

    max_oft_block_size: MaxOFTBlockSize = 16
    """Max OFT block size."""
    max_ofts: int = Field(default=1, ge=1)
    """Max number of OFTs in a single batch."""
    fully_sharded_ofts: bool = False
    """By default, only half of the OFT computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_ofts: int | None = None
    """Maximum number of OFTs to store in CPU memory. Must be >= than
    `max_ofts`."""
    oft_dtype: torch.dtype | OFTDType = "auto"
    """Data type for OFT. If auto, will default to base model dtype."""
    default_mm_ofts: dict[str, str] | None = None
    """Dictionary mapping specific modalities to OFT model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a OFT to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own OFT, we do NOT apply
    default_mm_ofts because we currently only support one oft adapter
    per prompt. When run in offline mode, the oft IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""

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

        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_oft_config(self) -> Self:
        if self.max_cpu_ofts is None:
            self.max_cpu_ofts = self.max_ofts
        elif self.max_cpu_ofts < self.max_ofts:
            raise ValueError(
                f"max_cpu_ofts ({self.max_cpu_ofts}) must be >= "
                f"max_ofts ({self.max_ofts})"
            )

        return self

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.oft_dtype in (None, "auto"):
            self.oft_dtype = model_config.dtype
        elif isinstance(self.oft_dtype, str):
            self.oft_dtype = getattr(torch, self.oft_dtype)
