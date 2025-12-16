# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal

from vllm.config.oft import OFTConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

logger = init_logger(__name__)


@dataclass
class PEFTHelper:
    """
    A helper class for PEFT configurations, specifically designed for OFT.
    This class handles configuration validation, compatibility checks for
    various OFT implementations.
    """

    # Required fields
    oft_block_size: int
    target_modules: list[str] | str

    bias: Literal["none"] = field(default="none")
    modules_to_save: list[str] | None = field(default=None)
    # Extra vllm field, start with 'vllm_' to avoid conflict
    vllm_max_position_embeddings: int | None = field(default=False)

    def _validate_features(self) -> list[str]:
        """
        Check if there are any unsupported OFT features.
        """
        error_msg = []
        if self.modules_to_save:
            error_msg.append("vLLM only supports modules_to_save being None.")
        return error_msg

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PEFTHelper":
        # Get all field information from the class
        class_fields = {f.name: f for f in fields(cls)}
        # Check for required fields
        required_fields = {
            name
            for name, f in class_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

        # Identify any missing required fields
        missing_fields = required_fields - set(config_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")

        # Filter out fields that aren't defined in the class
        filtered_dict = {k: v for k, v in config_dict.items() if k in class_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_local_dir(
        cls,
        oft_path: str,
        max_position_embeddings: int | None,
        tensorizer_config_dict: dict | None = None,
    ) -> "PEFTHelper":
        oft_config_path = os.path.join(oft_path, "adapter_config.json")

        if tensorizer_config_dict:
            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            from tensorizer.stream_io import open_stream

            oft_config_path = os.path.join(
                tensorizer_config.tensorizer_dir, "adapter_config.json"
            )
            with open_stream(
                oft_config_path, mode="rb", **tensorizer_args.stream_kwargs
            ) as f:
                config = json.load(f)

            logger.info(
                "Successfully deserialized OFT config from %s",
                tensorizer_config.tensorizer_dir,
            )

        else:
            with open(oft_config_path) as f:
                config = json.load(f)

        config["vllm_max_position_embeddings"] = max_position_embeddings
        return cls.from_dict(config)

    def validate_legal(self, oft_config: OFTConfig) -> None:
        """
        Validates the OFT configuration settings against application
        constraints and requirements.
        """
        error_msg = self._validate_features()
        if self.oft_block_size > oft_config.max_oft_block_size:
            error_msg.append(
                f"OFT rank {self.r} is greater than max_oft_block_size"
                f" {oft_config.max_oft_block_size}."
            )
        if self.bias != "none":
            error_msg.append("Adapter bias is not supported.")
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
