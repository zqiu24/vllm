# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define OFT functionality mixin for model runners.
"""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.oft import OFTConfig
from vllm.logger import init_logger
from vllm.oft.layers import OFTMapping
from vllm.oft.request import OFTRequest
from vllm.oft.worker_manager import LRUCacheWorkerOFTManager
from vllm.model_executor.models import supports_oft, supports_multimodal
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch = TPUInputBatch | GPUInputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class OFTModelRunnerMixin:
    def load_oft_model(
        self, model: nn.Module, vllm_config: VllmConfig, device: torch.device
    ) -> nn.Module:
        if not supports_oft(model):
            raise ValueError(f"{model.__class__.__name__} does not support OFT yet.")

        if supports_multimodal(model):
            logger.warning(
                "Regarding multimodal models, vLLM currently "
                "only supports adding OFT to language model."
            )
        # Add OFT Manager to the Model Runner
        self.oft_manager = LRUCacheWorkerOFTManager(
            vllm_config,
            device,
            model.embedding_modules,
            model.embedding_padding_modules,
        )
        return self.oft_manager.create_oft_manager(model)

    def _set_active_ofts(
        self,
        prompt_oft_mapping: tuple[int, ...],
        token_oft_mapping: tuple[int, ...],
        oft_requests: set[OFTRequest],
    ) -> None:
        self._ensure_oft_enabled()

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        oft_mapping = OFTMapping(
            token_oft_mapping, prompt_oft_mapping, is_prefill=True
        )
        self.oft_manager.set_active_adapters(oft_requests, oft_mapping)

    def _ensure_oft_enabled(self) -> None:
        if not hasattr(self, "oft_manager"):
            raise RuntimeError("OFT is not enabled. Use --enable-oft to enable OFT.")

    def set_active_ofts(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
    ) -> None:
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        prompt_oft_mapping: tuple[int, ...]  # of size np.sum(num_sampled_tokens)
        token_oft_mapping: tuple[int, ...]  # of size np.sum(num_scheduled_tokens)
        oft_requests: set[OFTRequest]
        prompt_oft_mapping, token_oft_mapping, oft_requests = (
            input_batch.make_oft_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        return self._set_active_ofts(
            prompt_oft_mapping, token_oft_mapping, oft_requests
        )

    @contextmanager
    def maybe_setup_dummy_ofts(
        self, oft_config: OFTConfig | None, remove_oft: bool = True
    ):
        if oft_config is None:
            yield
        else:
            # __enter__ code
            assert self.oft_manager is not None, "OFT is not enabled"

            num_ofts = oft_config.max_ofts
            oft_warmup_block_size = (
                oft_config.max_oft_block_size if oft_config.max_oft_block_size < 8 else 8
            )
            # Make dummy oft requests
            oft_requests: set[OFTRequest] = {
                OFTRequest(
                    oft_name=f"warmup_{oft_id}",
                    oft_int_id=oft_id,
                    oft_path="/not/a/real/path",
                )
                for oft_id in range(1, num_ofts + 1)
            }

            with self.oft_manager.dummy_oft_cache():
                # Add the dummy OFTs here so _set_active_ofts doesn't try to
                # load from disk.
                for lr in oft_requests:
                    self.oft_manager.add_dummy_oft(lr, block_size=oft_warmup_block_size)

                yield

            # __exit__ code
            if remove_oft:
                self.oft_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_ofts(
        self,
        oft_config: OFTConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        activate_oft: bool = True,
    ):
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        if oft_config is None:
            yield
        else:
            # __enter__ code
            assert self.oft_manager is not None, "OFT is not enabled"

            num_reqs = len(num_scheduled_tokens)
            num_ofts = oft_config.max_ofts

            # Make prompt oft mapping
            # Assign OFT IDs cyclically to simulate a worst-case scenario.
            if activate_oft:
                prompt_oft_mapping = (
                    np.arange(num_reqs, dtype=np.int32) % num_ofts
                ) + 1
            else:
                prompt_oft_mapping = np.zeros(num_reqs, dtype=np.int32)

            # Make sample oft mapping
            sample_oft_mapping = np.repeat(prompt_oft_mapping, num_sampled_tokens)

            # Make token oft mapping
            token_oft_mapping = np.repeat(prompt_oft_mapping, num_scheduled_tokens)

            # Make dummy oft requests
            oft_requests: set[OFTRequest] = {
                OFTRequest(
                    oft_name=f"warmup_{oft_id}",
                    oft_int_id=oft_id,
                    oft_path="/not/a/real/path",
                )
                for oft_id in range(1, num_ofts + 1)
            }

            self._set_active_ofts(
                tuple(sample_oft_mapping), tuple(token_oft_mapping), oft_requests
            )

            yield

    @contextmanager
    def maybe_dummy_run_with_oft(
        self,
        oft_config: OFTConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        activate_oft: bool = True,
        remove_oft: bool = True,
    ):
        with (
            self.maybe_setup_dummy_ofts(oft_config, remove_oft),
            self.maybe_select_dummy_ofts(
                oft_config, num_scheduled_tokens, num_sampled_tokens, activate_oft
            ),
        ):
            yield

    def maybe_remove_all_ofts(self, oft_config: OFTConfig | None):
        if oft_config is None:
            return
        self.oft_manager.remove_all_adapters()

    def add_oft(self, oft_request: OFTRequest) -> bool:
        self._ensure_oft_enabled()
        return self.oft_manager.add_adapter(oft_request)

    def remove_oft(self, oft_id: int) -> bool:
        self._ensure_oft_enabled()
        return self.oft_manager.remove_adapter(oft_id)

    def pin_oft(self, oft_id: int) -> bool:
        self._ensure_oft_enabled()
        return self.oft_manager.pin_adapter(oft_id)

    def list_ofts(self) -> set[int]:
        self._ensure_oft_enabled()
        return self.oft_manager.list_adapters()
