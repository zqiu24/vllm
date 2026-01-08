# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any, Literal, Optional, Union

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.oft.models import (OFTModel, OFTModelManager,
                              LRUCacheOFTModelManager, create_oft_manager)
from vllm.oft.peft_helper import PEFTHelper
from vllm.oft.request import OFTRequest
from vllm.oft.utils import get_adapter_absolute_path

logger = init_logger(__name__)


class WorkerOFTManager:
    """WorkerOFTManager that manages OFT models on the worker side.

    Every request, the requested OFTs will be loaded (unless they are already
    loaded), and every other OFT will be unloaded."""

    _manager_cls: type[OFTModelManager] = OFTModelManager

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        embedding_modules: dict[str, str],
        embedding_padding_modules: list[str],
        oft_model_cls: type[OFTModel] = OFTModel,
    ):
        self._oft_model_cls = oft_model_cls
        self.embedding_modules = embedding_modules
        self.embedding_padding_modules = embedding_padding_modules
        self._cached_dummy_oft: Union[None, Literal[False], OFTModel] = False
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.oft_config = vllm_config.oft_config

        # Use get_text_config() in case of multimodal models
        text_config = vllm_config.model_config.hf_config.get_text_config()

        self.max_position_embeddings = text_config.max_position_embeddings
        self.device = device
        # Lazily initialized by create_oft_manager.
        self._adapter_manager: OFTModelManager

    @contextmanager
    def dummy_oft_cache(self):
        """Use this context manager to reuse the dummy oft model
        to avoid creating it repeatedly."""
        self._cached_dummy_oft = None
        yield
        self._cached_dummy_oft = False

    @property
    def is_enabled(self) -> bool:
        return True

    def create_oft_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        oft_manager = create_oft_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            oft_config=self.oft_config,
            device=self.device,
            oft_manager_cls=self._manager_cls,
        )
        self._adapter_manager = oft_manager
        return oft_manager.model

    def _load_adapter(self, oft_request: OFTRequest) -> OFTModel:
        try:
            supported_oft_modules = (
                self._adapter_manager.supported_oft_modules)
            packed_modules_mapping = (
                self._adapter_manager.packed_modules_mapping)
            expected_oft_modules: list[str] = []
            for module in supported_oft_modules:
                if module in packed_modules_mapping:
                    expected_oft_modules.extend(
                        packed_modules_mapping[module])
                else:
                    expected_oft_modules.append(module)

            expected_oft_modules = list(set(expected_oft_modules))
            oft_path = get_adapter_absolute_path(oft_request.oft_path)

            peft_helper = PEFTHelper.from_local_dir(
                oft_path, self.max_position_embeddings,
                oft_request.tensorizer_config_dict)

            # Validates the OFT configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.oft_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of oft weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)

            oft = self._oft_model_cls.from_local_checkpoint(
                oft_path,
                expected_oft_modules,
                peft_helper=peft_helper,
                oft_model_id=oft_request.oft_int_id,
                device="cpu",
                dtype=self.oft_config.oft_dtype,
                target_embedding_padding=self.vocab_size +
                self.oft_config.oft_extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
                tensorizer_config_dict=oft_request.tensorizer_config_dict,
                weights_mapper=hf_to_vllm_mapper)

        except FileNotFoundError as e:
            # FileNotFoundError should be raised if both
            # - No adapter found to download from huggingface (or in
            #       offline mode)
            # - No local adapter files found at `oft_request.oft_path`
            # For NotFoundError
            raise ValueError(
                f"Loading oft {oft_request.oft_name} failed: No adapter "
                f"found for {oft_request.oft_path}") from e
        except Exception as e:
            # For BadRequestError
            raise e

        if oft.extra_vocab_size > self.oft_config.oft_extra_vocab_size:
            raise ValueError(f"OFT added vocab size {oft.extra_vocab_size} "
                             f"is greater than oft_extra_vocab_size "
                             f"{self.oft_config.oft_extra_vocab_size}.")
        return oft

    def add_dummy_oft(self, oft_request: OFTRequest, rank: int) -> bool:
        if oft_request.oft_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_oft, OFTModel):
            dummy_oft = self._cached_dummy_oft.clone(
                oft_request.oft_int_id)
        else:
            dummy_oft = self._adapter_manager.create_dummy_oft(
                oft_request.oft_int_id, rank, self.embedding_modules)
            if self._cached_dummy_oft is None:
                self._cached_dummy_oft = dummy_oft
        return self._adapter_manager.add_adapter(dummy_oft)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any],
                            mapping: Optional[Any]) -> None:
        self._apply_adapters(requests)
        if mapping is not None:
            self._adapter_manager.set_adapter_mapping(mapping)

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        existing_adapters = self.list_adapters()
        models_map = {
            adapter_request.adapter_id: adapter_request
            for adapter_request in adapter_requests if adapter_request
        }
        if len(models_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested models ({len(models_map)}) is greater "
                "than the number of GPU model slots "
                f"({self._adapter_manager.adapter_slots}).")
        requested_ids = set(models_map)
        for adapter_id in existing_adapters - requested_ids:
            self.remove_adapter(adapter_id)
        for adapter_id in requested_ids - existing_adapters:
            self.add_adapter(models_map[adapter_id])

    def add_adapter(self, adapter_request: Any) -> bool:
        if adapter_request.adapter_id in self.list_adapters():
            return False
        loaded_adapter = self._load_adapter(adapter_request)
        loaded = self._adapter_manager.add_adapter(loaded_adapter)
        self._adapter_manager.activate_adapter(loaded_adapter.id)
        return loaded

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return set(self._adapter_manager.list_adapters())


class LRUCacheWorkerOFTManager(WorkerOFTManager):
    """WorkerOFTManager that manages OFT models on the worker side.

    Uses an LRU Cache. Every request, the requested OFTs will be loaded
    (unless they are already loaded) and least recently used OFTs will
    be unloaded if the cache is above capacity."""

    _manager_cls: type[LRUCacheOFTModelManager] = LRUCacheOFTModelManager

    def create_oft_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        oft_manager = create_oft_manager(
            model,
            oft_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            oft_config=self.oft_config,
            device=self.device,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._adapter_manager = oft_manager
        return oft_manager.model

    def _apply_adapters(self, oft_requests: set[OFTRequest]) -> None:
        ofts_map = {
            oft_request.oft_int_id: oft_request
            for oft_request in oft_requests if oft_request
        }
        if len(ofts_map) > self._adapter_manager.oft_slots:
            raise RuntimeError(
                f"Number of requested OFTs ({len(ofts_map)}) is greater "
                "than the number of GPU OFT slots "
                f"({self._adapter_manager.oft_slots}).")
        for oft in ofts_map.values():
            self.add_adapter(oft)

    def add_adapter(self, oft_request: OFTRequest) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        if oft_request.oft_int_id not in self.list_adapters():
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded oft adapters to very temporarily
            # exceed `--max-cpu-ofts`.
            oft = self._load_adapter(oft_request)

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager,
                                  LRUCacheOFTModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(oft)
        else:
            # If the oft is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(
                oft_request.oft_int_id) is not None
        self._adapter_manager.activate_adapter(oft_request.oft_int_id)
        return loaded
