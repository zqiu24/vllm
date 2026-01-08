# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from collections.abc import Sequence
from typing import Callable, Optional, TypeVar, Union

import regex as re
import safetensors.torch
import torch
from torch import nn

from vllm.config.oft import OFTConfig
from vllm.logger import init_logger
from vllm.oft.layers import BaseLayerWithOFT, OFTMapping
from vllm.oft.oft_weights import OFTLayerWeights, PackedOFTLayerWeights
from vllm.oft.peft_helper import PEFTHelper
from vllm.oft.punica_wrapper import get_punica_wrapper
from vllm.oft.utils import (from_layer, from_layer_logits_processor,
                             get_supported_oft_modules,
                             is_regex_target_modules,
                             parse_fine_tuned_oft_name, replace_submodule)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import SupportsOFT, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping
from vllm.utils import LRUCache, is_pin_memory_available

logger = init_logger(__name__)

T = TypeVar("T")


class AdapterLRUCache(LRUCache[int, T]):

    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: Optional[T]):
        logger.debug("Removing adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


_GLOBAL_LORA_ID = 0


def get_oft_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


def is_moe_model(model: nn.Module) -> bool:
    """Checks if the model contains FusedMoE layers and warns the user."""
    if any(isinstance(module, FusedMoE) for module in model.modules()):
        logger.warning_once(
            "For MoE models, vLLM currently does not support fused MoE OFT "
            "inference. Please ensure that the loaded OFT model does not "
            "contain expert weights.")
        return True
    return False


class OFTModel:
    """A OFT fine-tuned model."""

    def __init__(
        self,
        oft_model_id: int,
        block_size: int,
        ofts: dict[str, OFTLayerWeights],
    ) -> None:
        """
        Args:
            oft_model_id: The integer id for the oft model.
            block_size: oft block size.
            ofts: module name -> weights for oft-replaced layers.

        """
        self.id = oft_model_id

        assert (
            oft_model_id
            > 0), f"a valid oft id should be greater than 0, got {self.id}"
        self.block_size = block_size
        self.ofts: dict[str, OFTLayerWeights] = ofts

    def clone(self, oft_model_id: int) -> "OFTModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            oft_model_id,
            block_size=self.block_size,
            ofts=self.ofts.copy(),
        )

    @property
    def extra_vocab_size(self) -> int:
        return max(oft.extra_vocab_size
                   for oft in self.ofts.values()) if self.ofts else 0

    def get_oft(self, module_name: str) -> Optional[OFTLayerWeights]:
        """Get OFT for a given module by name"""
        return self.ofts.get(module_name, None)

    def check_oft_name(self, oft_name: str) -> bool:
        return oft_name in self.ofts

    # (yard1): TODO see if we can derive target_embedding_padding automatically
    @classmethod
    def from_oft_tensors(
        cls,
        oft_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        embeddings: Optional[dict[str, torch.Tensor]] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[dict[str, str]] = None,
        embedding_padding_modules: Optional[list[str]] = None,
        weights_mapper: Optional[WeightsMapper] = None,
    ) -> "OFTModel":
        """Create a OFTModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        ofts: dict[str, OFTLayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_oft_a, is_bias = parse_fine_tuned_oft_name(
                tensor_name, weights_mapper)
            if module_name not in ofts:
                oft_embeddings_tensor = None
                if embeddings:
                    assert embedding_modules is not None
                    embeddings_module = next(
                        (k for k in embedding_modules if k in module_name),
                        None)
                    if embeddings_module:
                        oft_embeddings_tensor = embeddings[
                            embedding_modules[embeddings_module]].to(
                                device=device, dtype=dtype)
                        if pin_memory:
                            oft_embeddings_tensor = (
                                oft_embeddings_tensor.pin_memory())
                ofts[module_name] = OFTLayerWeights.from_config(
                    module_name, peft_helper, oft_embeddings_tensor)

            if is_bias:
                ofts[module_name].bias = tensor.to(device=device, dtype=dtype)
                bias = tensor.to(device=device, dtype=dtype)
                if pin_memory:
                    bias = bias.pin_memory()
                ofts[module_name].bias = bias
            elif is_oft_a:
                ofts[module_name].oft_a = tensor.to(device=device,
                                                      dtype=dtype)
                if pin_memory:
                    ofts[module_name].oft_a = ofts[
                        module_name].oft_a.pin_memory()
            else:
                ofts[module_name].oft_b = tensor.to(device=device,
                                                      dtype=dtype)
                assert embedding_padding_modules is not None
                if any(name in module_name
                       for name in embedding_padding_modules
                       ) and target_embedding_padding is not None:
                    oft_b = ofts[module_name].oft_b
                    assert target_embedding_padding >= oft_b.shape[0]
                    addition = target_embedding_padding - oft_b.shape[0]
                    ofts[module_name].oft_b = torch.nn.functional.pad(
                        oft_b, (0, 0, 0, addition))
                if pin_memory:
                    ofts[module_name].oft_b = ofts[
                        module_name].oft_b.pin_memory()

        for oft in ofts.values():
            oft.optimize()

        return cls(oft_model_id, peft_helper.r, ofts)

    @classmethod
    def from_local_checkpoint(
            cls,
            oft_dir: str,
            expected_oft_modules: list[str],
            peft_helper: PEFTHelper,
            *,
            oft_model_id: Optional[int] = None,
            device: str = "cuda",
            dtype: Optional[torch.dtype] = None,
            target_embedding_padding: Optional[int] = None,
            embedding_modules: Optional[dict[str, str]] = None,
            embedding_padding_modules: Optional[list[str]] = None,
            weights_mapper: Optional[WeightsMapper] = None,
            tensorizer_config_dict: Optional[dict] = None) -> "OFTModel":
        """Create a OFTModel from a local checkpoint.

        Args:
            oft_dir: The local path that has oft data.
            expected_oft_modules: Name of modules that are expected to be
                replaced by oft.
            peft_helper: Loaded oft configuration information.
            oft_model_id: OFT model id. If not given, automatically set by
                a global counter.
            device: Device where the oft model is loaded.
            dtype: dtype of the oft model weights.

        Returns:
            Loaded OFT Model.
        """
        oft_tensor_path = os.path.join(oft_dir, "adapter_model.safetensors")
        oft_bin_file_path = os.path.join(oft_dir, "adapter_model.bin")
        oft_pt_file_path = os.path.join(oft_dir, "adapter_model.pt")
        new_embeddings_tensor_path = os.path.join(
            oft_dir, "new_embeddings.safetensors")
        new_embeddings_bin_file_path = os.path.join(oft_dir,
                                                    "new_embeddings.bin")
        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[Union[list[str], str]] = []

        def check_unexpected_modules(modules: dict):
            for oft_module in modules.keys():  # noqa
                module_name, _, _ = parse_fine_tuned_oft_name(
                    oft_module, weights_mapper)
                part_name = module_name.split(".")[-1]
                if part_name not in expected_oft_modules:
                    unexpected_modules.append(module_name)
            if unexpected_modules:
                raise ValueError(
                    f"While loading {oft_dir}, expected"
                    f" target modules in {expected_oft_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded OFT module is correct")

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            oft_tensor_path = os.path.join(tensorizer_config.tensorizer_dir,
                                            "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                oft_tensor_path,
                dtype=tensorizer_config.dtype,
                **tensorizer_args.deserialization_kwargs)
            check_unexpected_modules(tensors)

        elif os.path.isfile(oft_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # oftified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(oft_tensor_path,
                                       framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(oft_bin_file_path) or os.path.isfile(
                oft_pt_file_path):
            # When a bin/pt file is provided, we rely on config to find
            # unexpected modules.
            unexpected_modules = []
            target_modules = peft_helper.target_modules
            if not isinstance(target_modules, list):
                target_modules = [target_modules]
            for module in target_modules:
                # Compatible with more modules,
                # such as:layers.11.self_attn.k_proj
                part_name = module.split(".")[-1]
                if part_name not in expected_oft_modules:
                    unexpected_modules.append(module)
            # loaded oft's target modules must be a subset of
            # expected_oft_modules. It is not reliable. See
            # https://github.com/vllm-project/vllm/pull/5909. But there's no
            # other better mechanism.
            if unexpected_modules and not is_regex_target_modules(
                    peft_helper.target_modules, expected_oft_modules):
                raise ValueError(
                    f"While loading {oft_dir}, expected"
                    f" target modules in {expected_oft_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded OFT module is correct")
            oft_file_path = (oft_bin_file_path
                              if os.path.isfile(oft_bin_file_path) else
                              oft_pt_file_path)
            tensors = torch.load(oft_file_path,
                                 map_location=device,
                                 weights_only=True)
        else:
            raise ValueError(f"{oft_dir} doesn't contain tensors")

        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(
                new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path,
                                    map_location=device,
                                    weights_only=True)

        return cls.from_oft_tensors(
            oft_model_id=get_oft_id()
            if oft_model_id is None else oft_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            embeddings=embeddings,
            target_embedding_padding=target_embedding_padding,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
            weights_mapper=weights_mapper)


class OFTModelManager:
    """A manager that manages multiple OFT-fine-tuned models."""

    def __init__(
        self,
        model: SupportsOFT,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        oft_config: OFTConfig,
        device: torch.device,
    ):
        """Create a OFTModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            oft_config: the OFT configuration.
        """
        self.model: SupportsOFT = model
        self._registered_adapters: dict[int, OFTModel] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._active_adapters: dict[int, None] = {}
        self.adapter_type = "OFT"
        self.oft_config = oft_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.oft_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.oft_index_to_id: list[Optional[int]] = [None] * self.oft_slots
        self.vocab_size = vocab_size
        self.punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            max_ofts=self.oft_config.max_ofts,
        )

        self.supported_oft_modules = get_supported_oft_modules(self.model)
        assert self.supported_oft_modules, "No supported OFT modules found in"
        f" {self.model.__class__.__name__}."

        self.packed_modules_mapping = get_packed_modules_mapping(self.model)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports OFT for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping"))
        self.is_pooling_model = is_pooling_model(self.model)
        self.is_moe_model = is_moe_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithOFT] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: Optional[OFTMapping] = None
        self._create_oft_modules()
        self.model.oft_manager = self

    def __len__(self) -> int:
        return len(self._registered_adapters)

    @property
    def capacity(self) -> int:
        return self.oft_config.max_cpu_ofts

    @property
    def oft_slots(self) -> int:
        return self.oft_config.max_ofts

    @property
    def adapter_slots(self) -> int:
        return self.oft_slots

    def activate_adapter(
        self,
        oft_id: int,
    ) -> bool:
        """Move OFT into a GPU buffer to be used in the forward pass."""
        if oft_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, oft_id) for i, oft_id in enumerate(self.oft_index_to_id)
             if oft_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free oft slots")
        index, _ = first_free_slot
        self._active_adapters[oft_id] = None
        oft_model = self._registered_adapters[oft_id]
        logger.debug("Activating OFT. int id: %d, slot index: %d",
                     oft_model.id, index)
        self.oft_index_to_id[index] = oft_model.id
        for module_name, module in self.modules.items():
            module_oft = self._get_oft_layer_weights(oft_model, module_name)
            if module_oft:
                module_oft.optimize()
                # Bias is not explicitly enabled with the flag enable_oft_bias.
                bias = module_oft.bias
                if ((torch.is_tensor(bias) or
                     (isinstance(bias, Sequence) and any(b is not None
                                                         for b in bias)))
                        and not self.oft_config.bias_enabled):
                    module_oft.bias = None
                    raise ValueError(
                        f"Adapter bias cannot be used for {module_name}"
                        " without --enable-oft-bias.")
                module.set_oft(index, module_oft.oft_a, module_oft.oft_b,
                                module_oft.embeddings_tensor,
                                module_oft.bias)
            else:
                module.reset_oft(index)
        return True

    def _deactivate_adapter(self, oft_id: int):
        try:
            index = self.oft_index_to_id.index(oft_id)
            self.oft_index_to_id[index] = None
        except ValueError:
            pass

    def _add_adapter(self, oft: OFTModel):
        self._create_merged_ofts_inplace(oft)
        self._registered_adapters[oft.id] = oft

    def pin_adapter(self, oft_id: int) -> bool:
        """Pin a OFTModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in OFTModelManager. "
            "Use LRUCacheOFTModelManager for pinning")  # type: ignore

    def _set_adapter_mapping(self, mapping: OFTMapping) -> None:
        # update oft states
        self.punica_wrapper.update_metadata(
            mapping,
            self.oft_index_to_id,
            self.oft_slots + 1,
            self.vocab_size,
            self.oft_config.oft_extra_vocab_size,
        )

    def remove_all_adapters(self):
        """Remove all OFTModels from the manager."""
        self._registered_adapters.clear()
        self.oft_index_to_id = [None] * self.oft_slots
        self._active_adapters.clear()

    def _create_oft_modules(self):

        def _parent_module(module_name: str) -> str:
            # module name is a dot separated name.
            # for example:
            #  - given an input 'x.y.z' return 'x.y'
            #  - given an input 'x' return ''
            return module_name.rpartition('.')[0]

        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue
            if not self._match_target_modules(module_name):
                continue
            # A temporary approach for multimodal models to support OFT
            # TODO: Remove this restriction
            if self._filter_unsupported_mm_module(module_name):
                logger.warning(
                    "Regarding multimodal models, vLLM currently only supports "
                    "adding OFT to language model, %s will be ignored.",
                    module_name,
                )
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.oft_slots, self.oft_config,
                           packed_moduled_lst, self.model.config))

            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                logits_processor_module_name = 'logits_processor'
                parent_module = _parent_module(module_name)
                if parent_module:
                    logits_processor_module_name = (
                        f"{parent_module}.{logits_processor_module_name}")

                logits_processor_module = self.model.get_submodule(
                    logits_processor_module_name)

                new_module = replace_submodule(
                    self.model, logits_processor_module_name,
                    from_layer_logits_processor(logits_processor_module,
                                                module, self.oft_slots,
                                                self.oft_config,
                                                self.model.config))

            # In some models, especially multimodal ones, layers with the same
            # name may have different types, such as nn.Linear and
            # ReplicatedLinear. The nn.Linear layers cannot be replaced with
            # OFT layers, leading to assertion error. The following check
            # aims to prevent this error
            if self.supports_mm and not isinstance(new_module,
                                                   BaseLayerWithOFT):
                continue
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            # All oft layers share the same punica_wrapper based on reference.
            new_module.set_mapping(self.punica_wrapper)

    def register_module(self, module_name: str, module: "BaseLayerWithOFT"):
        assert isinstance(module, BaseLayerWithOFT)
        self.modules[module_name] = module

    def create_dummy_oft(
            self,
            oft_id: int,
            block_size: int,
            embedding_modules: Optional[dict[str, str]] = None) -> OFTModel:
        """Create zero-initialized OFTModel for warmup."""
        model = OFTModel(oft_id, block_size, {})
        for module_name, module in self.model.named_modules():
            bias_enabled = self.oft_config.bias_enabled
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithOFT)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
                if parts[-1] in embedding_modules:
                    input_dim = (module.base_layer.org_vocab_size +
                                 self.oft_config.oft_extra_vocab_size if
                                 hasattr(module.base_layer, "org_vocab_size")
                                 else module.base_layer.weight.shape[1])
                    output_dim = module.base_layer.embedding_dim if hasattr(
                        module.base_layer,
                        "embedding_dim") else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = (module.base_layer.embedding_dim if
                                             hasattr(module.base_layer,
                                                     "embedding_dim") else
                                             module.base_layer.weight.shape[1])
                    oft = OFTLayerWeights.create_dummy_oft_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        block_size,
                        module.oft_a_stacked[0].dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim,
                        bias_enabled=bias_enabled)
                else:
                    oft = OFTLayerWeights.create_dummy_oft_weights(
                        module_name,
                        module.oft_a_stacked[0].shape[-1],
                        module.oft_b_stacked[0].shape[-2],
                        block_size,
                        module.oft_a_stacked[0].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
                    )
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subofts: list[Optional[OFTLayerWeights]] = []
                for i, r in enumerate(replacements):
                    oft = OFTLayerWeights.create_dummy_oft_weights(
                        module_name + "." + r,
                        module.oft_a_stacked[i].shape[-1],
                        module.oft_b_stacked[i].shape[-2],
                        block_size,
                        module.oft_a_stacked[i].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
                    )
                    subofts.append(oft)
                oft = PackedOFTLayerWeights.pack(subofts)
            model.ofts[module_name] = oft
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_oft_modules)

    def _filter_unsupported_mm_module(self, module_name: str) -> bool:
        """
        Regarding multimodal models, vLLM currently only supports adding OFT to
        language model. OFT for other modules, such as the vision tower, will
        be filtered out.
        """
        if self.supports_mm:
            module_mapping: MultiModelKeys = self.model.get_mm_mapping()
            prefix_lst = module_mapping.connector + module_mapping.tower_model
            return any(
                [module_name.startswith(prefix) for prefix in prefix_lst])
        return False

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name, [])
        # When replacements is less than or equal to 1, it indicates that this
        # module is not a packed module.
        if len(replacements) <= 1:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_ofts_inplace(self, oft_model: OFTModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_ofts: list[Optional[OFTLayerWeights]] = []
            replaced_module: set[str] = set()
            has_replacement = False
            for r in new_module_names:
                oft = self._get_oft_layer_weights(oft_model, r)
                replacement_ofts.append(oft)
                if oft:
                    has_replacement = True
                    replaced_module.add(r)
            if not has_replacement:
                continue
            for i in range(len(replacement_ofts)):
                if replacement_ofts[i]:
                    continue
                replacement_ofts[i] = None
            # HACK Temporary solution for the pool model.
            if self.is_pooling_model and not oft_model.check_oft_name(
                    module_name):
                replaced_module_name = module_name.replace("model.", "")
                if oft_model.check_oft_name(module_name):
                    module_name = replaced_module_name
            oft_model.ofts[module_name] = PackedOFTLayerWeights.pack(
                replacement_ofts)
            # Remove the modules that have been replaced.
            for module in replaced_module:
                oft_model.ofts.pop(module, None)

    def _get_oft_layer_weights(
            self, oft_model: OFTModel,
            module_name: str) -> Optional[OFTLayerWeights]:
        org_module_name = module_name
        if self.is_pooling_model and not oft_model.check_oft_name(
                module_name):
            # If it's a pool model, and the layer name is not found,
            # remove the prefix 'model.' and search again.
            module_name = module_name.replace("model.", "")
            if oft_model.check_oft_name(module_name):
                org_module_name = module_name
                logger.info_once(
                    "For the pool model, successfully loaded the OFT weights "
                    "after removing the prefix 'model.'.")
        return oft_model.get_oft(org_module_name)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self._active_adapters:
            return False
        self._deactivate_adapter(adapter_id)
        self._active_adapters.pop(adapter_id, None)
        return True

    def add_adapter(self, adapter: OFTModel) -> bool:
        logger.debug("Adding oft. Model id: %d, "
                     "int id: %d", adapter.id, adapter.id)
        if adapter.id in self._registered_adapters:
            return False
        if len(self._registered_adapters) >= self.capacity:
            raise RuntimeError("No free adapter slots.")
        self._add_adapter(adapter)
        return True

    def set_adapter_mapping(self, mapping: OFTMapping) -> None:
        if self._last_mapping != mapping:
            self._set_adapter_mapping(mapping)
            self._last_mapping = mapping

    def remove_adapter(self, adapter_id: int) -> bool:
        self.deactivate_adapter(adapter_id)
        if adapter_id not in self._registered_adapters:
            return False
        self._registered_adapters.pop(adapter_id, None)
        return True

    def list_adapters(self) -> dict[int, OFTModel]:
        return dict(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[OFTModel]:
        return self._registered_adapters.get(adapter_id)


class OFTLRUCache(AdapterLRUCache[OFTModel]):

    def __init__(self, capacity: int, deactivate_oft_fn: Callable[[int],
                                                                   bool]):
        super().__init__(capacity, deactivate_oft_fn)


class LRUCacheOFTModelManager(OFTModelManager):
    """A model manager that manages multiple OFTs with LRU cache."""

    def __init__(self, model: nn.Module, max_num_seqs: int,
                 max_num_batched_tokens: int, vocab_size: int,
                 oft_config: OFTConfig, device: torch.device):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, oft_config, device)
        self._registered_adapters: OFTLRUCache = OFTLRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters: OFTLRUCache = OFTLRUCache(
            self.oft_slots, self._deactivate_adapter)

    def list_adapters(self) -> dict[int, OFTModel]:
        """List all registered OFTModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, oft: OFTModel) -> bool:
        """Add a OFTModel to the manager."""
        logger.debug("Adding oft. Model id: %d, "
                     "int id: %d", oft.id, oft.id)
        if oft.id not in self._registered_adapters:
            self._add_adapter(oft)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(oft.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        oft_id: int,
    ) -> bool:
        if oft_id not in self._active_adapters and len(
                self._active_adapters) >= self.oft_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(oft_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(oft_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, oft_id: int) -> bool:
        """Pin a OFTModel in the manager cache."""
        self._pin_oft_in_cpu_cache(oft_id)
        self._pin_oft_in_gpu_cache(oft_id)
        return True

    def _pin_oft_in_cpu_cache(self, oft_id: int):
        try:
            self._registered_adapters.pin(oft_id)
        except ValueError as err:
            raise ValueError("Pinning failed. "
                             f"OFT {oft_id} is not registered.") from err

    def _pin_oft_in_gpu_cache(self, oft_id: int):
        if oft_id not in self._active_adapters:
            # move oft to gpu if not already active
            self.activate_adapter(oft_id)

        self._active_adapters.pin(oft_id)


def create_oft_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        oft_config: OFTConfig,
        device: torch.device,
        oft_manager_cls: type[OFTModelManager] = OFTModelManager,
        **kwargs) -> OFTModelManager:
    """Create a OFT adapter for a given model."""
    if not isinstance(model, SupportsOFT):
        raise ValueError(f"Model {type(model)} is not supported for OFT.")
    oft_manager = oft_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        oft_config=oft_config,
        device=device,
        **kwargs)
    return oft_manager
