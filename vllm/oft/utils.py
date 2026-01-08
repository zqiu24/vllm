# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional, Union

import huggingface_hub
import regex as re
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config.oft import OFTConfig
from vllm.logger import init_logger
# being imported for _all_oft_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.oft.layers import (BaseLayerWithOFT, ColumnParallelLinearWithOFT,
                              ColumnParallelLinearWithShardedOFT,
                              LogitsProcessorWithOFT,
                              MergedColumnParallelLinearWithOFT,
                              MergedColumnParallelLinearWithShardedOFT,
                              MergedQKVParallelLinearWithOFT,
                              MergedQKVParallelLinearWithShardedOFT,
                              QKVParallelLinearWithOFT,
                              QKVParallelLinearWithShardedOFT,
                              ReplicatedLinearWithOFT,
                              RowParallelLinearWithOFT,
                              RowParallelLinearWithShardedOFT,
                              VocabParallelEmbeddingWithOFT)
from vllm.model_executor.layers.linear import LinearBase

# yapf: enable

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_all_oft_classes: set[type[BaseLayerWithOFT]] = {
    VocabParallelEmbeddingWithOFT,
    ColumnParallelLinearWithOFT,
    MergedColumnParallelLinearWithOFT,
    QKVParallelLinearWithOFT,
    MergedQKVParallelLinearWithOFT,
    RowParallelLinearWithOFT,
    ReplicatedLinearWithOFT,
    LogitsProcessorWithOFT,
    ColumnParallelLinearWithShardedOFT,
    QKVParallelLinearWithShardedOFT,
    MergedColumnParallelLinearWithShardedOFT,
    MergedQKVParallelLinearWithShardedOFT,
    RowParallelLinearWithShardedOFT,
}


def from_layer(layer: nn.Module,
               max_ofts: int,
               oft_config: OFTConfig,
               packed_modules_list: list,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for oft_cls in _all_oft_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if oft_cls.can_replace_layer(source_layer=layer,
                                      oft_config=oft_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            instance_layer = oft_cls(layer)
            instance_layer.create_oft_weights(max_ofts, oft_config,
                                               model_config)
            return instance_layer
    return layer


def from_layer_logits_processor(
    layer: "LogitsProcessor",
    lm_head: "ParallelLMHead",
    max_ofts: int,
    oft_config: OFTConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithOFT:
    ret = LogitsProcessorWithOFT(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_oft_weights(max_ofts, oft_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_oft_name(
    name: str,
    weights_mapper: Optional["WeightsMapper"] = None
) -> tuple[str, bool, bool]:
    """Parse the name of oft weights.

    args:
        name: the name of the fine-tuned OFT, e.g.
            base_model.model.dense1.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
        tuple(module_name, is_oft_a):
            module_name: the name of the module, e.g. model.dense1,
            is_oft_r whether the tensor is oft_r.
            is_bias whether the tensor is oft bias.
    """

    # OFT weight qualified name usually starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if name.startswith("base_model.model."):
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        # recover the prefix `base_model.model.`
        name = "base_model.model." + name
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # In some situations, we may not start with `base_model.model.`.
    # If we don't (e.g., ibm-granite/granite-speech-3.3-8b),
    # we should keep the prefix intact.
    start_index = 2 if name.startswith("base_model.model.") else 0

    parts = name.split(".")
    if parts[-1] == "weight" and parts[-2] == "oft_R":
        new_name = ".".join(parts[start_index:-2])
        return new_name, parts[-2] == "oft_R", False

    if parts[-1] == "oft_embedding_R":
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "oft_embedding_R", False

    if parts[-1] == "bias":
        new_name = ".".join(parts[start_index:-2])
        return new_name, False, True

    raise ValueError(f"{name} is unsupported OFT weight")


def is_regex_target_modules(load_modules: Union[str, list[str]],
                            expected_oft_modules: list[str]) -> bool:
    """
    PEFT supports passing `target_modules` in the form of regular expressions, 
    such as `model.*(q_proj|k_proj|v_proj)$`. This function is mainly used to 
    determine whether the suffix in the regular expression is present in the 
    `expected_oft_modules`.
    """

    def is_valid_regex(pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def is_subset(sub_list, full_list):
        return set(sub_list).issubset(set(full_list))

    # Similar to PEFT's processing logic, regex-related operations are only
    #  executed when the load_modules is a `str`.
    if not isinstance(load_modules, str):
        return False

    if is_valid_regex(load_modules):
        match = re.search(r"\((.*?)\)\$?$", load_modules)
        if match:
            suffix = match.group(1).split("|")
            return is_subset(suffix, expected_oft_modules)
    return False


def get_supported_oft_modules(model: nn.Module) -> list[str]:
    """
    In vLLM, all linear layers support OFT.
    """

    supported_oft_modules: set[str] = set()
    for name, module in model.named_modules():
        # get the embedding modules if the module's embedding_modules
        # is not empty.
        embedding_modules = getattr(module, "embedding_modules", None)
        if embedding_modules is not None:
            for name in embedding_modules:
                supported_oft_modules.add(name)

        # get all the linear subfixes.
        if isinstance(module, (LinearBase, )):
            supported_oft_modules.add(name.split(".")[-1])

    return list(supported_oft_modules)


def get_adapter_absolute_path(oft_path: str) -> str:
    """
    Resolves the given oft_path to an absolute local path.

    If the oft_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the oft_path as a local file path and
    converts it to an absolute path.

    Parameters:
    oft_path (str): The path to the oft model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the oft model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(oft_path):
        return oft_path

    # If the path starts with ~, expand the user home directory.
    if oft_path.startswith('~'):
        return os.path.expanduser(oft_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(oft_path):
        return os.path.abspath(oft_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(
            repo_id=oft_path)
    except (HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError,
            HFValidationError):
        # Handle errors that may occur during the download
        # Return original path instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return oft_path

    return local_snapshot_path
